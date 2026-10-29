# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import threading
import time
import traceback
from typing import TYPE_CHECKING, cast

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
import requests
import torch

from tests.models.utils import check_logprobs_close
from tests.utils import VLLM_PATH, RemoteOpenAIServer
from tests.v1.e2e.general.accuracy_utils import AccuracyTolerance, check_accuracy_budget
from tests.v1.utils import (
    check_request_balancing,
    get_engine_request_counts,
    get_prometheus_metrics,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.worker.cpu_worker import CPUWorker

MODEL_NAME = "ibm-research/PowerMoE-3b"

# Number of data parallel ranks for multi-node internal LB testing
DP_SIZE = int(os.getenv("DP_SIZE", "2"))
# Default tensor parallel size to use
TP_SIZE = int(os.getenv("TP_SIZE", "1"))

# Number of nodes to simulate
NUM_NODES = 2


async def _make_completion_request(
    client: openai.AsyncOpenAI,
    model_name: str,
) -> openai.types.Completion:
    """Make a single completion request and validate the response.

    Uses temperature=1.0 to ensure diverse outputs across concurrent
    requests for realistic load balancer testing.
    """
    completion = await client.completions.create(
        model=model_name,
        prompt="Hello, my name is",
        max_tokens=5,
        temperature=1.0,
    )

    assert completion.id is not None, (
        f"Expected non-None completion id. usage={completion.usage!r}"
    )
    assert completion.choices is not None and len(completion.choices) == 1, (
        f"Expected 1 choice, got "
        f"{len(completion.choices) if completion.choices else 'None'}"
    )

    choice = completion.choices[0]
    # With temperature=1.0, the model may emit a stop token immediately,
    # producing empty text with finish_reason='stop'. This is valid
    # model behavior - the test's purpose is load balancing, not output
    # quality.
    assert choice.finish_reason in ("length", "stop"), (
        f"Expected finish_reason 'length' or 'stop', "
        f"got {choice.finish_reason!r}. text={choice.text!r}"
    )
    if choice.finish_reason == "length":
        assert len(choice.text) >= 1, (
            f"Expected non-empty text with finish_reason='length', got {choice.text!r}"
        )

    assert completion.usage.prompt_tokens > 0, (
        f"Expected positive prompt_tokens, got {completion.usage.prompt_tokens}"
    )
    assert completion.usage.total_tokens > 0, (
        f"Expected positive total_tokens, got {completion.usage.total_tokens}"
    )
    return completion


async def _run_request_bursts(
    client: openai.AsyncOpenAI,
    model_name: str,
    num_requests: int = 200,
    num_bursts: int = 2,
):
    """Send multiple bursts of completion requests and validate all succeed."""
    for burst in range(num_bursts):
        all_tasks = []
        for _ in range(num_requests):
            all_tasks.append(
                asyncio.create_task(_make_completion_request(client, model_name))
            )
            await asyncio.sleep(0.01)

        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        assert len(results) == num_requests, (
            f"Burst {burst}: expected {num_requests} results, got {len(results)}"
        )

        for result in results:
            if isinstance(result, BaseException):
                raise result

        assert all(completion is not None for completion in results), (
            f"Burst {burst}: some completions were None"
        )

        await asyncio.sleep(0.5)


class MultinodeInternalLBServerManager:
    """Manages multi-node data parallel vLLM server instances for internal
    load balancer testing using --headless mode."""

    def __init__(
        self,
        model_name: str,
        dp_size: int,
        api_server_count: int,
        base_server_args: list,
        dp_per_node: int = 1,
        tp_size: int = TP_SIZE,
    ):
        self.model_name = model_name
        self.dp_size = dp_size
        self.dp_per_node = dp_per_node
        self.tp_size = tp_size
        self.api_server_count = api_server_count
        self.base_server_args = base_server_args
        self.servers: list[tuple[RemoteOpenAIServer, list[str]] | None] = [None] * (
            dp_size // dp_per_node
        )
        self.server_threads: list[threading.Thread] = []

    def __enter__(self) -> list[tuple[RemoteOpenAIServer, list[str]]]:
        """Start all server instances for multi-node internal LB mode."""
        for server_idx, rank in enumerate(range(0, self.dp_size, self.dp_per_node)):
            # Create server args for this specific rank
            server_args = self.base_server_args.copy()

            if rank == 0:
                # Head node - runs API server and first DP rank
                server_args.extend(
                    [
                        "--data-parallel-size",
                        str(self.dp_size),
                        "--data-parallel-size-local",
                        str(self.dp_per_node),
                        "--tensor-parallel-size",
                        str(self.tp_size),
                        "--port",
                        "8000",  # Single endpoint for all requests
                        "--api-server-count",
                        str(self.api_server_count),
                        "--data-parallel-address",
                        "127.0.0.1",
                        "--data-parallel-rpc-port",
                        "13345",
                    ]
                )
            else:
                # Secondary nodes - run in headless mode
                server_args.extend(
                    [
                        "--headless",
                        "--data-parallel-size",
                        str(self.dp_size),
                        "--data-parallel-size-local",
                        str(self.dp_per_node),
                        "--data-parallel-start-rank",
                        str(rank),
                        "--tensor-parallel-size",
                        str(self.tp_size),
                        "--data-parallel-address",
                        "127.0.0.1",
                        "--data-parallel-rpc-port",
                        "13345",
                    ]
                )

            # Use a thread to start each server to allow parallel initialization
            def start_server(sidx: int, r: int, sargs: list[str]):
                gpus_per_node = self.tp_size * self.dp_per_node
                try:
                    # Start the server
                    server = RemoteOpenAIServer(
                        self.model_name,
                        sargs,
                        auto_port=False,
                        max_wait_seconds=int(
                            os.getenv("VLLM_ENGINE_READY_TIMEOUT_S", "480")
                        ),
                        env_dict={
                            "VLLM_SERVER_DEV_MODE": "1",
                            current_platform.device_control_env_var: ",".join(
                                str(current_platform.device_id_to_physical_device_id(i))
                                for i in range(r, r + gpus_per_node)
                            ),
                        },
                    )
                    server.__enter__()
                    if r == 0:
                        print(
                            f"Head node (rank {r}) started successfully with "
                            f"{self.api_server_count} API servers"
                        )
                    else:
                        print(f"Headless node (rank {r}) started successfully")
                    self.servers[sidx] = (server, sargs)
                except Exception as e:
                    print(f"Failed to start server rank {r}: {e}")
                    traceback.print_exc()
                    raise

            thread = threading.Thread(
                target=start_server, args=(server_idx, rank, server_args)
            )
            thread.start()

            self.server_threads.append(thread)

        # Wait for all servers to start
        for thread in self.server_threads:
            thread.join()

        # Give servers additional time to fully initialize and coordinate
        time.sleep(3)

        if not all(self.servers):
            raise Exception("Servers failed to start")

        return cast(list[tuple[RemoteOpenAIServer, list[str]]], self.servers)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop all server instances."""
        servers = [entry[0] for entry in self.servers if entry is not None]
        self.servers.clear()
        try:
            RemoteOpenAIServer.shutdown_many(servers)
        except Exception as e:
            print(f"Error stopping servers: {e}")
            traceback.print_exc()


class APIOnlyServerManager:
    """Manages API-only server (Node 0) and headless engines server (Node 1)
    for testing separated API server and engine configuration."""

    def __init__(
        self,
        model_name: str,
        dp_size: int,
        api_server_count: int,
        base_server_args: list,
        tp_size: int = TP_SIZE,
    ):
        self.model_name = model_name
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.api_server_count = api_server_count
        self.base_server_args = base_server_args
        self.servers: list[tuple[RemoteOpenAIServer, list[str]] | None] = [None] * 2
        self.server_threads: list[threading.Thread] = []

    def __enter__(self) -> list[tuple[RemoteOpenAIServer, list[str]]]:
        """Start API-only server and headless engines server."""
        # Start API-only server (Node 0) - no engines, only API server
        api_server_args = self.base_server_args.copy()
        api_server_args.extend(
            [
                "--data-parallel-size",
                str(self.dp_size),
                "--data-parallel-size-local",
                "0",  # No engines on this node
                "--tensor-parallel-size",
                str(self.tp_size),
                "--port",
                "8000",
                "--api-server-count",
                str(self.api_server_count),
                "--data-parallel-address",
                "127.0.0.1",
                "--data-parallel-rpc-port",
                "13345",
            ]
        )

        # Start headless engines server (Node 1) - all engines, no API server
        engines_server_args = self.base_server_args.copy()
        engines_server_args.extend(
            [
                "--headless",
                "--data-parallel-size",
                str(self.dp_size),
                "--data-parallel-size-local",
                str(self.dp_size),  # All engines on this node
                "--tensor-parallel-size",
                str(self.tp_size),
                "--data-parallel-address",
                "127.0.0.1",
                "--data-parallel-rpc-port",
                "13345",
            ]
        )

        # Use threads to start both servers in parallel
        def start_api_server():
            try:
                server = RemoteOpenAIServer(
                    self.model_name,
                    api_server_args,
                    auto_port=False,
                    env_dict={
                        "VLLM_SERVER_DEV_MODE": "1",
                        # No GPUs needed for API-only server
                    },
                )
                server.__enter__()
                print(
                    f"API-only server started successfully with "
                    f"{self.api_server_count} API servers"
                )
                self.servers[0] = (server, api_server_args)
            except Exception as e:
                print(f"Failed to start API-only server: {e}")
                raise

        def start_engines_server():
            try:
                server = RemoteOpenAIServer(
                    self.model_name,
                    engines_server_args,
                    auto_port=False,
                    env_dict={
                        current_platform.device_control_env_var: ",".join(
                            str(current_platform.device_id_to_physical_device_id(i))
                            for i in range(self.dp_size * self.tp_size)
                        ),
                    },
                )
                server.__enter__()
                print(
                    f"Headless engines server started successfully with "
                    f"{self.dp_size} engines"
                )
                self.servers[1] = (server, engines_server_args)
            except Exception as e:
                print(f"Failed to start headless engines server: {e}")
                raise

        # Start API server first
        api_thread = threading.Thread(target=start_api_server)
        api_thread.start()
        self.server_threads.append(api_thread)

        # Start engines server second
        engines_thread = threading.Thread(target=start_engines_server)
        engines_thread.start()
        self.server_threads.append(engines_thread)

        # Wait for both servers to start
        for thread in self.server_threads:
            thread.join()

        # Give servers additional time to fully initialize and coordinate
        time.sleep(3)

        if not all(self.servers):
            raise Exception("Both servers failed to start")

        return cast(list[tuple[RemoteOpenAIServer, list[str]]], self.servers)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop both server instances."""
        servers = [entry[0] for entry in self.servers if entry is not None]
        self.servers.clear()
        try:
            RemoteOpenAIServer.shutdown_many(servers)
        except Exception as e:
            print(f"Error stopping servers: {e}")
            traceback.print_exc()


@pytest.fixture(scope="module")
def default_server_args():
    return [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
    ]


@pytest.fixture(scope="module", params=[1, 4])
def server_manager(request, default_server_args):
    api_server_count = request.param
    server_manager = MultinodeInternalLBServerManager(
        MODEL_NAME,
        DP_SIZE,
        api_server_count,
        default_server_args,
        DP_SIZE // NUM_NODES,
        TP_SIZE,
    )

    with server_manager:
        yield server_manager


@pytest.fixture
def servers(server_manager):
    return server_manager.servers


@pytest.fixture(scope="module", params=[1, 4])
def api_only_servers(request, default_server_args):
    """Fixture for API-only server + headless engines configuration."""
    api_server_count = request.param
    with APIOnlyServerManager(
        MODEL_NAME, DP_SIZE, api_server_count, default_server_args, TP_SIZE
    ) as server_list:
        yield server_list


@pytest_asyncio.fixture
async def client(servers: list[tuple[RemoteOpenAIServer, list[str]]]):
    # For internal LB, we only connect to the head node (rank 0)
    # which provides the single API endpoint
    head_server = servers[0][0]
    async with head_server.get_async_client() as client:
        yield client


@pytest_asyncio.fixture
async def api_only_client(api_only_servers: list[tuple[RemoteOpenAIServer, list[str]]]):
    """Client fixture for API-only server configuration."""
    # Connect to the API-only server (first server in the list)
    api_server = api_only_servers[0][0]
    async with api_server.get_async_client() as client:
        yield client


def _get_parallel_config(server: RemoteOpenAIServer):
    response = requests.get(server.url_for("server_info?config_format=json"))
    response.raise_for_status()

    vllm_config = response.json()["vllm_config"]
    return vllm_config["parallel_config"]


def test_multinode_dp_server_info(server_manager):
    head_server = server_manager.servers[0][0]
    api_server_count = server_manager.api_server_count

    # Each request will hit one of the API servers
    # `n_reqs` is set so that there is a good chance each server
    # receives at least one request
    n_reqs = 2 * api_server_count * api_server_count
    parallel_configs = [_get_parallel_config(head_server) for _ in range(n_reqs)]
    api_process_counts = [c["_api_process_count"] for c in parallel_configs]
    api_process_ranks = [c["_api_process_rank"] for c in parallel_configs]

    assert all(c == api_server_count for c in api_process_counts), api_process_counts
    assert all(0 <= r < api_server_count for r in api_process_ranks), api_process_ranks


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_multinode_dp_completion(
    client: openai.AsyncOpenAI,
    servers: list[tuple[RemoteOpenAIServer, list[str]]],
    model_name: str,
) -> None:
    # Test single request
    result = await _make_completion_request(client, model_name)
    assert result is not None
    print("Multi-node internal LB handled single completion request successfully")

    await asyncio.sleep(0.5)

    # Send multiple bursts - internal LB should distribute across DP ranks
    await _run_request_bursts(client, model_name)

    _, server_args = servers[0]
    api_server_count = (
        server_args.count("--api-server-count")
        and server_args[server_args.index("--api-server-count") + 1]
        or 1
    )
    print(
        f"Successfully completed multi-node internal LB test with "
        f"{len(servers)} DP ranks (API server count: {api_server_count})"
    )

    # Check request balancing via Prometheus metrics
    head_server = servers[0][0]
    check_request_balancing(head_server, DP_SIZE)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_multinode_dp_completion_streaming(
    client: openai.AsyncOpenAI,
    servers: list[tuple[RemoteOpenAIServer, list[str]]],
    model_name: str,
) -> None:
    prompt = "What is an LLM?"

    async def make_streaming_request():
        # Perform a non-streaming request to get the expected full output
        single_completion = await client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=5,
            temperature=0.0,
        )
        single_output = single_completion.choices[0].text

        # Perform the streaming request
        stream = await client.completions.create(
            model=model_name, prompt=prompt, max_tokens=5, temperature=0.0, stream=True
        )
        chunks: list[str] = []
        finish_reason_count = 0
        last_chunk = None
        async for chunk in stream:
            chunks.append(chunk.choices[0].text)
            if chunk.choices[0].finish_reason is not None:
                finish_reason_count += 1
            last_chunk = chunk  # Keep track of the last chunk

        # finish reason should only return in the last block for OpenAI API
        assert finish_reason_count == 1, "Finish reason should appear exactly once."
        assert last_chunk is not None, "Stream should have yielded at least one chunk."
        assert last_chunk.choices[0].finish_reason == "length", (
            "Finish reason should be 'length'."
        )
        # Check that the combined text matches the non-streamed version.
        assert "".join(chunks) == single_output, (
            "Streamed output should match non-streamed output."
        )
        return True  # Indicate success for this request

    # Test single streaming request
    result = await make_streaming_request()
    assert result is not None
    print("Multi-node internal LB handled single streaming request successfully")

    await asyncio.sleep(0.5)

    # Send multiple streaming requests - internal LB should distribute across
    # DP ranks
    num_requests = 200
    all_tasks = []
    for _ in range(num_requests):
        all_tasks.append(asyncio.create_task(make_streaming_request()))
        await asyncio.sleep(0.01)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests
    assert all(results), "Not all streaming requests completed successfully."

    await asyncio.sleep(0.5)

    # Second burst of streaming requests
    all_tasks = []
    for _ in range(num_requests):
        all_tasks.append(asyncio.create_task(make_streaming_request()))
        await asyncio.sleep(0.01)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests
    assert all(results), "Not all streaming requests completed successfully."

    _, server_args = servers[0]
    api_server_count = (
        server_args.count("--api-server-count")
        and server_args[server_args.index("--api-server-count") + 1]
        or 1
    )
    print(
        f"Successfully completed multi-node internal LB streaming test with "
        f"{len(servers)} DP ranks (API server count: {api_server_count})"
    )

    # Check request balancing via Prometheus metrics
    head_server = servers[0][0]
    check_request_balancing(head_server, DP_SIZE)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_api_only_multinode_dp_completion(
    api_only_client: openai.AsyncOpenAI,
    api_only_servers: list[tuple[RemoteOpenAIServer, list[str]]],
    model_name: str,
) -> None:
    """Test API-only server with all engines on separate headless server."""
    # Test single request
    result = await _make_completion_request(api_only_client, model_name)
    assert result is not None
    print("API-only server handled single completion request successfully")

    await asyncio.sleep(0.5)

    # Send multiple bursts - should be distributed across engines on
    # headless server
    await _run_request_bursts(api_only_client, model_name)

    api_server, api_server_args = api_only_servers[0]
    api_server_count = (
        api_server_args.count("--api-server-count")
        and api_server_args[api_server_args.index("--api-server-count") + 1]
        or 1
    )
    print(
        f"Successfully completed API-only multi-node test with {DP_SIZE} "
        f"engines on headless server (API server count: {api_server_count})"
    )

    # Check request balancing via Prometheus metrics
    check_request_balancing(api_server, DP_SIZE)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_api_only_multinode_dp_completion_streaming(
    api_only_client: openai.AsyncOpenAI,
    api_only_servers: list[tuple[RemoteOpenAIServer, list[str]]],
    model_name: str,
) -> None:
    """Test API-only server streaming with all engines on separate
    headless server."""
    prompt = "What is an LLM?"

    async def make_streaming_request():
        # Perform a non-streaming request to get the expected full output
        single_completion = await api_only_client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=5,
            temperature=0.0,
        )
        single_output = single_completion.choices[0].text

        # Perform the streaming request
        stream = await api_only_client.completions.create(
            model=model_name, prompt=prompt, max_tokens=5, temperature=0.0, stream=True
        )
        chunks: list[str] = []
        finish_reason_count = 0
        last_chunk = None
        async for chunk in stream:
            chunks.append(chunk.choices[0].text)
            if chunk.choices[0].finish_reason is not None:
                finish_reason_count += 1
            last_chunk = chunk  # Keep track of the last chunk

        # finish reason should only return in the last block for OpenAI API
        assert finish_reason_count == 1, "Finish reason should appear exactly once."
        assert last_chunk is not None, "Stream should have yielded at least one chunk."
        assert last_chunk.choices[0].finish_reason == "length", (
            "Finish reason should be 'length'."
        )
        # Check that the combined text matches the non-streamed version.
        assert "".join(chunks) == single_output, (
            "Streamed output should match non-streamed output."
        )
        return True  # Indicate success for this request

    # Test single streaming request
    result = await make_streaming_request()
    assert result is not None
    print("API-only server handled single streaming request successfully")

    await asyncio.sleep(0.5)

    # Send multiple streaming requests - should be distributed across engines
    num_requests = 200
    all_tasks = []
    for _ in range(num_requests):
        all_tasks.append(asyncio.create_task(make_streaming_request()))
        await asyncio.sleep(0.01)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests
    assert all(results), "Not all streaming requests completed successfully."

    await asyncio.sleep(0.5)

    # Second burst of streaming requests
    all_tasks = []
    for _ in range(num_requests):
        all_tasks.append(asyncio.create_task(make_streaming_request()))
        await asyncio.sleep(0.01)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests
    assert all(results), "Not all streaming requests completed successfully."

    _, api_server_args = api_only_servers[0]
    api_server_count = (
        api_server_args.count("--api-server-count")
        and api_server_args[api_server_args.index("--api-server-count") + 1]
        or 1
    )
    print(
        f"Successfully completed API-only streaming test with {DP_SIZE} "
        f"engines on headless server (API server count: {api_server_count})"
    )

    # Check request balancing via Prometheus metrics
    api_server = api_only_servers[0][0]
    check_request_balancing(api_server, DP_SIZE)


class CpuEpInspection:
    """Read loaded expert ownership through the supported worker extension API."""

    def cpu_ep_expert_shards(self):
        import torch.distributed as dist

        from vllm.distributed import get_ep_group
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.triton_utils import HAS_TRITON

        worker = cast("CPUWorker", self)
        model = worker.get_model()
        layers = []
        for layer in model.modules():
            if isinstance(layer, RoutedExperts):
                expert_map = layer.expert_map
                assert expert_map is not None
                layers.append(
                    {
                        "global": layer.global_num_experts,
                        "local": layer.local_num_experts,
                        "weights": layer.w13_weight.shape[0],
                        "map": expert_map.tolist(),
                    }
                )
        group = get_ep_group()
        local = {
            "rank": group.rank_in_group,
            "runner": (
                f"{type(worker.model_runner).__module__}."
                f"{type(worker.model_runner).__qualname__}"
            ),
            "use_v2_model_runner": worker.use_v2_model_runner,
            "has_triton": HAS_TRITON,
            "group": group.unique_name,
            "compile_mode": worker.compilation_config.mode.name,
            "compile_backend": worker.compilation_config.backend,
            "cache_dir": worker.compilation_config.local_cache_dir,
            "aot_cache_loaded": any(
                getattr(layer, "was_aot_compile_fn_loaded_from_disk", False)
                for layer in model.modules()
            ),
            "layers": layers,
        }
        shards = [None] * group.world_size
        dist.all_gather_object(shards, local, group=group.cpu_group)
        return shards


async def _cpu_ep_completion(client, model, prompt, max_tokens, started=None):
    response = await client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0,
        logprobs=5,
        stream=started is not None,
        extra_body={"ignore_eos": True},
    )
    if started is None:
        choice = response.choices[0]
        assert choice.logprobs is not None
        assert len(choice.logprobs.tokens) == max_tokens
        return choice.logprobs.tokens, choice.text, choice.logprobs.top_logprobs
    tokens, text, logprobs = [], [], []
    async for chunk in response:
        choice = chunk.choices[0]
        text.append(choice.text)
        if choice.logprobs is not None:
            tokens.extend(choice.logprobs.tokens)
            logprobs.extend(choice.logprobs.top_logprobs)
            if choice.logprobs.tokens:
                started.set()
    assert len(tokens) == max_tokens
    return tokens, "".join(text), logprobs


async def _wait_cpu_ep_request_state(server, *, active):
    async with asyncio.timeout(30):
        while True:
            metrics = await asyncio.to_thread(get_prometheus_metrics, server)
            running = list(metrics.get("vllm:num_requests_running", {}).values())
            waiting = list(metrics.get("vllm:num_requests_waiting", {}).values())
            expected = [0, 1] if active else [0, 0]
            if sorted(running) == expected and len(waiting) == 2 and not any(waiting):
                return
            await asyncio.sleep(0.05)


async def _cpu_ep_score_prefix(client, model, prefix):
    response = await client.completions.create(
        model=model,
        prompt=prefix,
        max_tokens=0,
        echo=True,
        logprobs=5,
        temperature=0,
    )
    logprobs = response.choices[0].logprobs
    assert logprobs is not None
    assert len(logprobs.tokens) == len(logprobs.token_logprobs) > 1
    assert logprobs.token_logprobs[0] is None
    assert all(value is not None for value in logprobs.token_logprobs[1:])
    values = torch.tensor(logprobs.token_logprobs[1:], dtype=torch.float32)
    assert torch.isfinite(values).all()
    return logprobs.tokens[1:], values


def _check_cpu_ep_scores(reference, actual, label):
    tolerance = AccuracyTolerance(logprob_atol=0.5, greedy_atol=0.25)
    assert len(reference) == len(actual)
    for index, ((reference_tokens, expected), (tokens, values)) in enumerate(
        zip(reference, actual)
    ):
        assert tokens == reference_tokens
        delta = (values - expected).abs()
        bound = tolerance.logprob_error(expected)
        print(
            "CPU_EP_LOGPROB_DELTA",
            {
                "phase": label,
                "prefix": index,
                "tokens": len(tokens),
                "absolute_bound": tolerance.logprob_atol,
                "mean_bound": tolerance.logprob_mean_atol,
                "max_abs_delta": delta.max().item(),
                "mean_abs_delta": delta.mean().item(),
                "max_tolerance_ratio": (delta / bound).max().item(),
            },
            flush=True,
        )
        assert (delta <= bound).all(), f"{label}, prefix={index}: {delta}"
        check_accuracy_budget([delta], [], tolerance, f"{label}, prefix={index}")


def _check_cpu_ep_generated_prefix(reference, actual, label):
    for baseline, output in zip(reference, actual):
        shared = 0
        for expected_token, token in zip(baseline[0], output[0]):
            if expected_token != token:
                break
            shared += 1
        if shared:
            tokens = baseline[0][:shared]
            expected = torch.tensor(
                [scores[token] for token, scores in zip(tokens, baseline[2][:shared])]
            )
            values = torch.tensor(
                [scores[token] for token, scores in zip(tokens, output[2][:shared])]
            )
            _check_cpu_ep_scores([(tokens, expected)], [(tokens, values)], label)


async def _cpu_ep_inspect(server):
    response = await asyncio.to_thread(
        requests.post,
        server.url_for("collective_rpc"),
        json={"method": "cpu_ep_expert_shards", "timeout": 30},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["results"][0]


@pytest.mark.cpu_test
@pytest.mark.asyncio
@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU serving required")
@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "compiled"])
async def test_cpu_ep_internal_lb_matches_reference_after_idle(
    monkeypatch, tmp_path, enforce_eager
):
    """Loaded EP shards match the full model across idle ranks, waves and restart."""
    model = "axolotl-ai-co/tiny-mixtral-30m"
    revision = "99512191d2561cb631b0d3176e59ade2d76b287b"
    args = [
        "--revision",
        revision,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "512",
        "--max-num-batched-tokens",
        "256",
        "--max-num-seqs",
        "4",
        "--kv-cache-memory-bytes",
        "67108864",
        "--host",
        "127.0.0.1",
        "--api-server-count",
        "1",
        "--distributed-executor-backend",
        "mp",
        "--shutdown-timeout",
        "5",
    ]
    if enforce_eager:
        args.append("--enforce-eager")
    env = {
        "OMP_NUM_THREADS": "1",
        "VLLM_CPU_OMP_THREADS_BIND": "nobind",
        "VLLM_HOST_IP": "127.0.0.1",
        "VLLM_LOG_STATS_INTERVAL": "0.1",
        "VLLM_SERVER_DEV_MODE": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "VLLM_CACHE_ROOT": str(tmp_path / "vllm_cache"),
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(VLLM_PATH), os.environ.get("PYTHONPATH")])
        ),
    }
    monkeypatch.delenv("VLLM_CPU_KVCACHE_SPACE", raising=False)
    monkeypatch.delenv("VLLM_CPU_CI_ENV", raising=False)
    long_prompt = "Describe a tiny robot exploring a forest. " * 8
    short_prompt = "Hello, my name is"
    workloads = [
        (long_prompt, 128),
        (short_prompt, 4),
        (long_prompt, 8),
        (short_prompt, 16),
    ]
    with RemoteOpenAIServer(model, args, env_dict=env) as reference_server:
        async with reference_server.get_async_client(timeout=90) as client:
            reference = [
                await _cpu_ep_completion(client, model, *work) for work in workloads
            ]
            prefixes = [
                work[0] + output[1] for work, output in zip(workloads, reference)
            ]
            reference_scores = [
                await _cpu_ep_score_prefix(client, model, prefix) for prefix in prefixes
            ]

    ep_args = args + [
        "--data-parallel-size",
        "2",
        "--data-parallel-size-local",
        "2",
        "--data-parallel-backend",
        "mp",
        "--enable-expert-parallel",
        "--all2all-backend",
        "allgather_reducescatter",
        "--worker-extension-cls",
        f"{__name__}.CpuEpInspection",
    ]
    with RemoteOpenAIServer(model, ep_args, env_dict=env) as server:
        shards = await _cpu_ep_inspect(server)
        print("CPU_EP_EXECUTION", shards, flush=True)
        assert len(shards) == 2
        if not enforce_eager:
            assert all(shard["cache_dir"] for shard in shards)
            assert len({shard["cache_dir"] for shard in shards}) == 2
        for shard in shards:
            if enforce_eager:
                assert shard["compile_mode"] == "NONE"
            else:
                assert shard["compile_mode"] != "NONE"
                assert shard["compile_backend"] == "inductor"
            assert shard["layers"]
            for layer in shard["layers"]:
                assert (layer["global"], layer["local"], layer["weights"]) == (8, 4, 4)
                owned = [idx for idx, local in enumerate(layer["map"]) if local >= 0]
                assert owned == list(range(shard["rank"] * 4, (shard["rank"] + 1) * 4))
                assert [layer["map"][idx] for idx in owned] == [0, 1, 2, 3]
        async with server.get_async_client(timeout=90) as client:
            started = asyncio.Event()
            async with asyncio.timeout(90), asyncio.TaskGroup() as tasks:
                long_request = tasks.create_task(
                    _cpu_ep_completion(client, model, *workloads[0], started=started)
                )
                await started.wait()
                short_output = await _cpu_ep_completion(client, model, *workloads[1])
                await _wait_cpu_ep_request_state(server, active=True)
                assert not long_request.done()
                outputs = [await long_request, short_output]
            await _wait_cpu_ep_request_state(server, active=False)
            check_logprobs_close(
                outputs_0_lst=reference[:2],
                outputs_1_lst=outputs,
                name_0="CPU full model",
                name_1="CPU DP2 EP2 asymmetric",
                always_check_logprobs=True,
            )
            _check_cpu_ep_generated_prefix(reference[:2], outputs, "asymmetric decode")
            async with asyncio.timeout(90):
                outputs = await asyncio.gather(
                    *[
                        _cpu_ep_completion(client, model, *work)
                        for work in workloads[2:]
                    ]
                )
            await _wait_cpu_ep_request_state(server, active=False)
            check_logprobs_close(
                outputs_0_lst=reference[2:],
                outputs_1_lst=outputs,
                name_0="CPU full model",
                name_1="CPU DP2 EP2 later wave",
                always_check_logprobs=True,
            )
            _check_cpu_ep_generated_prefix(reference[2:], outputs, "later wave decode")
            scores = await asyncio.gather(
                *[_cpu_ep_score_prefix(client, model, prefix) for prefix in prefixes]
            )
            _check_cpu_ep_scores(reference_scores, scores, "after idle waves")
            await _wait_cpu_ep_request_state(server, active=False)
        engine_counts = get_engine_request_counts(get_prometheus_metrics(server))
        assert set(engine_counts) == {"0", "1"}
        assert all(count > 0 for count in engine_counts.values())
    with RemoteOpenAIServer(model, ep_args, env_dict=env) as restarted:
        restarted_shards = await _cpu_ep_inspect(restarted)
        print("CPU_EP_RESTARTED", restarted_shards, flush=True)
        if not enforce_eager:
            assert {
                shard["rank"]: shard["cache_dir"] for shard in restarted_shards
            } == {shard["rank"]: shard["cache_dir"] for shard in shards}
            assert all(shard["aot_cache_loaded"] for shard in restarted_shards)
        async with restarted.get_async_client(timeout=90) as client:
            output = await _cpu_ep_completion(client, model, *workloads[1])
            scores = await _cpu_ep_score_prefix(client, model, prefixes[1])
            _check_cpu_ep_scores(reference_scores[1:2], [scores], "restarted")
        check_logprobs_close(
            outputs_0_lst=reference[1:2],
            outputs_1_lst=[output],
            name_0="CPU full model",
            name_1="CPU DP2 EP2 restarted",
            always_check_logprobs=True,
        )
        _check_cpu_ep_generated_prefix(reference[1:2], [output], "restarted decode")
