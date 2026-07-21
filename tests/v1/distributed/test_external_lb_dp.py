# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AsyncExitStack
from types import SimpleNamespace

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
import requests

from tests.utils import RemoteOpenAIServer
from vllm.distributed.elastic_ep.external_elastic_ep import (
    ExternalElasticEPScaleCoordinator,
)
from vllm.platforms import current_platform

MODEL_NAME = os.getenv("MODEL_NAME", "ibm-research/PowerMoE-3b")
ELASTIC_EP_MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite-Chat"

# Number of data parallel ranks for external LB testing
DP_SIZE = int(os.getenv("DP_SIZE", "2"))
# Default tensor parallel size to use
TP_SIZE = int(os.getenv("TP_SIZE", "1"))


class DictStore:
    def __init__(self, values: dict[str, bytes]):
        self.values = values

    def check(self, keys: list[str]) -> bool:
        return all(key in self.values for key in keys)

    def get(self, key: str) -> bytes:
        return self.values[key]


@pytest.mark.asyncio
async def test_external_elastic_ep_late_rank_observes_epoch_error():
    coordinator = ExternalElasticEPScaleCoordinator(SimpleNamespace())
    epoch = "failed-epoch"
    store = DictStore(
        {
            coordinator.key("current_epoch"): epoch.encode(),
            coordinator.key(epoch, "error"): b"scale failed",
        }
    )

    with pytest.raises(RuntimeError, match="scale failed"):
        await coordinator._wait_for_bootstrap(store, requested_new_dp_size=3)


class ExternalLBServerManager:
    """Manages data parallel vLLM server instances for external
    load balancer testing."""

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
        self.servers: list[tuple[RemoteOpenAIServer, list[str]]] = []
        self.server_threads: list[threading.Thread] = []

    def start_rank(
        self,
        rank: int,
        dp_size: int | None = None,
        elastic_ep_scale_up_launch: bool = False,
    ) -> tuple[RemoteOpenAIServer, list[str]]:
        """Start one externally managed DP rank."""
        dp_size = self.dp_size if dp_size is None else dp_size
        server_args = self.base_server_args.copy()
        server_args.extend(
            [
                "--data-parallel-size",
                str(dp_size),
                "--data-parallel-rank",
                str(rank),
                "--data-parallel-size-local",
                "1",
                "--tensor-parallel-size",
                str(self.tp_size),
                "--port",
                str(8000 + rank),
                "--api-server-count",
                str(self.api_server_count),
            ]
        )

        env_dict = {
            "VLLM_SERVER_DEV_MODE": "1",
            current_platform.device_control_env_var: ",".join(
                str(current_platform.device_id_to_physical_device_id(i))
                for i in range(rank * self.tp_size, (rank + 1) * self.tp_size)
            ),
        }
        if elastic_ep_scale_up_launch:
            env_dict["VLLM_ELASTIC_EP_SCALE_UP_LAUNCH"] = "1"

        server = RemoteOpenAIServer(
            self.model_name,
            server_args,
            auto_port=False,
            env_dict=env_dict,
        )
        print(
            f"Server rank {rank} started successfully with "
            f"{self.api_server_count} API servers"
        )
        server_and_args = (server, server_args)
        self.servers.append(server_and_args)
        return server_and_args

    def __enter__(self) -> list[tuple[RemoteOpenAIServer, list[str]]]:
        """Start all initial server instances for external LB mode."""
        start_errors: list[Exception] = []
        for rank in range(self.dp_size):
            # Use a thread to start each server to allow parallel initialization
            def start_server(r: int):
                try:
                    self.start_rank(r)
                except Exception as e:
                    print(f"Failed to start server rank {r}: {e}")
                    start_errors.append(e)

            thread = threading.Thread(target=start_server, args=(rank,))
            thread.start()

            self.server_threads.append(thread)

        # Wait for all servers to start
        for thread in self.server_threads:
            thread.join()

        if start_errors:
            start_error = start_errors[0]
            raise RuntimeError("Failed to start external LB servers") from start_error

        # Give servers additional time to fully initialize and coordinate
        time.sleep(2)

        if len(self.servers) != self.dp_size:
            raise Exception("Servers failed to start")

        return self.servers

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop all server instances."""
        servers = [s for s, _ in self.servers]
        self.servers.clear()
        try:
            RemoteOpenAIServer.shutdown_many(servers)
        except Exception as e:
            print(f"Error stopping servers: {e}")


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
    server_manager = ExternalLBServerManager(
        MODEL_NAME, DP_SIZE, api_server_count, default_server_args
    )

    with server_manager:
        yield server_manager


@pytest.fixture
def servers(server_manager):
    return server_manager.servers


@pytest_asyncio.fixture
async def clients(servers: list[tuple[RemoteOpenAIServer, list[str]]]):
    # Create a client for each server
    async with AsyncExitStack() as stack:
        yield [
            await stack.enter_async_context(server.get_async_client())
            for server, _ in servers
        ]


def _get_parallel_config(server: RemoteOpenAIServer):
    response = requests.get(server.url_for("server_info?config_format=json"))
    response.raise_for_status()

    vllm_config = response.json()["vllm_config"]
    return vllm_config["parallel_config"]


def _send_scale_command(
    server: RemoteOpenAIServer, new_dp_size: int
) -> requests.Response:
    return requests.post(
        server.url_for("scale_elastic_ep"),
        json={"new_data_parallel_size": new_dp_size},
        timeout=600,
    )


def _wait_for_scale_requests(
    servers: list[RemoteOpenAIServer],
    scale_requests: list[Future[requests.Response]],
    timeout: float = 30,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for request in scale_requests:
            if request.done():
                response = request.result()
                pytest.fail(
                    "Scale request finished before the new rank was launched: "
                    f"{response.status_code} {response.text}"
                )

        scaling = []
        for server in servers:
            try:
                response = requests.post(
                    server.url_for("is_scaling_elastic_ep"), timeout=1
                )
                if response.status_code == 503:
                    scaling.append(True)
                    continue
                response.raise_for_status()
                scaling.append(response.json()["is_scaling_elastic_ep"])
            except requests.RequestException:
                scaling.append(False)
        if all(scaling):
            return
        time.sleep(0.1)

    pytest.fail("Timed out waiting for all existing ranks to start scaling")


@pytest.mark.distributed(num_gpus=3)
@pytest.mark.skipif(
    TP_SIZE != 1 or current_platform.device_count() < 3,
    reason="External Elastic EP scale-up requires three GPUs with TP_SIZE=1",
)
@pytest.mark.skipif(
    os.getenv("VLLM_USE_RUST_FRONTEND") == "1",
    reason="Elastic EP scaling is not supported by the Rust frontend",
)
def test_external_lb_elastic_ep_scale_up(default_server_args) -> None:
    server_args = [
        *default_server_args,
        "--enable-expert-parallel",
        "--all2all-backend",
        "allgather_reducescatter",
        "--attention-backend",
        "TRITON_MLA",
        "--enable-elastic-ep",
        "--enable-eplb",
        "--eplb-config",
        '{"communicator":"pynccl","num_redundant_experts":0,"use_async":false}',
    ]
    server_manager = ExternalLBServerManager(
        ELASTIC_EP_MODEL_NAME,
        dp_size=2,
        api_server_count=1,
        base_server_args=server_args,
    )

    with server_manager:
        existing_servers = [server for server, _ in server_manager.servers]
        with ThreadPoolExecutor(max_workers=2) as executor:
            scale_requests = [
                executor.submit(_send_scale_command, server, 3)
                for server in existing_servers
            ]
            _wait_for_scale_requests(existing_servers, scale_requests)
            server_manager.start_rank(
                rank=2,
                dp_size=3,
                elastic_ep_scale_up_launch=True,
            )

            for request in scale_requests:
                response = request.result(timeout=600)
                assert response.status_code == 200, response.text

        for server, _ in server_manager.servers:
            assert _get_parallel_config(server)["data_parallel_size"] == 3
            with server.get_client() as client:
                completion = client.completions.create(
                    model=ELASTIC_EP_MODEL_NAME,
                    prompt="Hello, my name is",
                    max_tokens=5,
                    temperature=0.0,
                )
            assert completion.choices[0].finish_reason in ("length", "stop")


def test_external_lb_server_info(server_manager):
    servers = server_manager.servers
    api_server_count = server_manager.api_server_count

    for i, (server, _) in enumerate(servers):
        print(f"Testing {i=}")

        # Each request will hit one of the API servers
        # `n_reqs` is set so that there is a good chance each server
        # receives at least one request
        n_reqs = 2 * api_server_count * api_server_count
        parallel_configs = [_get_parallel_config(server) for _ in range(n_reqs)]
        api_process_counts = [c["_api_process_count"] for c in parallel_configs]
        api_process_ranks = [c["_api_process_rank"] for c in parallel_configs]

        assert all(c == api_server_count for c in api_process_counts), (
            api_process_counts
        )
        assert all(0 <= r < api_server_count for r in api_process_ranks), (
            api_process_ranks
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_external_lb_single_completion(
    clients: list[openai.AsyncOpenAI],
    servers: list[tuple[RemoteOpenAIServer, list[str]]],
    model_name: str,
) -> None:
    async def make_request(client: openai.AsyncOpenAI):
        completion = await client.completions.create(
            model=model_name, prompt="Hello, my name is", max_tokens=10, temperature=1.0
        )

        assert completion.id is not None
        assert completion.choices is not None and len(completion.choices) == 1

        choice = completion.choices[0]
        # The exact number of tokens can vary slightly with temperature=1.0,
        # so we check for a reasonable minimum length.
        assert len(choice.text) >= 1
        # Finish reason might not always be 'length' if the model finishes early
        # or due to other reasons, especially with high temperature.
        # So, we'll accept 'length' or 'stop'.
        assert choice.finish_reason in ("length", "stop")

        # Token counts can also vary, so we check they are positive.
        assert completion.usage.completion_tokens > 0
        assert completion.usage.prompt_tokens > 0
        assert completion.usage.total_tokens > 0
        return completion

    # Test single request to each server
    for i, client in enumerate(clients):
        result = await make_request(client)
        assert result is not None
        print(f"Server {i} handled single completion request successfully")

    await asyncio.sleep(0.5)

    # Send requests to all servers in round-robin fashion
    num_requests_per_server = 25  # Total 50 requests across 2 servers
    all_tasks = []

    for i, client in enumerate(clients):
        tasks = [make_request(client) for _ in range(num_requests_per_server)]
        all_tasks.extend(tasks)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests_per_server * len(clients)
    assert all(completion is not None for completion in results)

    await asyncio.sleep(0.5)

    # Second burst of requests
    all_tasks = []
    for i, client in enumerate(clients):
        tasks = [make_request(client) for _ in range(num_requests_per_server)]
        all_tasks.extend(tasks)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests_per_server * len(clients)
    assert all(completion is not None for completion in results)

    _, server_args = servers[0]
    api_server_count = (
        server_args.count("--api-server-count")
        and server_args[server_args.index("--api-server-count") + 1]
        or 1
    )
    print(
        f"Successfully completed external LB test with {len(clients)} servers "
        f"(API server count: {api_server_count})"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_external_lb_completion_streaming(
    clients: list[openai.AsyncOpenAI],
    servers: list[tuple[RemoteOpenAIServer, list[str]]],
    model_name: str,
) -> None:
    prompt = "What is an LLM?"

    async def make_streaming_request(client: openai.AsyncOpenAI):
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

    # Test single request to each server
    for i, client in enumerate(clients):
        result = await make_streaming_request(client)
        assert result is not None
        print(f"Server {i} handled single streaming request successfully")

    await asyncio.sleep(0.5)

    # Send streaming requests to all servers in round-robin fashion
    num_requests_per_server = 25  # Total 50 requests across 2 servers
    all_tasks = []

    for i, client in enumerate(clients):
        tasks = [make_streaming_request(client) for _ in range(num_requests_per_server)]
        all_tasks.extend(tasks)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests_per_server * len(clients)
    assert all(results), "Not all streaming requests completed successfully."

    await asyncio.sleep(0.5)

    # Second burst of streaming requests
    all_tasks = []
    for i, client in enumerate(clients):
        tasks = [make_streaming_request(client) for _ in range(num_requests_per_server)]
        all_tasks.extend(tasks)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests_per_server * len(clients)
    assert all(results), "Not all streaming requests completed successfully."

    _, server_args = servers[0]
    api_server_count = (
        server_args.count("--api-server-count")
        and server_args[server_args.index("--api-server-count") + 1]
        or 1
    )
    print(
        f"Successfully completed external LB streaming test with "
        f"{len(clients)} servers (API server count: {api_server_count})"
    )
