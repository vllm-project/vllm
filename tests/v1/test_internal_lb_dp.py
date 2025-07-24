# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import threading
import time

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer
from tests.v1.test_utils import check_request_balancing
from vllm.platforms import current_platform

MODEL_NAME = "ibm-research/PowerMoE-3b"

# Number of data parallel ranks for multi-node internal LB testing
DP_SIZE = int(os.getenv("DP_SIZE", "2"))
# Default tensor parallel size to use
TP_SIZE = int(os.getenv("TP_SIZE", "1"))

# Number of nodes to simulate
NUM_NODES = 2


class MultinodeInternalLBServerManager:
    """Manages multi-node data parallel vLLM server instances for internal
    load balancer testing using --headless mode."""

    def __init__(self,
                 model_name: str,
                 dp_size: int,
                 api_server_count: int,
                 base_server_args: list,
                 dp_per_node: int = 1,
                 tp_size: int = TP_SIZE):
        self.model_name = model_name
        self.dp_size = dp_size
        self.dp_per_node = dp_per_node
        self.tp_size = tp_size
        self.api_server_count = api_server_count
        self.base_server_args = base_server_args
        self.servers: list[tuple[RemoteOpenAIServer, list[str]]] = []
        self.server_threads: list[threading.Thread] = []

    def __enter__(self) -> list[tuple[RemoteOpenAIServer, list[str]]]:
        """Start all server instances for multi-node internal LB mode."""
        for rank in range(0, self.dp_size, self.dp_per_node):
            # Create server args for this specific rank
            server_args = self.base_server_args.copy()

            if rank == 0:
                # Head node - runs API server and first DP rank
                server_args.extend([
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
                ])
            else:
                # Secondary nodes - run in headless mode
                server_args.extend([
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
                ])

            # Use a thread to start each server to allow parallel initialization
            def start_server(r: int, sargs: list[str]):
                gpus_per_node = self.tp_size * self.dp_per_node
                try:
                    # Start the server
                    server = RemoteOpenAIServer(
                        self.model_name,
                        sargs,
                        auto_port=False,
                        env_dict={
                            current_platform.device_control_env_var:
                            ",".join(
                                str(
                                    current_platform.
                                    device_id_to_physical_device_id(i))
                                for i in range(r, r + gpus_per_node))
                        })
                    server.__enter__()
                    if r == 0:
                        print(
                            f"Head node (rank {r}) started successfully with "
                            f"{self.api_server_count} API servers")
                    else:
                        print(f"Headless node (rank {r}) started successfully")
                    self.servers.append((server, sargs))
                except Exception as e:
                    print(f"Failed to start server rank {r}: {e}")
                    raise

            thread = threading.Thread(target=start_server,
                                      args=(rank, server_args))
            thread.start()

            self.server_threads.append(thread)

        # Wait for all servers to start
        for thread in self.server_threads:
            thread.join()

        # Give servers additional time to fully initialize and coordinate
        time.sleep(3)

        if len(self.servers) != self.dp_size // self.dp_per_node:
            raise Exception("Servers failed to start")

        return self.servers

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop all server instances."""
        while self.servers:
            try:
                self.servers.pop()[0].__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                print(f"Error stopping server: {e}")


class APIOnlyServerManager:
    """Manages API-only server (Node 0) and headless engines server (Node 1)
    for testing separated API server and engine configuration."""

    def __init__(self,
                 model_name: str,
                 dp_size: int,
                 api_server_count: int,
                 base_server_args: list,
                 tp_size: int = TP_SIZE):
        self.model_name = model_name
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.api_server_count = api_server_count
        self.base_server_args = base_server_args
        self.servers: list[tuple[RemoteOpenAIServer, list[str]]] = []
        self.server_threads: list[threading.Thread] = []

    def __enter__(self) -> list[tuple[RemoteOpenAIServer, list[str]]]:
        """Start API-only server and headless engines server."""

        # Start API-only server (Node 0) - no engines, only API server
        api_server_args = self.base_server_args.copy()
        api_server_args.extend([
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
        ])

        # Start headless engines server (Node 1) - all engines, no API server
        engines_server_args = self.base_server_args.copy()
        engines_server_args.extend([
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
        ])

        # Use threads to start both servers in parallel
        def start_api_server():
            try:
                server = RemoteOpenAIServer(
                    self.model_name,
                    api_server_args,
                    auto_port=False,
                    env_dict={})  # No GPUs needed for API-only server
                server.__enter__()
                print(f"API-only server started successfully with "
                      f"{self.api_server_count} API servers")
                self.servers.append((server, api_server_args))
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
                        current_platform.device_control_env_var:
                        ",".join(
                            str(
                                current_platform.
                                device_id_to_physical_device_id(i))
                            for i in range(self.dp_size * self.tp_size))
                    })
                server.__enter__()
                print(f"Headless engines server started successfully with "
                      f"{self.dp_size} engines")
                self.servers.append((server, engines_server_args))
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

        if len(self.servers) != 2:
            raise Exception("Both servers failed to start")

        return self.servers

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop both server instances."""
        while self.servers:
            try:
                self.servers.pop()[0].__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                print(f"Error stopping server: {e}")


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
def servers(request, default_server_args):
    api_server_count = request.param
    with MultinodeInternalLBServerManager(MODEL_NAME, DP_SIZE,
                                          api_server_count,
                                          default_server_args,
                                          DP_SIZE // NUM_NODES,
                                          TP_SIZE) as server_list:
        yield server_list


@pytest.fixture(scope="module", params=[1, 4])
def api_only_servers(request, default_server_args):
    """Fixture for API-only server + headless engines configuration."""
    api_server_count = request.param
    with APIOnlyServerManager(MODEL_NAME, DP_SIZE, api_server_count,
                              default_server_args, TP_SIZE) as server_list:
        yield server_list


@pytest_asyncio.fixture
async def client(servers: list[tuple[RemoteOpenAIServer, list[str]]]):
    # For internal LB, we only connect to the head node (rank 0)
    # which provides the single API endpoint
    head_server = servers[0][0]
    async with head_server.get_async_client() as client:
        yield client


@pytest_asyncio.fixture
async def api_only_client(api_only_servers: list[tuple[RemoteOpenAIServer,
                                                       list[str]]]):
    """Client fixture for API-only server configuration."""
    # Connect to the API-only server (first server in the list)
    api_server = api_only_servers[0][0]
    async with api_server.get_async_client() as client:
        yield client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_multinode_dp_completion(client: openai.AsyncOpenAI,
                                       servers: list[tuple[RemoteOpenAIServer,
                                                           list[str]]],
                                       model_name: str) -> None:

    async def make_request():
        completion = await client.completions.create(
            model=model_name,
            prompt="Hello, my name is",
            max_tokens=5,
            temperature=1.0)

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

    # Test single request
    result = await make_request()
    assert result is not None
    print(
        "Multi-node internal LB handled single completion request successfully"
    )

    await asyncio.sleep(0.5)

    # Send multiple requests - internal LB should distribute across DP ranks
    num_requests = 200
    all_tasks = []
    for _ in range(num_requests):
        all_tasks.append(asyncio.create_task(make_request()))
        await asyncio.sleep(0.01)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests
    assert all(completion is not None for completion in results)

    await asyncio.sleep(0.5)

    # Second burst of requests
    all_tasks = []
    for _ in range(num_requests):
        all_tasks.append(asyncio.create_task(make_request()))
        await asyncio.sleep(0.01)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests
    assert all(completion is not None for completion in results)

    _, server_args = servers[0]
    api_server_count = (
        server_args.count('--api-server-count')
        and server_args[server_args.index('--api-server-count') + 1] or 1)
    print(f"Successfully completed multi-node internal LB test with "
          f"{len(servers)} DP ranks (API server count: {api_server_count})")

    # Check request balancing via Prometheus metrics
    head_server = servers[0][0]
    check_request_balancing(head_server, DP_SIZE)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_multinode_dp_completion_streaming(client: openai.AsyncOpenAI,
                                                 servers: list[
                                                     tuple[RemoteOpenAIServer,
                                                           list[str]]],
                                                 model_name: str) -> None:
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
        stream = await client.completions.create(model=model_name,
                                                 prompt=prompt,
                                                 max_tokens=5,
                                                 temperature=0.0,
                                                 stream=True)
        chunks: list[str] = []
        finish_reason_count = 0
        last_chunk = None
        async for chunk in stream:
            chunks.append(chunk.choices[0].text)
            if chunk.choices[0].finish_reason is not None:
                finish_reason_count += 1
            last_chunk = chunk  # Keep track of the last chunk

        # finish reason should only return in the last block for OpenAI API
        assert finish_reason_count == 1, (
            "Finish reason should appear exactly once.")
        assert last_chunk is not None, (
            "Stream should have yielded at least one chunk.")
        assert last_chunk.choices[
            0].finish_reason == "length", "Finish reason should be 'length'."
        # Check that the combined text matches the non-streamed version.
        assert "".join(
            chunks
        ) == single_output, "Streamed output should match non-streamed output."
        return True  # Indicate success for this request

    # Test single streaming request
    result = await make_streaming_request()
    assert result is not None
    print(
        "Multi-node internal LB handled single streaming request successfully")

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
        server_args.count('--api-server-count')
        and server_args[server_args.index('--api-server-count') + 1] or 1)
    print(f"Successfully completed multi-node internal LB streaming test with "
          f"{len(servers)} DP ranks (API server count: {api_server_count})")

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
        api_only_servers: list[tuple[RemoteOpenAIServer,
                                     list[str]]], model_name: str) -> None:
    """Test API-only server with all engines on separate headless server."""

    async def make_request():
        completion = await api_only_client.completions.create(
            model=model_name,
            prompt="Hello, my name is",
            max_tokens=5,
            temperature=1.0)

        assert completion.id is not None
        assert completion.choices is not None and len(completion.choices) == 1

        choice = completion.choices[0]
        # The exact number of tokens can vary slightly with temperature=1.0,
        # so we check for a reasonable minimum length.
        assert len(choice.text) >= 1
        # Finish reason might not always be 'length' if the model finishes
        # early or due to other reasons, especially with high temperature.
        # So, we'll accept 'length' or 'stop'.
        assert choice.finish_reason in ("length", "stop")

        # Token counts can also vary, so we check they are positive.
        assert completion.usage.completion_tokens > 0
        assert completion.usage.prompt_tokens > 0
        assert completion.usage.total_tokens > 0
        return completion

    # Test single request
    result = await make_request()
    assert result is not None
    print("API-only server handled single completion request successfully")

    await asyncio.sleep(0.5)

    # Send multiple requests - should be distributed across engines on
    # headless server
    num_requests = 200
    all_tasks = []
    for _ in range(num_requests):
        all_tasks.append(asyncio.create_task(make_request()))
        await asyncio.sleep(0.01)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests
    assert all(completion is not None for completion in results)

    await asyncio.sleep(0.5)

    # Second burst of requests
    all_tasks = []
    for _ in range(num_requests):
        all_tasks.append(asyncio.create_task(make_request()))
        await asyncio.sleep(0.01)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests
    assert all(completion is not None for completion in results)

    _, api_server_args = api_only_servers[0]
    api_server_count = (
        api_server_args.count('--api-server-count')
        and api_server_args[api_server_args.index('--api-server-count') + 1]
        or 1)
    print(f"Successfully completed API-only multi-node test with {DP_SIZE} "
          f"engines on headless server (API server count: {api_server_count})")

    # Check request balancing via Prometheus metrics
    api_server = api_only_servers[0][0]
    check_request_balancing(api_server, DP_SIZE)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_api_only_multinode_dp_completion_streaming(
        api_only_client: openai.AsyncOpenAI,
        api_only_servers: list[tuple[RemoteOpenAIServer,
                                     list[str]]], model_name: str) -> None:
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
        stream = await api_only_client.completions.create(model=model_name,
                                                          prompt=prompt,
                                                          max_tokens=5,
                                                          temperature=0.0,
                                                          stream=True)
        chunks: list[str] = []
        finish_reason_count = 0
        last_chunk = None
        async for chunk in stream:
            chunks.append(chunk.choices[0].text)
            if chunk.choices[0].finish_reason is not None:
                finish_reason_count += 1
            last_chunk = chunk  # Keep track of the last chunk

        # finish reason should only return in the last block for OpenAI API
        assert finish_reason_count == 1, (
            "Finish reason should appear exactly once.")
        assert last_chunk is not None, (
            "Stream should have yielded at least one chunk.")
        assert last_chunk.choices[
            0].finish_reason == "length", "Finish reason should be 'length'."
        # Check that the combined text matches the non-streamed version.
        assert "".join(
            chunks
        ) == single_output, "Streamed output should match non-streamed output."
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
        api_server_args.count('--api-server-count')
        and api_server_args[api_server_args.index('--api-server-count') + 1]
        or 1)
    print(f"Successfully completed API-only streaming test with {DP_SIZE} "
          f"engines on headless server (API server count: {api_server_count})")

    # Check request balancing via Prometheus metrics
    api_server = api_only_servers[0][0]
    check_request_balancing(api_server, DP_SIZE)
