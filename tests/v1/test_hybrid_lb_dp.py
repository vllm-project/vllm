# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import threading
import time
from contextlib import AsyncExitStack

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer
from tests.v1.test_utils import check_request_balancing
from vllm.platforms import current_platform

MODEL_NAME = "ibm-research/PowerMoE-3b"

# Number of data parallel ranks for hybrid LB testing (4 total)
DP_SIZE = int(os.getenv("DP_SIZE", "4"))
# Default tensor parallel size to use
TP_SIZE = int(os.getenv("TP_SIZE", "1"))

# Number of nodes (2 nodes, each with 2 DP ranks)
NUM_NODES = 2
DP_SIZE_LOCAL = DP_SIZE // NUM_NODES  # 2 ranks per node


class HybridLBServerManager:
    """Manages hybrid data parallel vLLM server instances where each node 
    runs a single logical API server that balances requests only to the 
    DP engines running on that same node."""

    def __init__(self,
                 model_name: str,
                 dp_size: int,
                 api_server_count: int,
                 base_server_args: list,
                 dp_size_local: int = DP_SIZE_LOCAL,
                 tp_size: int = TP_SIZE):
        self.model_name = model_name
        self.dp_size = dp_size
        self.dp_size_local = dp_size_local
        self.tp_size = tp_size
        self.api_server_count = api_server_count
        self.base_server_args = base_server_args
        self.servers: list[tuple[RemoteOpenAIServer, list[str]]] = []
        self.server_threads: list[threading.Thread] = []
        self.num_nodes = dp_size // dp_size_local

    def __enter__(self) -> list[tuple[RemoteOpenAIServer, list[str]]]:
        """Start all server instances for hybrid LB mode."""
        for node_id in range(self.num_nodes):
            # Create server args for this specific node
            server_args = self.base_server_args.copy()

            # Calculate start rank for this node
            start_rank = node_id * self.dp_size_local

            # Add hybrid LB specific arguments
            server_args.extend([
                "--data-parallel-size",
                str(self.dp_size),
                "--data-parallel-size-local",
                str(self.dp_size_local),
                "--data-parallel-start-rank",
                str(start_rank),
                "--data-parallel-hybrid-lb",  # Enable hybrid LB mode
                "--tensor-parallel-size",
                str(self.tp_size),
                "--port",
                str(8000 + node_id),  # Different port for each node
                "--api-server-count",
                str(self.api_server_count),
                "--data-parallel-address",
                "127.0.0.1",
                "--data-parallel-rpc-port",
                "13345",
            ])

            # Use a thread to start each server to allow parallel initialization
            def start_server(node: int, sargs: list[str]):
                try:
                    # Calculate GPU devices for this node
                    gpus_per_node = self.dp_size_local * self.tp_size
                    gpu_start = node * gpus_per_node
                    gpu_end = gpu_start + gpus_per_node

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
                                for i in range(gpu_start, gpu_end))
                        })
                    server.__enter__()
                    print(f"Hybrid LB node {node} started successfully with "
                          f"{self.dp_size_local} local DP ranks and "
                          f"{self.api_server_count} API servers")
                    self.servers.append((server, sargs))
                except Exception as e:
                    print(f"Failed to start hybrid LB node {node}: {e}")
                    raise

            thread = threading.Thread(target=start_server,
                                      args=(node_id, server_args))
            thread.start()

            self.server_threads.append(thread)

        # Wait for all servers to start
        for thread in self.server_threads:
            thread.join()

        # Give servers additional time to fully initialize and coordinate
        time.sleep(3)

        if len(self.servers) != self.num_nodes:
            raise Exception("Servers failed to start")

        return self.servers

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop all server instances."""
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
    with HybridLBServerManager(MODEL_NAME, DP_SIZE, api_server_count,
                               default_server_args, DP_SIZE_LOCAL,
                               TP_SIZE) as server_list:
        yield server_list


@pytest_asyncio.fixture
async def clients(servers: list[tuple[RemoteOpenAIServer, list[str]]]):
    # Create a client for each node (each node has its own API endpoint)
    async with AsyncExitStack() as stack:
        yield [
            await stack.enter_async_context(server.get_async_client())
            for server, _ in servers
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_hybrid_lb_completion(clients: list[openai.AsyncOpenAI],
                                    servers: list[tuple[RemoteOpenAIServer,
                                                        list[str]]],
                                    model_name: str) -> None:

    async def make_request(client: openai.AsyncOpenAI):
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

    # Test single request to each node
    for i, client in enumerate(clients):
        result = await make_request(client)
        assert result is not None
        print(
            f"Hybrid LB node {i} handled single completion request successfully"
        )

    await asyncio.sleep(0.5)

    # Send requests to all nodes - each should balance within its local DP ranks
    num_requests = 200  # Total 200 requests across 2 nodes
    all_tasks = []
    for i in range(num_requests):
        client = clients[i % len(clients)]
        all_tasks.append(asyncio.create_task(make_request(client)))
        await asyncio.sleep(0.01)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests
    assert all(completion is not None for completion in results)

    await asyncio.sleep(0.5)

    # Second burst of requests
    all_tasks = []
    for i in range(num_requests):
        client = clients[i % len(clients)]
        all_tasks.append(asyncio.create_task(make_request(client)))
        await asyncio.sleep(0.01)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests
    assert all(completion is not None for completion in results)

    _, server_args = servers[0]
    api_server_count = (
        server_args.count('--api-server-count')
        and server_args[server_args.index('--api-server-count') + 1] or 1)
    print(
        f"Successfully completed hybrid LB test with {len(clients)} nodes "
        f"({DP_SIZE_LOCAL} DP ranks each, API server count: {api_server_count})"
    )

    # Check request balancing within each node
    for i, (server, _) in enumerate(servers):
        print(f"Checking request balancing for node {i}")
        check_request_balancing(server, DP_SIZE_LOCAL)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_hybrid_lb_completion_streaming(clients: list[
    openai.AsyncOpenAI], servers: list[tuple[RemoteOpenAIServer, list[str]]],
                                              model_name: str) -> None:
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

    # Test single request to each node
    for i, client in enumerate(clients):
        result = await make_streaming_request(client)
        assert result is not None
        print(
            f"Hybrid LB node {i} handled single streaming request successfully"
        )

    await asyncio.sleep(0.5)

    # Send streaming requests to all nodes
    num_requests = 200  # Total 200 requests across 2 nodes
    all_tasks = []
    for i in range(num_requests):
        client = clients[i % len(clients)]
        all_tasks.append(asyncio.create_task(make_streaming_request(client)))
        await asyncio.sleep(0.01)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests
    assert all(results), "Not all streaming requests completed successfully."

    await asyncio.sleep(0.5)

    # Second burst of streaming requests
    all_tasks = []
    for i in range(num_requests):
        client = clients[i % len(clients)]
        all_tasks.append(asyncio.create_task(make_streaming_request(client)))
        await asyncio.sleep(0.01)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests
    assert all(results), "Not all streaming requests completed successfully."

    _, server_args = servers[0]
    api_server_count = (
        server_args.count('--api-server-count')
        and server_args[server_args.index('--api-server-count') + 1] or 1)
    print(f"Successfully completed hybrid LB streaming test with "
          f"{len(clients)} nodes ({DP_SIZE_LOCAL} DP ranks each, "
          f"API server count: {api_server_count})")

    # Check request balancing within each node
    for i, (server, _) in enumerate(servers):
        print(f"Checking streaming request balancing for node {i}")
        check_request_balancing(server, DP_SIZE_LOCAL)
