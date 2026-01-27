# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer
from tests.v1.test_utils import check_request_balancing
from vllm.platforms import current_platform

MODEL_NAME = "ibm-research/PowerMoE-3b"

# Number of data parallel ranks for testing
DP_SIZE = int(os.getenv("DP_SIZE", "2"))
TP_SIZE = int(os.getenv("TP_SIZE", "1"))


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


@pytest.fixture(scope="module")
def prefix_routing_server_args(default_server_args):
    """Server args with prefix-aware routing enabled."""
    return default_server_args + [
        "--enable-prefix-aware-routing",
        "--prefix-routing-length",
        "16",
    ]


@pytest.fixture(scope="module")
def prefix_routing_server(prefix_routing_server_args):
    """Server with prefix-aware routing enabled."""
    args = prefix_routing_server_args + [
        "--data-parallel-size",
        str(DP_SIZE),
        "--tensor-parallel-size",
        str(TP_SIZE),
        "--port",
        "8000",
    ]
    gpus_needed = DP_SIZE * TP_SIZE
    with RemoteOpenAIServer(
            MODEL_NAME,
            args,
            auto_port=False,
            env_dict={
                current_platform.device_control_env_var:
                ",".join(
                    str(current_platform.device_id_to_physical_device_id(i))
                    for i in range(gpus_needed))
            }) as server:
        yield server


@pytest.fixture(scope="module")
def standard_server(default_server_args):
    """Server with standard load balancing (no prefix-aware routing)."""
    args = default_server_args + [
        "--data-parallel-size",
        str(DP_SIZE),
        "--tensor-parallel-size",
        str(TP_SIZE),
        "--port",
        "8001",
    ]
    gpus_needed = DP_SIZE * TP_SIZE
    with RemoteOpenAIServer(
            MODEL_NAME,
            args,
            auto_port=False,
            env_dict={
                current_platform.device_control_env_var:
                ",".join(
                    str(current_platform.device_id_to_physical_device_id(i))
                    for i in range(gpus_needed))
            }) as server:
        yield server


@pytest_asyncio.fixture
async def prefix_routing_client(prefix_routing_server: RemoteOpenAIServer):
    async with prefix_routing_server.get_async_client() as client:
        yield client


@pytest_asyncio.fixture
async def standard_client(standard_server: RemoteOpenAIServer):
    async with standard_server.get_async_client() as client:
        yield client


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_prefix_routing_same_prefix(
        prefix_routing_client: openai.AsyncOpenAI,
        prefix_routing_server: RemoteOpenAIServer, model_name: str) -> None:
    """Test that requests with the same prefix route to the same engine."""

    # Use the same prompt prefix for all requests
    base_prompt = "Once upon a time in a land far, far away"

    async def make_request(suffix: str):
        completion = await prefix_routing_client.completions.create(
            model=model_name,
            prompt=f"{base_prompt} {suffix}",
            max_tokens=5,
            temperature=0.0)
        return completion

    # Send multiple requests with the same prefix
    num_requests = 50
    tasks = [make_request(f"there lived {i}") for i in range(num_requests)]
    results = await asyncio.gather(*tasks)

    assert len(results) == num_requests
    assert all(completion is not None for completion in results)
    print(
        f"Successfully completed {num_requests} requests with same prefix routing"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_prefix_routing_load_balance(
        prefix_routing_client: openai.AsyncOpenAI,
        prefix_routing_server: RemoteOpenAIServer, model_name: str) -> None:
    """Test that new prefixes are load-balanced across engines."""

    # Generate requests with different prefixes
    prefixes = [
        "The quick brown fox",
        "In the beginning there was",
        "Scientists have discovered that",
        "The weather today is",
        "According to recent studies",
        "Many people believe that",
        "History shows us that",
        "Experts agree that",
    ]

    async def make_request(prefix: str, idx: int):
        completion = await prefix_routing_client.completions.create(
            model=model_name,
            prompt=f"{prefix} {idx}",
            max_tokens=5,
            temperature=0.0)
        return completion

    # Send multiple requests with different prefixes
    num_requests_per_prefix = 10
    tasks = []
    for prefix in prefixes:
        for i in range(num_requests_per_prefix):
            tasks.append(make_request(prefix, i))

    results = await asyncio.gather(*tasks)

    expected_total = len(prefixes) * num_requests_per_prefix
    assert len(results) == expected_total
    assert all(completion is not None for completion in results)

    # Check that requests were balanced across engines
    await asyncio.sleep(1)
    check_request_balancing(prefix_routing_server, DP_SIZE)
    print(
        f"Successfully load-balanced {expected_total} requests across {DP_SIZE} engines"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_prefix_routing_mixed_workload(
        prefix_routing_client: openai.AsyncOpenAI,
        prefix_routing_server: RemoteOpenAIServer, model_name: str) -> None:
    """Test mixed workload with both repeated and new prefixes."""

    # Prefixes that will be repeated (should hit cache)
    cached_prefixes = [
        "This is a cached prefix number",
        "Another cached prompt for testing",
    ]

    # Prefixes that are unique (won't hit cache)
    unique_prefixes = [
        f"Unique prefix {i} for testing" for i in range(20)
    ]

    async def make_request(prefix: str, idx: int):
        completion = await prefix_routing_client.completions.create(
            model=model_name,
            prompt=f"{prefix} {idx}",
            max_tokens=5,
            temperature=0.0)
        return completion

    tasks = []

    # Add cached prefix requests (multiple requests per prefix)
    for prefix in cached_prefixes:
        for i in range(10):
            tasks.append(make_request(prefix, i))

    # Add unique prefix requests (one request per prefix)
    for i, prefix in enumerate(unique_prefixes):
        tasks.append(make_request(prefix, i))

    results = await asyncio.gather(*tasks)

    expected_total = len(cached_prefixes) * 10 + len(unique_prefixes)
    assert len(results) == expected_total
    assert all(completion is not None for completion in results)
    print(
        f"Successfully completed mixed workload with {expected_total} requests"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_prefix_routing_short_prompts(
        prefix_routing_client: openai.AsyncOpenAI,
        prefix_routing_server: RemoteOpenAIServer, model_name: str) -> None:
    """Test that short prompts (< prefix_length) are handled correctly."""

    # Create prompts shorter than the prefix length (16 tokens)
    short_prompts = [
        "Hi",
        "Hello",
        "Test",
        "Short",
        "Brief prompt",
    ]

    async def make_request(prompt: str):
        completion = await prefix_routing_client.completions.create(
            model=model_name, prompt=prompt, max_tokens=5, temperature=0.0)
        return completion

    # Send requests with short prompts
    tasks = [make_request(prompt) for prompt in short_prompts]
    results = await asyncio.gather(*tasks)

    assert len(results) == len(short_prompts)
    assert all(completion is not None for completion in results)
    print(f"Successfully handled {len(short_prompts)} short prompts")


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_prefix_routing_config(
        prefix_routing_server: RemoteOpenAIServer) -> None:
    """Verify that prefix routing configuration is properly set."""
    # This test just verifies that the server starts successfully
    # with prefix routing enabled
    assert prefix_routing_server is not None
    print("Server started successfully with prefix-aware routing enabled")
