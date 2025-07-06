# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import re

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
import requests

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "ibm-research/PowerMoE-3b"

DP_SIZE = os.getenv("DP_SIZE", "1")


def get_prometheus_metrics(
        server: RemoteOpenAIServer) -> dict[str, dict[str, float]]:
    """Fetch and parse Prometheus metrics from the /metrics endpoint.
    
    Returns:
        Dict mapping metric names to their values grouped by labels.
        For example: {"vllm:request_success": {
            "engine=0": 5.0, "engine=1": 3.0}
        }
    """
    try:
        response = requests.get(server.url_for("metrics"), timeout=10)
        response.raise_for_status()

        metrics: dict[str, dict[str, float]] = {}

        # Regex patterns for Prometheus metrics
        metric_with_labels = re.compile(
            r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]*)\}\s+([\d\.\-\+e]+)$')
        metric_simple = re.compile(
            r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([\d\.\-\+e]+)$')

        for line in response.text.split('\n'):
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Try to match metric with labels first
            match = metric_with_labels.match(line)
            if match:
                metric_name, labels_part, value_str = match.groups()
                try:
                    value = float(value_str)
                    if metric_name not in metrics:
                        metrics[metric_name] = {}
                    metrics[metric_name][f'{{{labels_part}}}'] = value
                except ValueError:
                    continue
            else:
                # Try simple metric without labels
                match = metric_simple.match(line)
                if match:
                    metric_name, value_str = match.groups()
                    try:
                        value = float(value_str)
                        if metric_name not in metrics:
                            metrics[metric_name] = {}
                        metrics[metric_name][''] = value
                    except ValueError:
                        continue

        return metrics
    except Exception as e:
        pytest.fail(f"Failed to fetch Prometheus metrics: {e}")
        return {}


def get_engine_request_counts(
        metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    """Extract request counts per engine from Prometheus metrics.
    
    Returns:
        Dict mapping engine indices to request counts.
        For example: {"0": 15.0, "1": 12.0}
    """
    engine_counts = {}

    # Look for request success metrics with engine labels
    success_metrics = metrics.get("vllm:request_success_total", {})
    engine_pattern = re.compile(r'engine="([^"]*)"')

    for labels, count in success_metrics.items():
        # Extract engine ID from labels using regex
        match = engine_pattern.search(labels)
        if match:
            engine_id = match.group(1)
            if engine_id not in engine_counts:
                engine_counts[engine_id] = 0.0
            engine_counts[engine_id] += count

    return engine_counts


def check_request_balancing(server: RemoteOpenAIServer):
    """Check request balancing via Prometheus metrics if DP_SIZE > 1.
    
    Args:
        server: The RemoteOpenAIServer instance
    """
    dp_size = int(DP_SIZE)
    if dp_size <= 1:
        return

    # Get metrics after all requests are completed
    metrics = get_prometheus_metrics(server)
    engine_counts = get_engine_request_counts(metrics)

    # Check that multiple engines received requests
    engines_with_requests = [
        engine for engine, count in engine_counts.items() if count > 0
    ]
    assert len(engines_with_requests) == dp_size, (
        f"Expected requests to be distributed across multiple engines,"
        f" but only engine(s) {engines_with_requests} received "
        f"requests. Engine counts: {engine_counts}")

    # Verify that the load is reasonably balanced
    # (no engine should handle all requests)
    total_requests = sum(engine_counts.values())

    for count in engine_counts.values():
        assert count > total_requests // (dp_size + 1), (
            f"requests are imbalanced: {engine_counts}")


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
        "--api-server-count",
        "4",
        "--data_parallel_size",
        DP_SIZE,
    ]


@pytest.fixture(scope="module")
def server(default_server_args):
    with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_single_completion(client: openai.AsyncOpenAI,
                                 server: RemoteOpenAIServer,
                                 model_name: str) -> None:

    async def make_request():
        completion = await client.completions.create(
            model=model_name,
            prompt="Hello, my name is",
            max_tokens=10,
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

    await asyncio.sleep(0.5)

    # Send two bursts of requests
    num_requests = 100
    tasks = [make_request() for _ in range(num_requests)]
    results = await asyncio.gather(*tasks)
    assert len(results) == num_requests
    assert all(completion is not None for completion in results)

    await asyncio.sleep(0.5)

    tasks = [make_request() for _ in range(num_requests)]
    results = await asyncio.gather(*tasks)
    assert len(results) == num_requests
    assert all(completion is not None for completion in results)

    # Check request balancing via Prometheus metrics if DP_SIZE > 1
    check_request_balancing(server)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_completion_streaming(client: openai.AsyncOpenAI,
                                    server: RemoteOpenAIServer,
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

    # Test single request
    result = await make_streaming_request()
    assert result is not None

    await asyncio.sleep(0.5)

    # Send two bursts of requests
    num_requests = 100
    tasks = [make_streaming_request() for _ in range(num_requests)]
    results = await asyncio.gather(*tasks)

    assert len(
        results
    ) == num_requests, f"Expected {num_requests} results, got {len(results)}"
    assert all(results), "Not all streaming requests completed successfully."

    await asyncio.sleep(0.5)

    tasks = [make_streaming_request() for _ in range(num_requests)]
    results = await asyncio.gather(*tasks)

    assert len(
        results
    ) == num_requests, f"Expected {num_requests} results, got {len(results)}"
    assert all(results), "Not all streaming requests completed successfully."

    # Check request balancing via Prometheus metrics if DP_SIZE > 1
    check_request_balancing(server)
