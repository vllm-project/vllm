# SPDX-License-Identifier: Apache-2.0
import asyncio
import os

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "ibm-research/PowerMoE-3b"

DP_SIZE = os.getenv("DP_SIZE", "1")


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_completion_streaming(client: openai.AsyncOpenAI,
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
