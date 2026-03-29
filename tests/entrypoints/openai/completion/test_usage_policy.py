# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Integration tests for usage policy configuration in Completion API.

Tests verify default behavior (no usage policy):
1. Non-streaming: Should return usage in response
2. Streaming without stream_options: Should NOT return usage in chunks
3. Streaming with stream_options.include_usage: Should return usage in final chunk

Based on test_completion.py pattern.
"""

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"

BASE_ARGS = [
    "--max-model-len",
    "4096",
    "--gpu-memory-utilization",
    "0.8",
    "--enforce-eager",
]


@pytest.fixture(scope="module")
def server_args():
    return BASE_ARGS


@pytest.fixture(scope="module")
def server(server_args):
    with RemoteOpenAIServer(MODEL_NAME, server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


class TestUsagePolicyDefault:
    @pytest.mark.asyncio
    async def test_non_streaming(self, client: openai.AsyncOpenAI):
        """Non-streaming should always return usage in response."""
        response = await client.completions.create(
            model=MODEL_NAME,
            prompt="Hello",
            max_tokens=5,
            temperature=0.0,
            stream=False,
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    @pytest.mark.asyncio
    async def test_streaming_no_usage(self, client: openai.AsyncOpenAI):
        """Streaming without stream_options should NOT return usage in chunks."""
        stream = await client.completions.create(
            model=MODEL_NAME,
            prompt="Hello",
            max_tokens=5,
            temperature=0.0,
            stream=True,
        )

        chunk_count = 0
        usage_in_chunks = False

        async for chunk in stream:
            chunk_count += 1
            if chunk.usage is not None and not chunk.choices:
                usage_in_chunks = True

        assert not usage_in_chunks

    @pytest.mark.asyncio
    async def test_streaming_with_usage_option(self, client: openai.AsyncOpenAI):
        """Streaming with stream_options.include_usage=True should return usage."""
        stream = await client.completions.create(
            model=MODEL_NAME,
            prompt="Hello",
            max_tokens=5,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
        )

        chunk_count = 0
        final_chunk_with_usage = False

        async for chunk in stream:
            chunk_count += 1
            if chunk.usage is not None and not chunk.choices:
                final_chunk_with_usage = True
                assert chunk.usage.prompt_tokens > 0

        assert final_chunk_with_usage
