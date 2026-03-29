# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Integration tests for usage policy configuration in Embeddings API.

Tests verify default behavior (no usage policy):
1. Non-streaming: Should always return usage with prompt_tokens and total_tokens
2. Embeddings API does not support streaming in the traditional sense

Based on test_online.py pattern.
"""

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "BAAI/bge-base-en-v1.5"

BASE_ARGS = [
    "--runner",
    "pooling",
    "--dtype",
    "bfloat16",
    "--enforce-eager",
    "--max-model-len",
    "512",
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
        """Non-streaming embeddings should always return usage."""
        response = await client.embeddings.create(
            model=MODEL_NAME,
            input="Hello world",
            encoding_format="float",
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens == 0
        assert response.usage.total_tokens == response.usage.prompt_tokens

    @pytest.mark.asyncio
    async def test_non_streaming_batch(self, client: openai.AsyncOpenAI):
        """Batch embeddings should return usage per item."""
        response = await client.embeddings.create(
            model=MODEL_NAME,
            input=["Hello", "World"],
            encoding_format="float",
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens == 0
        assert response.usage.total_tokens > 0
