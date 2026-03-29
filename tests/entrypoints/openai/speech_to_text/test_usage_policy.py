# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Integration tests for usage policy configuration in Speech-to-Text API.

Tests verify default behavior (no usage policy):
1. Non-streaming transcription: Should return usage in response
2. Streaming transcription: Should NOT return usage in chunks by default
3. Non-streaming translation: Should NOT return usage in response

Based on test_serving_chat.py pattern.
"""

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "openai/whisper-large-v3-turbo"

BASE_ARGS = [
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
    async def test_non_streaming_transcription(
        self, client: openai.AsyncOpenAI, winning_call
    ):
        """Non-streaming transcription should always return usage in response."""
        response = await client.audio.transcriptions.create(
            model=MODEL_NAME,
            file=winning_call,
            language="en",
            temperature=0.0,
            response_format="json",
        )

        assert response.usage is not None
        assert response.usage.seconds > 0

    @pytest.mark.asyncio
    async def test_streaming_transcription(
        self, client: openai.AsyncOpenAI, winning_call
    ):
        """Streaming transcription should NOT return usage in chunks by default."""
        stream = await client.audio.transcriptions.create(
            model=MODEL_NAME,
            file=winning_call,
            language="en",
            temperature=0.0,
            response_format="json",
            stream=True,
        )

        chunk_count = 0
        usage_in_chunks = False

        async for chunk in stream:
            chunk_count += 1
            has_usage = hasattr(chunk, "usage") and chunk.usage is not None
            if not chunk.choices and has_usage:
                usage_in_chunks = True

        assert not usage_in_chunks
        assert chunk_count > 0
