# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Integration tests for usage policy configuration in Anthropic Messages API.

Tests verify default behavior (no usage policy):
1. Non-streaming: Should return usage in response
2. Streaming: Usage should appear in message_start and message_delta events

Based on test_messages.py pattern.
"""

import anthropic
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
    async with server.get_async_client_anthropic() as async_client:
        yield async_client


class TestUsagePolicyDefault:
    @pytest.mark.asyncio
    async def test_non_streaming(self, client: anthropic.AsyncAnthropic):
        """Non-streaming should always return usage in response."""
        response = await client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    @pytest.mark.asyncio
    async def test_streaming(self, client: anthropic.AsyncAnthropic):
        """Streaming: Usage should appear in message_start and message_delta."""
        stream = await client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

        message_start_usage = None
        message_delta_usage = None

        async for event in stream:
            if event.type == "message_start":
                message_start_usage = event.message.usage
            elif event.type == "message_delta":
                message_delta_usage = event.usage

        assert message_start_usage is not None
        assert message_start_usage.input_tokens > 0
        assert message_start_usage.output_tokens == 0

        assert message_delta_usage is not None
        assert message_delta_usage.input_tokens > 0
        assert message_delta_usage.output_tokens > 0
