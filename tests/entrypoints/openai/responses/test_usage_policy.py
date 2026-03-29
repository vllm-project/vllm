# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Integration tests for usage policy configuration in Responses API.

Tests verify default behavior (no usage policy):
1. Non-streaming: Should return usage in response
2. Streaming: Usage should be in response.completed event, not in earlier events

Based on test_basic.py pattern.
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
        response = await client.responses.create(
            model=MODEL_NAME,
            input="Hello",
        )

        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0
        assert response.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_streaming(self, client: openai.AsyncOpenAI):
        """Streaming: Usage should only be in response.completed event."""
        stream = await client.responses.create(
            model=MODEL_NAME,
            input="Hello",
            stream=True,
        )

        events_with_usage_before_completed = []
        final_event = None

        async for event in stream:
            if event.type == "response.completed":
                final_event = event
                break
            if (
                hasattr(event, "response")
                and event.response
                and event.response.usage is not None
            ):
                events_with_usage_before_completed.append(event.type)

        assert final_event is not None
        assert len(events_with_usage_before_completed) == 0

        assert final_event.response.usage is not None
        assert final_event.response.usage.input_tokens > 0
        assert final_event.response.usage.output_tokens > 0
