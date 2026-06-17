# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import anthropic
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "512",
        "--enforce-eager",
        "--enable-prefix-caching",
        "--served-model-name",
        "claude-3-7-sonnet-latest",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args, auto_port=True) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client_anthropic() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_cache_accounting_non_streaming(client: anthropic.AsyncAnthropic):
    """Second request with same prefix should show cache hits in usage."""
    system = "You are a helpful assistant."
    messages = [{"role": "user", "content": "What is the capital of France?"}]

    # first request - cold cache
    first = await client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=5,
        system=system,
        messages=messages,
    )
    assert first.usage is not None
    assert first.usage.cache_read_input_tokens == 0
    # second request - same prefix, should hit cache
    second = await client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=5,
        system=system,
        messages=messages,
    )
    assert second.usage is not None
    assert second.usage.cache_read_input_tokens > 0


@pytest.mark.asyncio
async def test_cache_accounting_streaming(client: anthropic.AsyncAnthropic):
    """Streaming: final usage in message_delta event should show cache hits."""
    system = "You are a helpful assistant."
    messages = [{"role": "user", "content": "What is the capital of Germany?"}]

    # warm the cache
    await client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=5,
        system=system,
        messages=messages,
    )

    # second request streaming - should hit cache
    final_usage = None
    async with client.messages.stream(
        model="claude-3-7-sonnet-latest",
        max_tokens=5,
        system=system,
        messages=messages,
    ) as stream:
        async for event in stream:
            if (
                hasattr(event, "type")
                and event.type == "message_delta"
                and hasattr(event, "usage")
                and event.usage is not None
            ):
                final_usage = event.usage

    assert final_usage is not None
    assert hasattr(final_usage, "cache_read_input_tokens")
    assert final_usage.cache_read_input_tokens > 0