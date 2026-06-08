# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import openai
import pytest
import pytest_asyncio
from tests.utils import RemoteOpenAIServer


@pytest.fixture(scope="module")
def chat_server_with_prefix_caching(request):
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "512",
        "--enforce-eager",
        "--max-num-seqs",
        "4",
        "--enable-prefix-caching",
        "--gpu-memory-utilization",
        "0.3",
    ]
    with RemoteOpenAIServer(
        "Qwen/Qwen3-0.6B", args, auto_port=True
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def chat_client_with_prefix_caching(chat_server_with_prefix_caching):
    async with chat_server_with_prefix_caching.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_cache_accounting_non_streaming(
    chat_client_with_prefix_caching: openai.AsyncOpenAI,
):
    """Second request with same prefix should show cache hits."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    # first request — cold, everything is a cache write
    first = await chat_client_with_prefix_caching.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=messages,
        max_completion_tokens=5,
        temperature=0.0,
    )
    assert first.usage is not None
    assert hasattr(first.usage, "cache_read_input_tokens")
    assert hasattr(first.usage, "cache_creation_input_tokens")
    # first request: nothing cached yet
    assert first.usage.cache_read_input_tokens == 0
    assert first.usage.cache_creation_input_tokens == first.usage.prompt_tokens

    # second request — same prefix, should hit cache
    second = await chat_client_with_prefix_caching.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=messages,
        max_completion_tokens=5,
        temperature=0.0,
    )
    assert second.usage is not None
    assert second.usage.cache_read_input_tokens > 0
    assert (
        second.usage.cache_read_input_tokens
        + second.usage.cache_creation_input_tokens
        == second.usage.prompt_tokens
    )


@pytest.mark.asyncio
async def test_cache_accounting_streaming(
    chat_client_with_prefix_caching: openai.AsyncOpenAI,
):
    """Streaming: final usage chunk should contain cache fields."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Germany?"},
    ]

    # warm the cache
    await chat_client_with_prefix_caching.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=messages,
        max_completion_tokens=5,
        temperature=0.0,
    )

    # second request streaming
    stream = await chat_client_with_prefix_caching.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=messages,
        max_completion_tokens=5,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True},
    )

    final_usage = None
    async for chunk in stream:
        if chunk.usage is not None:
            final_usage = chunk.usage

    assert final_usage is not None
    assert hasattr(final_usage, "cache_read_input_tokens")
    assert hasattr(final_usage, "cache_creation_input_tokens")
    assert final_usage.cache_read_input_tokens > 0