# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from pathlib import Path

import aiohttp
import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer


@pytest.fixture(scope="module")
def chat_server_with_force_include_usage(request):
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "128",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-num-seqs",
        "4",
        "--enable-force-include-usage",
        "--override-generation-config",
        '{"temperature": 0.0}',
        "--port",
        "55857",
        "--gpu-memory-utilization",
        "0.2",
        "--kv-cache-memory-bytes",
        str(128 * 1024 * 1024),
    ]

    with RemoteOpenAIServer(
        "Qwen/Qwen3-0.6B",
        args,
        auto_port=False,
        env_dict={"VLLM_BATCH_INVARIANT": "1"},
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def chat_client_with_force_include_usage(chat_server_with_force_include_usage):
    async with chat_server_with_force_include_usage.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_chat_with_enable_force_include_usage(
    chat_client_with_force_include_usage: openai.AsyncOpenAI,
):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    stream = await chat_client_with_force_include_usage.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=messages,
        max_completion_tokens=10,
        extra_body=dict(min_tokens=10),
        temperature=0.0,
        stream=True,
    )
    last_completion_tokens = 0
    async for chunk in stream:
        assert chunk.usage.prompt_tokens >= 0
        assert (
            last_completion_tokens == 0
            or chunk.usage.completion_tokens > last_completion_tokens
            or (
                not chunk.choices
                and chunk.usage.completion_tokens == last_completion_tokens
            )
        )
        assert chunk.usage.total_tokens == (
            chunk.usage.prompt_tokens + chunk.usage.completion_tokens
        )
        last_completion_tokens = chunk.usage.completion_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_multi_turn_benchmark_with_force_include_usage(
    chat_server_with_force_include_usage,
    chat_client_with_force_include_usage: openai.AsyncOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
):
    """The benchmark retains generated text when the server forces usage."""
    benchmark_dir = Path(__file__).parents[4] / "benchmarks" / "multi_turn"
    monkeypatch.syspath_prepend(str(benchmark_dir))
    benchmark = importlib.import_module("benchmark_serving_multi_turn")
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    expected = await chat_client_with_force_include_usage.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=messages,
        max_tokens=10,
        temperature=0.0,
    )

    async with aiohttp.ClientSession() as session:
        result = await benchmark.send_request(
            session,
            messages,
            chat_server_with_force_include_usage.url_for("v1/chat/completions"),
            "Qwen/Qwen3-0.6B",
            stream=stream,
            max_tokens=10,
        )

    assert result.valid
    assert result.content == expected.choices[0].message.content != ""
    if stream:
        assert result.first_chunk
        assert result.content.startswith(result.first_chunk)
        assert 0 <= result.ttft_ms <= result.latency_ms
