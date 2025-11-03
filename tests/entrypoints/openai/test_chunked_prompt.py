# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio

from ...utils import RemoteOpenAIServer

# any model with a chat template should work here
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"


@pytest.fixture(scope="module")
def server():
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--enforce-eager",
        # lora config below
        "--max-num-seqs",
        "128",
        "--enable-chunked-prefill",
        "--max-num-batched-tokens",
        "1000",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_completion_stream_options_and_logprobs_with_long_prompts(
    client: openai.AsyncOpenAI,
):
    # Test stream with long prompt
    prompt = "What is the capital of France?" * 400

    stream = await client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
        stream=True,
        stream_options={
            "include_usage": True,
            "continuous_usage_stats": True,
        },
        logprobs=5,
    )

    tokens_received = 0
    finished = False
    async for chunk in stream:
        assert chunk.usage.prompt_tokens >= 0
        assert chunk.usage.completion_tokens >= 0
        assert chunk.usage.total_tokens == (
            chunk.usage.prompt_tokens + chunk.usage.completion_tokens
        )
        if not finished:
            tokens_received += 1
            assert chunk.choices[0].text

            if chunk.choices[0].finish_reason is not None:
                finished = True

        if finished:
            assert chunk.usage.completion_tokens == tokens_received


@pytest.mark.asyncio
async def test_chat_completion_stream_options_and_logprobs_with_long_prompts(
    client: openai.AsyncOpenAI,
):
    # Test stream with long prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?" * 400},
    ]
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=5,
        temperature=0.0,
        stream=True,
        stream_options={
            "include_usage": True,
            "continuous_usage_stats": True,
        },
        logprobs=True,
        top_logprobs=5,
    )

    tokens_received = 0
    empty_chunks_received = 0
    finished = False
    async for chunk in stream:
        assert chunk.usage.prompt_tokens >= 0
        assert chunk.usage.completion_tokens >= 0
        assert chunk.usage.total_tokens == (
            chunk.usage.prompt_tokens + chunk.usage.completion_tokens
        )

        if not finished:
            if chunk.choices[0].delta.content == "":
                # when there is no tokens generated
                assert chunk.usage.completion_tokens == 0
                assert chunk.choices[0].logprobs is None
                empty_chunks_received += 1
            else:
                tokens_received += 1

            if chunk.choices[0].finish_reason is not None:
                finished = True

        if finished:
            assert chunk.usage.completion_tokens == tokens_received

    assert empty_chunks_received <= 1
