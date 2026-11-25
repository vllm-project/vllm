# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Clients send ordinary full chat histories without any session id; the
server must still reuse the previous turn's KV."""

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"
BLOCK_SIZE = 64
SYSTEM = "You are a terse assistant. " * 40


@pytest.fixture(scope="module")
def server():
    args = [
        "--session-cache",
        "--block-size",
        str(BLOCK_SIZE),
        "--max-model-len",
        "4096",
        "--enforce-eager",
        "--enable-prompt-tokens-details",
        "--attention-backend",
        "TRITON_ATTN",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


async def _turn(client: openai.AsyncOpenAI, messages: list[dict], stream: bool):
    kwargs = dict(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=32,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    if not stream:
        response = await client.chat.completions.create(**kwargs)
        return response.choices[0].message.content, response.usage
    content, usage = "", None
    async for chunk in await client.chat.completions.create(
        **kwargs, stream=True, stream_options={"include_usage": True}
    ):
        if chunk.choices and chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content
        if chunk.usage is not None:
            usage = chunk.usage
    return content, usage


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_full_history_reuses_previous_turn(
    client: openai.AsyncOpenAI, stream: bool
):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Say hello ({stream=})."},
    ]
    reply, first = await _turn(client, messages, stream)
    messages += [
        {"role": "assistant", "content": reply},
        {"role": "user", "content": "Now say goodbye."},
    ]
    _, second = await _turn(client, messages, stream)
    # The whole first prompt is reused up to its last block boundary; the
    # re-rendered assistant turn may differ from the generated tokens.
    first_prompt = first.prompt_tokens - first.prompt_tokens % BLOCK_SIZE
    assert second.prompt_tokens_details.cached_tokens >= first_prompt
