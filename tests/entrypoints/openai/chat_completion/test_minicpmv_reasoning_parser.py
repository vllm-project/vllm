# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM-V 4.7 thinks only on request, so a default request must return
the answer as `content`."""

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "openbmb/MiniCPM-V-4.7-1B"
MESSAGES = [{"role": "user", "content": "What is 2+2? Answer in one sentence."}]


@pytest.fixture(scope="module")
def server():
    args = ["--reasoning-parser", "minicpmv", "--max-model-len", "4096"]
    args += ["--enforce-eager", "--gpu-memory-utilization", "0.4"]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_default_request_returns_content(client: openai.AsyncOpenAI):
    kwargs = dict(model=MODEL_NAME, messages=MESSAGES, max_tokens=64, temperature=0)
    response = await client.chat.completions.create(**kwargs)
    message = response.choices[0].message
    assert message.content
    assert not getattr(message, "reasoning", None)

    stream = await client.chat.completions.create(**kwargs, stream=True)
    chunks = [chunk async for chunk in stream if chunk.choices]
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == message.content
