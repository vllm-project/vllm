# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plain replies that start with "[" must not be held back as tool calls."""

import openai
import pytest
import pytest_asyncio
import regex as re

from tests.utils import RemoteOpenAIServer

MODEL = "Qwen/Qwen3-0.6B"
REPLY = "[1] Paris is the capital of France."
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


@pytest.fixture(scope="module", params=["xlam", "hunyuan_a13b"])
def server(request):
    args = [
        "--enforce-eager",
        "--max-model-len",
        "1024",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        request.param,
    ]
    with RemoteOpenAIServer(MODEL, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server: RemoteOpenAIServer):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_streamed_reply_starting_with_bracket(client: openai.AsyncOpenAI):
    """The streamed reply equals the non-streamed one, not empty content."""
    request = dict(
        model=MODEL,
        messages=[{"role": "user", "content": "Hi"}],
        tools=TOOLS,
        max_tokens=32,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "structured_outputs": {"regex": re.escape(REPLY)},
        },
    )

    response = await client.chat.completions.create(**request)
    assert response.choices[0].message.content == REPLY
    assert not response.choices[0].message.tool_calls

    stream = await client.chat.completions.create(stream=True, **request)
    chunks = [chunk async for chunk in stream if chunk.choices]
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == REPLY
    assert not any(c.choices[0].delta.tool_calls for c in chunks)
