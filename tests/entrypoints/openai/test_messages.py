# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import anthropic
import pytest
import pytest_asyncio

from ...utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def server():  # noqa: F811
    args = [
        "--max-model-len",
        "2048",
        "--enforce-eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--served-model-name",
        "claude-3-7-sonnet-latest",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client_anthropic() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_simple_messages(client: anthropic.AsyncAnthropic):
    resp = await client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=1024,
        messages=[{"role": "user", "content": "how are you!"}],
    )
    assert resp.stop_reason == "end_turn"
    assert resp.role == "assistant"

    print(f"Anthropic response: {resp.model_dump_json()}")


@pytest.mark.asyncio
async def test_system_message(client: anthropic.AsyncAnthropic):
    resp = await client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=1024,
        system="you are a helpful assistant",
        messages=[{"role": "user", "content": "how are you!"}],
    )
    assert resp.stop_reason == "end_turn"
    assert resp.role == "assistant"

    print(f"Anthropic response: {resp.model_dump_json()}")


@pytest.mark.asyncio
async def test_anthropic_streaming(client: anthropic.AsyncAnthropic):
    resp = await client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=1024,
        messages=[{"role": "user", "content": "how are you!"}],
        stream=True,
    )

    async for chunk in resp:
        print(chunk.model_dump_json())


@pytest.mark.asyncio
async def test_anthropic_tool_call(client: anthropic.AsyncAnthropic):
    resp = await client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "What's the weather like in New York today?"}
        ],
        tools=[
            {
                "name": "get_current_weather",
                "description": "Useful for querying the weather in a specified city.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City or region, for example: "
                            "New York, London, Tokyo, etc.",
                        }
                    },
                    "required": ["location"],
                },
            }
        ],
        stream=False,
    )
    assert resp.stop_reason == "tool_use"
    assert resp.role == "assistant"

    print(f"Anthropic response: {resp.model_dump_json()}")


@pytest.mark.asyncio
async def test_anthropic_tool_call_streaming(client: anthropic.AsyncAnthropic):
    resp = await client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "What's the weather like in New York today?",
            }
        ],
        tools=[
            {
                "name": "get_current_weather",
                "description": "Useful for querying the weather in a specified city.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City or region, for example: "
                            "New York, London, Tokyo, etc.",
                        }
                    },
                    "required": ["location"],
                },
            }
        ],
        stream=True,
    )

    async for chunk in resp:
        print(chunk.model_dump_json())
