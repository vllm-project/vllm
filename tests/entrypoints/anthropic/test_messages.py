# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import anthropic
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "2048",
        "--enforce-eager",
        "--enable-prompt-tokens-details",
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

    first_chunk = None
    chunk_count = 0
    async for chunk in resp:
        chunk_count += 1
        if first_chunk is None and chunk.type == "message_start":
            first_chunk = chunk
        print(chunk.model_dump_json())

    assert chunk_count > 0
    assert first_chunk is not None, "message_start chunk was never observed"
    assert first_chunk.message is not None, "first chunk should include message"
    assert first_chunk.message.usage is not None, (
        "first chunk should include usage stats"
    )
    assert first_chunk.message.usage.output_tokens == 0
    assert first_chunk.message.usage.input_tokens > 5


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


@pytest.mark.asyncio
async def test_anthropic_structured_output(client: anthropic.AsyncAnthropic):
    response = await client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Extract the key information from this email:"
                "John Smith (john@example.com) is interested in our "
                "Enterprise plan and wants to schedule a demo for next Tuesday at 2pm.",
            }
        ],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                        "plan_interest": {"type": "string"},
                        "demo_requested": {"type": "boolean"},
                    },
                    "required": ["name", "email", "plan_interest", "demo_requested"],
                    "additionalProperties": False,
                },
            }
        },
    )
    print(response.content[0].text)
    json_obj = json.loads(response.content[0].text)
    for key in ["name", "email", "plan_interest", "demo_requested"]:
        assert key in json_obj, f"Missing key in output: {key}"


@pytest.mark.asyncio
async def test_anthropic_streaming_cache_usage(client: anthropic.AsyncAnthropic):
    async def get_stream_usage(resp):
        prompt_tokens = None
        usage = None
        async for chunk in resp:
            if (
                chunk.type == "message_start"
                and chunk.message is not None
                and chunk.message.usage is not None
            ):
                prompt_tokens = chunk.message.usage.input_tokens
            elif chunk.type == "message_delta" and chunk.usage is not None:
                usage = chunk.usage

        assert usage is not None
        assert usage.input_tokens >= 0
        assert usage.output_tokens >= 0
        cache_created = usage.cache_creation_input_tokens
        cache_read = usage.cache_read_input_tokens
        assert cache_read is not None
        assert cache_created is not None
        assert cache_created >= 0
        assert cache_read >= 0
        assert prompt_tokens == usage.input_tokens + cache_created + cache_read
        return usage

    request = dict(
        model="claude-3-7-sonnet-latest",
        max_tokens=1,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": "Cache coverage sentinel. " * 256
                + "Answer with exactly one word: ok.",
            }
        ],
        stream=True,
    )

    cold_usage = await get_stream_usage(await client.messages.create(**request))
    assert cold_usage.cache_read_input_tokens == 0
    assert cold_usage.cache_creation_input_tokens is not None
    assert cold_usage.cache_creation_input_tokens > 0

    warm_usage = await get_stream_usage(await client.messages.create(**request))
    assert warm_usage.cache_read_input_tokens is not None
    assert warm_usage.cache_read_input_tokens > 0
