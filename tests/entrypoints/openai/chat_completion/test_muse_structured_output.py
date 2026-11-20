# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured outputs for Muse through the OpenAI server.

Muse ends reasoning with `<|eom|>`, then writes the answer header
`<|start|>assistant to=user<|message|>`. The grammar must apply to that answer
body only: before the fix the server returned prose, and with tools offered it
forced the schema into the tool call.
"""

import json

import jsonschema
import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer, large_gpu_mark

MODEL_NAME = "meta-models/Muse-Glimmer-30B"

pytestmark = large_gpu_mark(min_gb=80)

COUNTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "capital": {"type": "string"},
        "continent": {"type": "string"},
        "population_millions": {"type": "number"},
    },
    "required": ["name", "capital", "continent", "population_millions"],
    "additionalProperties": False,
}


@pytest.fixture(scope="module")
def server():
    args = [
        "--reasoning-parser",
        "muse",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "muse",
        "--max-model-len",
        "16384",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


async def _complete(
    client: openai.AsyncOpenAI, stream: bool, **kwargs
) -> tuple[str, str | None, list[str]]:
    """Return (content, finish_reason, tool names) for one greedy completion."""
    model = (await client.models.list()).data[0].id
    kwargs = dict(model=model, temperature=0, max_completion_tokens=4096, **kwargs)
    if not stream:
        choice = (await client.chat.completions.create(**kwargs)).choices[0]
        calls = [tc.function.name for tc in choice.message.tool_calls or []]
        return choice.message.content or "", choice.finish_reason, calls
    content, finish_reason, calls = "", None, []
    async for chunk in await client.chat.completions.create(stream=True, **kwargs):
        if chunk.choices:
            delta = chunk.choices[0].delta
            content += delta.content or ""
            calls += [
                tc.function.name
                for tc in delta.tool_calls or []
                if tc.function and tc.function.name
            ]
            finish_reason = chunk.choices[0].finish_reason or finish_reason
    return content, finish_reason, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_json_schema_applies_to_user_answer(client, stream):
    # The prompt does not ask for JSON: only the grammar can make the answer JSON.
    content, finish_reason, _ = await _complete(
        client,
        stream,
        messages=[{"role": "user", "content": "What is France?"}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "country",
                "schema": COUNTRY_SCHEMA,
                "strict": True,
            },
        },
    )
    assert finish_reason == "stop", content
    jsonschema.validate(json.loads(content), COUNTRY_SCHEMA)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_json_object_applies_to_user_answer(client, stream):
    content, finish_reason, _ = await _complete(
        client,
        stream,
        messages=[{"role": "user", "content": "Name the capital of France."}],
        response_format={"type": "json_object"},
    )
    assert finish_reason == "stop", content
    assert isinstance(json.loads(content), dict), content


@pytest.mark.asyncio
async def test_responses_text_format_applies_to_user_answer(client):
    # /v1/responses carries the schema in `text.format` instead of response_format.
    model = (await client.models.list()).data[0].id
    response = await client.responses.create(
        model=model,
        input="What is France?",
        text={
            "format": {
                "type": "json_schema",
                "name": "country",
                "schema": COUNTRY_SCHEMA,
                "strict": True,
            }
        },
        temperature=0,
        max_output_tokens=4096,
    )
    assert response.status == "completed", response.output_text
    jsonschema.validate(json.loads(response.output_text), COUNTRY_SCHEMA)


WEATHER_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "temperature_c": {"type": "number"}},
    "required": ["city", "temperature_c"],
    "additionalProperties": False,
}
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_json_schema_with_tools(client, stream):
    # A turn ends in a tool call or in the answer; only the answer is constrained.
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "weather", "schema": WEATHER_SCHEMA, "strict": True},
    }
    question = {
        "role": "user",
        "content": "Look up the weather in Paris, then report it.",
    }
    content, finish_reason, calls = await _complete(
        client,
        stream,
        messages=[question],
        tools=[WEATHER_TOOL],
        response_format=response_format,
    )
    assert calls == ["get_weather"], content
    assert finish_reason == "tool_calls", content

    call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                },
            }
        ],
    }
    result = {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"temperature_c": 18, "condition": "cloudy"}',
    }
    content, finish_reason, calls = await _complete(
        client,
        stream,
        messages=[question, call, result],
        tools=[WEATHER_TOOL],
        response_format=response_format,
    )
    assert finish_reason == "stop" and not calls, content
    report = json.loads(content)
    jsonschema.validate(report, WEATHER_SCHEMA)
    assert report["temperature_c"] == 18, content
