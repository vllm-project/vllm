# SPDX-License-Identifier: Apache-2.0

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio

# downloading lora to test lora requests
from ...utils import RemoteOpenAIServer

# any model with a chat template should work here
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def server():  # noqa: F811
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "half",
        "--enable-auto-tool-choice",
        "--guided-decoding-backend",
        "xgrammar",
        "--tool-call-parser",
        "hermes"
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_required_tool_use(client: openai.AsyncOpenAI, model_name: str):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description":
                            "The city to find the weather for, e.g. 'Vienna'",
                            "default": "Vienna",
                        },
                        "country": {
                            "type":
                            "string",
                            "description":
                            "The country that the city is in, e.g. 'Austria'",
                        },
                        "unit": {
                            "type": "string",
                            "description":
                            "The unit to fetch the temperature in",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["country", "unit"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_forecast",
                "description": "Get the weather forecast for a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description":
                            "The city to get the forecast for, e.g. 'Vienna'",
                            "default": "Vienna",
                        },
                        "country": {
                            "type":
                            "string",
                            "description":
                            "The country that the city is in, e.g. 'Austria'",
                        },
                        "days": {
                            "type":
                            "integer",
                            "description":
                            "Number of days to get the forecast for (1-7)",
                        },
                        "unit": {
                            "type": "string",
                            "description":
                            "The unit to fetch the temperature in",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["country", "days", "unit"],
                },
            },
        },
    ]

    messages = [
        {
            "role": "user",
            "content": "Hi! How are you doing today?"
        },
        {
            "role": "assistant",
            "content": "I'm doing well! How can I help you?"
        },
        {
            "role":
            "user",
            "content":
            "Can you tell me what the current weather is in Berlin and the "\
            "forecast for the next 5 days, in fahrenheit?",
        },
    ]

    # Non-streaming test
    chat_completion = await client.chat.completions.create(
        messages=messages,
        model=model_name,
        tools=tools,
        tool_choice="required",
    )

    assert chat_completion.choices[0].message.tool_calls is not None
    assert len(chat_completion.choices[0].message.tool_calls) > 0

    # Streaming test
    stream = await client.chat.completions.create(
        messages=messages,
        model=model_name,
        tools=tools,
        tool_choice="required",
        stream=True,
    )

    output = []
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.tool_calls:
            output.extend(chunk.choices[0].delta.tool_calls)

    assert len(output) > 0
