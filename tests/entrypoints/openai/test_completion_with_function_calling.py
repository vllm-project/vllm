# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio

# downloading lora to test lora requests
from ...utils import RemoteOpenAIServer

# any model with a chat template should work here
MODEL_NAME = "Qwen/Qwen3-0.6B"


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
        "hermes",
        "--reasoning-parser",
        "qwen3",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("tool_choice", ["auto", "required"])
@pytest.mark.parametrize("enable_thinking", [True, False])
async def test_function_tool_use(client: openai.AsyncOpenAI, model_name: str,
                                 stream: bool, tool_choice: str,
                                 enable_thinking: bool):
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
    if not stream:
        # Non-streaming test
        chat_completion = await client.chat.completions.create(
            messages=messages,
            model=model_name,
            tools=tools,
            tool_choice=tool_choice,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": enable_thinking
                }
            })

        assert chat_completion.choices[0].message.tool_calls is not None
        assert len(chat_completion.choices[0].message.tool_calls) > 0
    else:
        # Streaming test
        output_stream = await client.chat.completions.create(
            messages=messages,
            model=model_name,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": enable_thinking
                }
            })

        output = []
        async for chunk in output_stream:
            if chunk.choices and chunk.choices[0].delta.tool_calls:
                output.extend(chunk.choices[0].delta.tool_calls)

        assert len(output) > 0
