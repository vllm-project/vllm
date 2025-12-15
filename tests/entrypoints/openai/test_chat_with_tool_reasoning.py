# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio

from ...utils import RemoteOpenAIServer

# a reasoning and tool calling model
MODEL_NAME = "Qwen/QwQ-32B"


@pytest.fixture(scope="module")
def server():  # noqa: F811
    args = [
        "--max-model-len",
        "8192",
        "--enforce-eager",
        "--reasoning-parser",
        "deepseek_r1",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


TOOLS = [
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
                        "description": "The city to find the weather for, e.g. "
                        "'San Francisco'",
                    },
                    "state": {
                        "type": "string",
                        "description": "the two-letter abbreviation for the state that "
                        "the city is in, e.g. 'CA' which would mean 'California'",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "state", "unit"],
            },
        },
    }
]

MESSAGES = [
    {"role": "user", "content": "Hi! How are you doing today?"},
    {"role": "assistant", "content": "I'm doing well! How can I help you?"},
    {
        "role": "user",
        "content": "Can you tell me what the temperate will be in Dallas, "
        "in fahrenheit?",
    },
]

FUNC_NAME = "get_current_weather"
FUNC_ARGS = """{"city": "Dallas", "state": "TX", "unit": "fahrenheit"}"""


def extract_reasoning_and_calls(chunks: list):
    reasoning = ""
    tool_call_idx = -1
    arguments = []
    function_names = []
    for chunk in chunks:
        if chunk.choices[0].delta.tool_calls:
            tool_call = chunk.choices[0].delta.tool_calls[0]
            if tool_call.index != tool_call_idx:
                tool_call_idx = chunk.choices[0].delta.tool_calls[0].index
                arguments.append("")
                function_names.append("")

            if tool_call.function:
                if tool_call.function.name:
                    function_names[tool_call_idx] = tool_call.function.name

                if tool_call.function.arguments:
                    arguments[tool_call_idx] += tool_call.function.arguments
        else:
            if hasattr(chunk.choices[0].delta, "reasoning"):
                reasoning += chunk.choices[0].delta.reasoning
    return reasoning, arguments, function_names


# test streaming
@pytest.mark.asyncio
async def test_chat_streaming_of_tool_and_reasoning(client: openai.AsyncOpenAI):
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        tools=TOOLS,
        temperature=0.0,
        stream=True,
    )

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    reasoning, arguments, function_names = extract_reasoning_and_calls(chunks)
    assert len(reasoning) > 0
    assert len(function_names) > 0 and function_names[0] == FUNC_NAME
    assert len(arguments) > 0 and arguments[0] == FUNC_ARGS


# test full generate
@pytest.mark.asyncio
async def test_chat_full_of_tool_and_reasoning(client: openai.AsyncOpenAI):
    tool_calls = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        tools=TOOLS,
        temperature=0.0,
        stream=False,
    )

    assert len(tool_calls.choices[0].message.reasoning) > 0
    assert tool_calls.choices[0].message.tool_calls[0].function.name == FUNC_NAME
    assert tool_calls.choices[0].message.tool_calls[0].function.arguments == FUNC_ARGS
