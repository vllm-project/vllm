# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import openai
import pytest
import pytest_asyncio
from ...utils import RemoteOpenAIServer

MODEL_NAME = "openai/gpt-oss-20b"

@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len", "8192",
        "--enforce-eager",
        "--reasoning-parser", "deepseek_r1",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "openai"
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server

@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client

# ------------------------------
# Tool definitions
# ------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Performs arithmetic calculations",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time in a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }
]

# ------------------------------
# Messages examples
# ------------------------------
MESSAGES_CALC = [
    {"role": "user", "content": "Calculate 123 + 456 using the calculator."}
]

MESSAGES_MULTIPLE_CALLS = [
    {"role": "user", "content": "What is 7*8? And what time is it in New York?"},
]

MESSAGES_INVALID_CALL = [
    {"role": "user", "content": "Use the calculator but give no expression."}
]

FUNC_CALC = "calculator"
FUNC_ARGS_CALC = """{"expression": "123 + 456"}"""

FUNC_TIME = "get_time"
FUNC_ARGS_TIME = """{"city": "New York"}"""

# ------------------------------
# Utility to extract reasoning and tool calls
# ------------------------------
def extract_reasoning_and_calls(chunks: list):
    reasoning_content = ""
    tool_call_idx = -1
    arguments = []
    function_names = []
    for chunk in chunks:
        if chunk.choices[0].delta.tool_calls:
            tool_call = chunk.choices[0].delta.tool_calls[0]
            if tool_call.index != tool_call_idx:
                tool_call_idx = tool_call.index
                arguments.append("")
                function_names.append("")
            if tool_call.function:
                if tool_call.function.name:
                    function_names[tool_call_idx] = tool_call.function.name
                if tool_call.function.arguments:
                    arguments[tool_call_idx] += tool_call.function.arguments
        else:
            if hasattr(chunk.choices[0].delta, "reasoning_content"):
                reasoning_content += chunk.choices[0].delta.reasoning_content
    return reasoning_content, arguments, function_names

# ------------------------------
# Test Scenarios
# ------------------------------

# 1. Single tool call (calculator)
@pytest.mark.asyncio
async def test_single_tool_call(client: openai.AsyncOpenAI):
    stream = await client.chat.completions.create(
        model=MODEL_NAME, messages=MESSAGES_CALC, tools=TOOLS,
        temperature=0.0, stream=True
    )
    chunks = [chunk async for chunk in stream]
    reasoning, arguments, function_names = extract_reasoning_and_calls(chunks)
    assert FUNC_CALC in function_names
    assert FUNC_ARGS_CALC in arguments
    assert len(reasoning) > 0

# 2. Multiple tools in a single query
@pytest.mark.asyncio
async def test_multiple_tool_calls(client: openai.AsyncOpenAI):
    tool_calls = await client.chat.completions.create(
        model=MODEL_NAME, messages=MESSAGES_MULTIPLE_CALLS,
        tools=TOOLS, temperature=0.0, stream=False
    )
    calls = tool_calls.choices[0].message.tool_calls
    assert any(c.function.name == FUNC_CALC for c in calls)
    assert any(c.function.name == FUNC_TIME for c in calls)
    assert len(tool_calls.choices[0].message.reasoning_content) > 0

# 3. Invalid tool input should raise an error
@pytest.mark.asyncio
async def test_invalid_tool_call(client: openai.AsyncOpenAI):
    with pytest.raises(Exception):
        await client.chat.completions.create(
            model=MODEL_NAME, messages=MESSAGES_INVALID_CALL,
            tools=TOOLS, temperature=0.0
        )

# 4. Streaming + multiple reasoning + tool calls
@pytest.mark.asyncio
async def test_streaming_multiple_tools(client: openai.AsyncOpenAI):
    stream = await client.chat.completions.create(
        model=MODEL_NAME, messages=MESSAGES_MULTIPLE_CALLS,
        tools=TOOLS, temperature=0.0, stream=True
    )
    chunks = [chunk async for chunk in stream]
    reasoning, arguments, function_names = extract_reasoning_and_calls(chunks)
    assert FUNC_CALC in function_names
    assert FUNC_TIME in function_names
    assert len(reasoning) > 0

# 5. Tool call with temperature effect (non-zero randomness)
@pytest.mark.asyncio
async def test_tool_call_with_temperature(client: openai.AsyncOpenAI):
    tool_calls = await client.chat.completions.create(
        model=MODEL_NAME, messages=MESSAGES_CALC,
        tools=TOOLS, temperature=0.7, stream=False
    )
    calls = tool_calls.choices[0].message.tool_calls
    assert any(c.function.name == FUNC_CALC for c in calls)
