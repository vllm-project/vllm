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
    """Async fixture providing an OpenAI-compatible vLLM client."""
    async with server.get_async_client() as async_client:
        yield async_client


# ==========================================================
# Tool Definitions
# ==========================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Performs basic arithmetic calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression to evaluate, e.g. '123 + 456'."
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Retrieves the current local time for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'New York'."
                    }
                },
                "required": ["city"],
            },
        },
    },
]


# ==========================================================
# Message Examples
# ==========================================================
MESSAGES_CALC = [
    {"role": "user", "content": "Calculate 123 + 456 using the calculator."}
]

MESSAGES_MULTIPLE_CALLS = [
    {"role": "user", "content": "What is 7 * 8? And what time is it in New York?"}
]

MESSAGES_INVALID_CALL = [
    {"role": "user", "content": "Use the calculator but give no expression."}
]


# Expected outputs
FUNC_CALC = "calculator"
FUNC_ARGS_CALC = '{"expression": "123 + 456"}'

FUNC_TIME = "get_time"
FUNC_ARGS_TIME = '{"city": "New York"}'


# ==========================================================
# Utility to extract reasoning and tool calls
# ==========================================================
def extract_reasoning_and_calls(chunks: list):
    reasoning_content = ""
    tool_call_idx = -1
    arguments = []
    function_names = []

    for chunk in chunks:
        choice = getattr(chunk.choices[0], "delta", None)
        if not choice:
            continue

        if getattr(choice, "tool_calls", None):
            tool_call = choice.tool_calls[0]
            if tool_call.index != tool_call_idx:
                tool_call_idx = tool_call.index
                arguments.append("")
                function_names.append("")
            if tool_call.function:
                if tool_call.function.name:
                    function_names[tool_call_idx] = tool_call.function.name
                if tool_call.function.arguments:
                    arguments[tool_call_idx] += tool_call.function.arguments
        elif hasattr(choice, "reasoning_content"):
            reasoning_content += choice.reasoning_content

    return reasoning_content, arguments, function_names


# ==========================================================
# Test Scenarios
# ==========================================================
@pytest.mark.asyncio
async def test_single_tool_call(client: openai.AsyncOpenAI):
    """Verify single tool call reasoning with the calculator."""
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES_CALC,
        tools=TOOLS,
        temperature=0.0,
        stream=True
    )
    chunks = [chunk async for chunk in stream]
    reasoning, arguments, function_names = extract_reasoning_and_calls(chunks)

    assert FUNC_CALC in function_names, "Calculator function not called"
    assert FUNC_ARGS_CALC in arguments, "Calculator arguments mismatch"
    assert len(reasoning) > 0, "Expected reasoning content missing"


@pytest.mark.asyncio
async def test_multiple_tool_calls(client: openai.AsyncOpenAI):
    """Verify model handles multiple tools in one query."""
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES_MULTIPLE_CALLS,
        tools=TOOLS,
        temperature=0.0,
        stream=False,
    )

    calls = response.choices[0].message.tool_calls
    assert any(c.function.name == FUNC_CALC for c in calls), "Calculator tool missing"
    assert any(c.function.name == FUNC_TIME for c in calls), "Time tool missing"
    assert len(response.choices[0].message.reasoning_content) > 0


@pytest.mark.asyncio
async def test_invalid_tool_call(client: openai.AsyncOpenAI):
    """Ensure invalid tool parameters raise an exception."""
    with pytest.raises(Exception):
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=MESSAGES_INVALID_CALL,
            tools=TOOLS,
            temperature=0.0,
        )


@pytest.mark.asyncio
async def test_streaming_multiple_tools(client: openai.AsyncOpenAI):
    """Test streamed multi-tool response with reasoning."""
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES_MULTIPLE_CALLS,
        tools=TOOLS,
        temperature=0.0,
        stream=True,
    )
    chunks = [chunk async for chunk in stream]
    reasoning, arguments, function_names = extract_reasoning_and_calls(chunks)

    assert FUNC_CALC in function_names
    assert FUNC_TIME in function_names
    assert len(reasoning) > 0


@pytest.mark.asyncio
async def test_tool_call_with_temperature(client: openai.AsyncOpenAI):
    """Check tool-call behavior under non-deterministic sampling."""
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES_CALC,
        tools=TOOLS,
        temperature=0.7,
        stream=False,
    )

    calls = response.choices[0].message.tool_calls
    assert any(c.function.name == FUNC_CALC for c in calls)
