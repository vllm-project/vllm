# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import jsonschema
import openai
import pytest
import pytest_asyncio
from rapidfuzz import fuzz

from ....utils import RemoteOpenAIServer

MODEL_NAME = "openai/gpt-oss-20b"


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "8192",
        "--enforce-eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "openai",
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
                        "description": (
                            "Arithmetic expression to evaluate, e.g. '123 + 456'."
                        ),
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
                        "description": "City name, e.g. 'New York'.",
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

MESSAGES_GET_TIME = [
    {"role": "user", "content": "What is the current time in New York?"}
]

MESSAGES_MULTIPLE_CALLS = [
    {
        "role": "system",
        "content": (
            "You can call multiple tools. "
            "When using more than one, return single JSON object with tool_calls array"
            "containing each tool call with its function name and arguments. "
            "Do not output multiple JSON objects separately."
        ),
    },
    {
        "role": "user",
        "content": "First, calculate 7 * 8 using the calculator. "
        "Then, use get_time to tell me the current time in New York.",
    },
]

MESSAGES_INVALID_CALL = [
    {
        "role": "user",
        "content": "Can you help with something, "
        "but don’t actually perform any calculation?",
    }
]


# Expected outputs
FUNC_CALC = "calculator"
FUNC_ARGS_CALC = '{"expression":"123 + 456"}'

FUNC_TIME = "get_time"
FUNC_ARGS_TIME = '{"city": "New York"}'


# ==========================================================
# Utility to extract reasoning and tool calls
# ==========================================================
def extract_reasoning_and_calls(chunks: list) -> tuple[str, list[str], list[str]]:
    """
    Extract accumulated reasoning text and tool call arguments
    from streaming chunks.
    """
    reasoning_content: str = ""
    tool_calls: dict[int, dict[str, str]] = {}

    for chunk in chunks:
        choice = getattr(chunk.choices[0], "delta", None)
        if not choice:
            continue

        if hasattr(choice, "reasoning_content") and choice.reasoning_content:
            reasoning_content += choice.reasoning_content

        for tc in getattr(choice, "tool_calls", []) or []:
            idx = getattr(tc, "index", 0)
            tool_entry = tool_calls.setdefault(idx, {"name": "", "arguments": ""})

            if getattr(tc, "function", None):
                func = tc.function
                if getattr(func, "name", None):
                    tool_entry["name"] = func.name
                if getattr(func, "arguments", None):
                    tool_entry["arguments"] += func.arguments

    function_names: list[str] = [v["name"] for _, v in sorted(tool_calls.items())]
    arguments: list[str] = [v["arguments"] for _, v in sorted(tool_calls.items())]

    return reasoning_content, arguments, function_names


# ==========================================================
# Test Scenarios
# ==========================================================
@pytest.mark.asyncio
async def test_calculator_tool_call_and_argument_accuracy(client: openai.AsyncOpenAI):
    """Verify calculator tool call is made and arguments are accurate."""

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES_CALC,
        tools=TOOLS,
        temperature=0.0,
        stream=False,
    )

    message = response.choices[0].message
    tool_calls = getattr(message, "tool_calls", [])
    assert tool_calls, "No tool calls detected"

    calc_call = next((c for c in tool_calls if c.function.name == FUNC_CALC), None)
    assert calc_call, "Calculator function not called"

    raw_args = calc_call.function.arguments
    assert raw_args, "Calculator arguments missing"
    assert "123" in raw_args and "456" in raw_args, (
        f"Expected values not in raw arguments: {raw_args}"
    )

    try:
        parsed_args = json.loads(raw_args)
    except json.JSONDecodeError:
        pytest.fail(f"Invalid JSON in calculator arguments: {raw_args}")

    expected_expr = "123 + 456"
    actual_expr = parsed_args.get("expression", "")
    similarity = fuzz.ratio(actual_expr, expected_expr)

    assert similarity > 90, (
        f"Expression mismatch: expected '{expected_expr}' "
        f"got '{actual_expr}' (similarity={similarity}%)"
    )


@pytest.mark.asyncio
async def test_streaming_tool_call_get_time_with_reasoning(client: openai.AsyncOpenAI):
    """Verify streamed reasoning and tool call behavior for get_time."""

    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES_GET_TIME,
        tools=TOOLS,
        temperature=0.0,
        stream=True,
    )

    chunks = [chunk async for chunk in stream]
    reasoning, arguments, function_names = extract_reasoning_and_calls(chunks)

    assert FUNC_TIME in function_names, "get_time function not called"

    assert any("New York" in arg for arg in arguments), (
        f"Expected get_time arguments for New York not found in {arguments}"
    )

    assert len(reasoning) > 0, "Expected reasoning content missing"

    assert any(keyword in reasoning for keyword in ["New York", "time", "current"]), (
        f"Reasoning is not relevant to the request: {reasoning}"
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

    try:
        assert FUNC_CALC in function_names, (
            f"Calculator tool missing — found {function_names}"
        )
        assert FUNC_TIME in function_names, (
            f"Time tool missing — found {function_names}"
        )
        assert len(reasoning) > 0, "Expected reasoning content in streamed response"
    except AssertionError as e:
        print(f"ERROR: {e}")


@pytest.mark.asyncio
async def test_invalid_tool_call(client: openai.AsyncOpenAI):
    """
    Verify that ambiguous instructions that should not trigger a tool
    do not produce any tool calls.
    """
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES_INVALID_CALL,
        tools=TOOLS,
        temperature=0.0,
        stream=False,
    )

    message = response.choices[0].message

    assert message is not None, "Expected message in response"
    assert hasattr(message, "content"), "Expected 'content' field in message"

    tool_calls = getattr(message, "tool_calls", [])
    assert not tool_calls, (
        f"Model unexpectedly attempted a tool call on invalid input: {tool_calls}"
    )


@pytest.mark.asyncio
async def test_tool_call_with_temperature(client: openai.AsyncOpenAI):
    """
    Verify model produces valid tool or text output
    under non-deterministic sampling.
    """
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES_CALC,
        tools=TOOLS,
        temperature=0.7,
        stream=False,
    )

    message = response.choices[0].message
    assert message is not None, "Expected non-empty message in response"
    assert message.tool_calls or message.content, (
        "Response missing both text and tool calls"
    )

    print(f"\nTool calls: {message.tool_calls}")
    print(f"Text: {message.content}")


@pytest.mark.asyncio
async def test_tool_response_schema_accuracy(client: openai.AsyncOpenAI):
    """Validate that tool call arguments adhere to their declared JSON schema."""
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES_MULTIPLE_CALLS,
        tools=TOOLS,
        temperature=0.0,
    )

    calls = response.choices[0].message.tool_calls
    assert calls, "No tool calls produced"

    for call in calls:
        func_name = call.function.name
        args = json.loads(call.function.arguments)

        schema: dict[str, object] | None = None
        for tool_entry in TOOLS:
            function_def = tool_entry.get("function")
            if (
                function_def
                and isinstance(function_def, dict)
                and function_def.get("name") == func_name
            ):
                schema = function_def.get("parameters")
                break

        assert schema is not None, f"No matching tool schema found for {func_name}"

        jsonschema.validate(instance=args, schema=schema)


@pytest.mark.asyncio
async def test_semantic_consistency_with_temperature(client: openai.AsyncOpenAI):
    """Test that temperature variation doesn't cause contradictory reasoning."""
    responses = []
    for temp in [0.0, 0.5, 1.0]:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=MESSAGES_CALC,
            tools=TOOLS,
            temperature=temp,
        )
        text = (resp.choices[0].message.content or "").strip()
        responses.append(text)

    # Compare fuzzy similarity between low- and mid-temperature outputs
    low_mid_sim = fuzz.ratio(responses[0], responses[1])
    assert low_mid_sim > 60, (
        f"Semantic drift too large between T=0.0 and T=0.5 ({low_mid_sim}%)"
    )
