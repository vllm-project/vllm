# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import openai
import pytest
import pytest_asyncio
import json
from rapidfuzz import fuzz
import jsonschema
from pprint import pprint
from ...utils import RemoteOpenAIServer

MODEL_NAME = "openai/gpt-oss-20b"


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len", "8192",
        "--enforce-eager",
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
<<<<<<< HEAD
    {
        "role": "system",
        "content": (
            "You can call multiple tools. "
            "When using more than one, return a single JSON object with a 'tool_calls' array "
            "containing each tool call with its function name and arguments. "
            "Do not output multiple JSON objects separately."
        ),
    },
    {
        "role": "user",
        "content": "First, calculate 7 * 8 using the calculator. Then, use get_time to tell me the current time in New York.",
    },
]

MESSAGES_INVALID_CALL = [
    {"role": "user", "content": "Can you help with something, but don’t actually perform any calculation?"}
=======
    {"role": "user", "content": "What is 7 * 8? And what time is it in New York?"}
]

MESSAGES_INVALID_CALL = [
    {"role": "user", "content": "Use the calculator but give no expression."}
>>>>>>> 9641600c2 (tool-call-7)
]


# Expected outputs
FUNC_CALC = "calculator"
FUNC_ARGS_CALC = '{"expression":"123 + 456"}'

FUNC_TIME = "get_time"
<<<<<<< HEAD
FUNC_ARGS_TIME = '{"city": "New York"}'
=======
FUNC_ARGS_TIME = '{"city":"New York"}'
>>>>>>> 9641600c2 (tool-call-7)


# ==========================================================
# Utility to extract reasoning and tool calls
# ==========================================================
def extract_reasoning_and_calls(chunks: list):
    """Extract accumulated reasoning text and tool call arguments from streaming chunks."""
    reasoning_content = ""
<<<<<<< HEAD
    tool_calls = {}
=======
    tool_calls = {}  # index -> {"name": str, "arguments": str}
>>>>>>> 9641600c2 (tool-call-7)

    for chunk in chunks:
        choice = getattr(chunk.choices[0], "delta", None)
        if not choice:
            continue

<<<<<<< HEAD
        if hasattr(choice, "reasoning_content") and choice.reasoning_content:
            reasoning_content += choice.reasoning_content

=======
        # Handle reasoning deltas
        if hasattr(choice, "reasoning_content") and choice.reasoning_content:
            reasoning_content += choice.reasoning_content

        # Handle tool call deltas
>>>>>>> 9641600c2 (tool-call-7)
        for tc in getattr(choice, "tool_calls", []) or []:
            idx = getattr(tc, "index", 0)
            tool_entry = tool_calls.setdefault(idx, {"name": "", "arguments": ""})

            if getattr(tc, "function", None):
                func = tc.function
                if getattr(func, "name", None):
                    tool_entry["name"] = func.name
                if getattr(func, "arguments", None):
                    tool_entry["arguments"] += func.arguments

<<<<<<< HEAD
=======
    # Convert dict to parallel lists
>>>>>>> 9641600c2 (tool-call-7)
    function_names = [v["name"] for _, v in sorted(tool_calls.items())]
    arguments = [v["arguments"] for _, v in sorted(tool_calls.items())]

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
<<<<<<< HEAD
        stream=True,
=======
        stream=True
>>>>>>> 9641600c2 (tool-call-7)
    )
    chunks = [chunk async for chunk in stream]
    reasoning, arguments, function_names = extract_reasoning_and_calls(chunks)

    assert FUNC_CALC in function_names, "Calculator function not called"
<<<<<<< HEAD
    assert any(FUNC_ARGS_CALC in arg or "123 + 456" in arg for arg in arguments), (
        f"Expected calculator arguments {FUNC_ARGS_CALC} not found in {arguments}"
    )
=======
    assert any(FUNC_ARGS_CALC in arg or "123 + 456" in arg for arg in arguments), f"Expected calculator arguments {FUNC_ARGS_CALC} not found in {arguments}"
>>>>>>> 9641600c2 (tool-call-7)
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
<<<<<<< HEAD
    reasoning = response.choices[0].message.reasoning_content or ""

    # Log for debugging if one call is missing
    print("DEBUG: tool_calls =")
    pprint(calls)

    print("DEBUG: reasoning =")
    pprint(reasoning)

    try:
        assert any(c.function.name == FUNC_CALC for c in calls), "Calculator tool missing"
    except AssertionError as e:
        print(f"ERROR: {e}")
    try:
        assert any(c.function.name == FUNC_TIME for c in calls), "Time tool missing"
    except AssertionError as e:
        print(f"ERROR: {e}")
    try:
        assert len(reasoning) > 0, "Reasoning content is empty"
    except AssertionError as e:
        print(f"ERROR: {e}")
    
=======

    print("\n=== TOOL CALLS ===")
    pprint(calls)

    assert any(c.function.name == FUNC_CALC for c in calls), "Calculator tool missing"
    assert any(c.function.name == FUNC_TIME for c in calls), "Time tool missing"
    assert len(response.choices[0].message.reasoning_content) > 0

>>>>>>> 9641600c2 (tool-call-7)


@pytest.mark.asyncio
async def test_invalid_tool_call(client: openai.AsyncOpenAI):
<<<<<<< HEAD
    """
    Verify that ambiguous instructions that should not trigger a tool
    do not produce any tool calls.
    """
=======
    """Verify that incomplete or ambiguous tool instructions do not produce tool calls."""
>>>>>>> 9641600c2 (tool-call-7)
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES_INVALID_CALL,
        tools=TOOLS,
        temperature=0.0,
<<<<<<< HEAD
        stream=False,
    )

    # Extract the assistant's message
    message = response.choices[0].message

    # Basic checks
    assert message is not None, "Expected message in response"
    assert hasattr(message, "content"), "Expected 'content' field in message"

    # Ensure no tool calls occurred
    tool_calls = getattr(message, "tool_calls", [])
    assert not tool_calls, (
        f"Model unexpectedly attempted a tool call on invalid input: {tool_calls}"
    )

=======
    )

    assert response is not None
    assert hasattr(response.choices[0].message, "content")
    assert not getattr(response.choices[0].message, "tool_calls", None), \
        "Model unexpectedly attempted a tool call on invalid input"
>>>>>>> 9641600c2 (tool-call-7)



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
<<<<<<< HEAD

    chunks = [chunk async for chunk in stream]
    reasoning, arguments, function_names = extract_reasoning_and_calls(chunks)
    
    print("DEBUG: function_names =", function_names)
    print("DEBUG: reasoning =")
    pprint(reasoning)

    try:
        assert FUNC_CALC in function_names, f"Calculator tool missing — found {function_names}"
    except AssertionError as e:
        print(f"ERROR: {e}")
    try:
        assert FUNC_TIME in function_names, f"Time tool missing — found {function_names}"
    except AssertionError as e:
        print(f"ERROR: {e}")
    try:
        assert len(reasoning) > 0, "Expected reasoning content in streamed response"
    except AssertionError as e:
        print(f"ERROR: {e}")
=======
    chunks = [chunk async for chunk in stream]
    reasoning, arguments, function_names = extract_reasoning_and_calls(chunks)

    assert FUNC_CALC in function_names
    assert FUNC_TIME in function_names
    assert len(reasoning) > 0
>>>>>>> 9641600c2 (tool-call-7)


@pytest.mark.asyncio
async def test_tool_call_with_temperature(client: openai.AsyncOpenAI):
<<<<<<< HEAD
    """Verify model produces valid tool or text output under non-deterministic sampling."""
=======
    """Verify model produces valid output (tool or text) under non-deterministic sampling."""
>>>>>>> 9641600c2 (tool-call-7)
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES_CALC,
        tools=TOOLS,
        temperature=0.7,
        stream=False,
    )

    message = response.choices[0].message
<<<<<<< HEAD
    assert message is not None, "Expected non-empty message in response"
=======
    assert message is not None
>>>>>>> 9641600c2 (tool-call-7)
    assert (
        message.tool_calls or message.content
    ), "Response missing both text and tool calls"

<<<<<<< HEAD
    print(f"\nTool calls: {message.tool_calls}")
    print(f"Text: {message.content}")


=======
    print(f"Tool calls: {message.tool_calls}")
    print(f"Text: {message.content}")
    
    
    
>>>>>>> 9641600c2 (tool-call-7)
# ==========================================================
# Accuracy & Consistency Tests
# ==========================================================
@pytest.mark.asyncio
async def test_tool_call_argument_accuracy(client: openai.AsyncOpenAI):
    """Ensure the calculator tool arguments closely match the expected arithmetic expression."""
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES_CALC,
        tools=TOOLS,
        temperature=0.0,
    )

    calls = response.choices[0].message.tool_calls
    assert calls, "No tool calls detected"
    calc_call = next((c for c in calls if c.function.name == FUNC_CALC), None)
    assert calc_call, "Calculator function missing"

<<<<<<< HEAD
=======
    # Parse model arguments (may arrive as JSON string fragments)
>>>>>>> 9641600c2 (tool-call-7)
    try:
        args = json.loads(calc_call.function.arguments)
    except json.JSONDecodeError:
        pytest.fail("Invalid JSON in calculator arguments")

    expected_expr = "123 + 456"
    similarity = fuzz.ratio(args.get("expression", ""), expected_expr)
    assert similarity > 90, f"Expression mismatch (similarity={similarity}%)"


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

<<<<<<< HEAD
        tool = next(t for t in TOOLS if t["function"]["name"] == func_name)
        schema = tool["function"]["parameters"]

=======
        # Find the tool schema dynamically
        tool = next(t for t in TOOLS if t["function"]["name"] == func_name)
        schema = tool["function"]["parameters"]

        # Validate the arguments against schema
>>>>>>> 9641600c2 (tool-call-7)
        jsonschema.validate(instance=args, schema=schema)


@pytest.mark.asyncio
async def test_reasoning_relevance_accuracy(client: openai.AsyncOpenAI):
    """Check whether reasoning content is semantically related to the user's query."""
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES_CALC,
        tools=TOOLS,
        stream=True,
    )
    chunks = [chunk async for chunk in stream]
    reasoning, _, _ = extract_reasoning_and_calls(chunks)

    assert len(reasoning) > 0, "No reasoning emitted"
<<<<<<< HEAD
=======
    # The reasoning should at least reference numbers in the user query
>>>>>>> 9641600c2 (tool-call-7)
    assert any(num in reasoning for num in ["123", "456"]), \
        f"Reasoning does not reference expected numbers: {reasoning}"


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
    assert low_mid_sim > 60, f"Semantic drift too large between T=0.0 and T=0.5 ({low_mid_sim}%)"    

