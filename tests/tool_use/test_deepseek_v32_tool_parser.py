# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.deepseek_v32_tool_parser import (
    DeepSeekV32ToolParser,
)
from vllm.tokenizers import get_tokenizer

pytestmark = pytest.mark.cpu_test

MODEL = "deepseek-ai/DeepSeek-V3"


@pytest.fixture(scope="module")
def deepseek_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def deepseek_tool_parser(deepseek_tokenizer):
    return DeepSeekV32ToolParser(deepseek_tokenizer)


@pytest.fixture
def sample_tools():
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"},
                        "date": {"type": "string", "description": "The date"},
                    },
                    "required": ["location", "date"],
                },
            },
        ),
    ]


def assert_tool_calls(
    actual_tool_calls: list[ToolCall], expected_tool_calls: list[ToolCall]
):
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(
        actual_tool_calls, expected_tool_calls
    ):
        assert actual_tool_call.type == "function"
        assert actual_tool_call.function.name == expected_tool_call.function.name
        try:
            assert json.loads(actual_tool_call.function.arguments) == json.loads(
                expected_tool_call.function.arguments
            )
        except json.JSONDecodeError as e:
            print(e)
            print("actual_tool_call", actual_tool_call.function.arguments)
            print("expected_tool_call", expected_tool_call.function.arguments)


def test_extract_tool_calls_single_function(
    deepseek_tool_parser,
    sample_tools,
):
    """Test extracting a single function call"""
    model_output = """<｜DSML｜function_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location" string="true">Hangzhou</｜DSML｜parameter>
<｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"""

    expected_tool_calls = [
        ToolCall(
            function=FunctionCall(
                name="get_weather",
                arguments=json.dumps({"location": "Hangzhou", "date": "2024-01-16"}),
            )
        ),
    ]

    request = ChatCompletionRequest(
        model="deepseek-v3",
        messages=[{"role": "user", "content": "What is the weather?"}],
        tools=sample_tools,
    )

    extracted = deepseek_tool_parser.extract_tool_calls(model_output, request)

    assert extracted.tools_called
    assert extracted.content is None
    assert_tool_calls(extracted.tool_calls, expected_tool_calls)


def test_extract_tool_calls_multiple_functions(
    deepseek_tool_parser,
    sample_tools,
):
    """Test extracting multiple function calls"""
    model_output = """<｜DSML｜function_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location" string="true">Hangzhou</｜DSML｜parameter>
<｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
</｜DSML｜invoke>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>
<｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"""

    expected_tool_calls = [
        ToolCall(
            function=FunctionCall(
                name="get_weather",
                arguments=json.dumps({"location": "Hangzhou", "date": "2024-01-16"}),
            )
        ),
        ToolCall(
            function=FunctionCall(
                name="get_weather",
                arguments=json.dumps({"location": "Beijing", "date": "2024-01-16"}),
            )
        ),
    ]

    request = ChatCompletionRequest(
        model="deepseek-v3",
        messages=[{"role": "user", "content": "What is the weather?"}],
        tools=sample_tools,
    )

    extracted = deepseek_tool_parser.extract_tool_calls(model_output, request)

    assert extracted.tools_called
    assert extracted.content is None
    assert extracted.tool_calls[0].id != extracted.tool_calls[1].id
    assert_tool_calls(extracted.tool_calls, expected_tool_calls)


def test_extract_tool_calls_with_end_of_sentence_token(
    deepseek_tool_parser,
    sample_tools,
):
    """Test extracting function calls with end-of-sentence token"""
    model_output = """<｜DSML｜function_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location" string="true">Hangzhou</｜DSML｜parameter>
<｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls><｜end▁of▁sentence｜>"""

    expected_tool_calls = [
        ToolCall(
            function=FunctionCall(
                name="get_weather",
                arguments=json.dumps({"location": "Hangzhou", "date": "2024-01-16"}),
            )
        ),
    ]

    request = ChatCompletionRequest(
        model="deepseek-v3",
        messages=[{"role": "user", "content": "What is the weather?"}],
        tools=sample_tools,
    )

    extracted = deepseek_tool_parser.extract_tool_calls(model_output, request)

    assert extracted.tools_called
    assert extracted.content is None
    assert_tool_calls(extracted.tool_calls, expected_tool_calls)


def test_extract_tool_calls_streaming(
    deepseek_tool_parser,
    sample_tools,
):
    """Test streaming extraction of function calls"""
    # Simulate streaming chunks
    chunks = [
        "<｜DSML｜function_calls>",
        '\n<｜DSML｜invoke name="get_weather">',
        '\n<｜DSML｜parameter name="location" string="true">',
        "Hangzhou",
        "</｜DSML｜parameter>",
        '\n<｜DSML｜parameter name="date" string="true">',
        "2024-01-16",
        "</｜DSML｜parameter>",
        "\n</｜DSML｜invoke>",
        "\n</｜DSML｜function_calls>",
    ]

    request = ChatCompletionRequest(
        model="deepseek-v3",
        messages=[{"role": "user", "content": "What is the weather?"}],
        tools=sample_tools,
    )

    # Track accumulated state
    tool_states = {}
    previous_text = ""

    for chunk in chunks:
        current_text = previous_text + chunk
        delta_text = chunk

        # Call streaming extraction
        delta_message = deepseek_tool_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )

        if delta_message and delta_message.tool_calls:
            for tool_call in delta_message.tool_calls:
                idx = tool_call.index if tool_call.index is not None else 0

                # Initialize state for new tool
                if idx not in tool_states:
                    tool_states[idx] = {
                        "id": None,
                        "name": None,
                        "arguments": "",
                        "type": None,
                    }

                # Update state
                if tool_call.id:
                    tool_states[idx]["id"] = tool_call.id

                if tool_call.type:
                    tool_states[idx]["type"] = tool_call.type

                if tool_call.function:
                    if tool_call.function.name:
                        tool_states[idx]["name"] = tool_call.function.name

                    if tool_call.function.arguments is not None:
                        tool_states[idx]["arguments"] += tool_call.function.arguments

        previous_text = current_text

    # Verify final state
    assert len(tool_states) == 1, f"Expected 1 tool call, got {len(tool_states)}"

    state = tool_states[0]
    assert state["id"] is not None, "Tool call ID should be set"
    assert state["type"] == "function", f"Expected type 'function', got {state['type']}"
    assert state["name"] == "get_weather", (
        f"Expected name 'get_weather', got {state['name']}"
    )
    assert state["arguments"] is not None
    # Verify arguments
    arguments = json.loads(state["arguments"])
    assert arguments == {
        "location": "Hangzhou",
        "date": "2024-01-16",
    }, f"Unexpected arguments: {arguments}"


def test_extract_tool_calls_streaming_multiple_functions(
    deepseek_tool_parser,
    sample_tools,
):
    """Test streaming extraction of multiple function calls"""
    # Simulate streaming chunks for two function calls
    chunks = [
        "<｜DSML｜function_calls>",
        '\n<｜DSML｜invoke name="get_weather">',
        '\n<｜DSML｜parameter name="location" string="true">Hangzhou</｜DSML｜parameter>',  # noqa: E501
        '\n<｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>',
        "\n</｜DSML｜invoke>",
        '\n<｜DSML｜invoke name="get_weather">',
        '\n<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>',  # noqa: E501
        '\n<｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>',
        "\n</｜DSML｜invoke>",
        "\n</｜DSML｜function_calls>",
    ]

    request = ChatCompletionRequest(
        model="deepseek-v3",
        messages=[{"role": "user", "content": "What is the weather?"}],
        tools=sample_tools,
    )

    # Track accumulated state
    tool_states = {}
    previous_text = ""

    for chunk in chunks:
        current_text = previous_text + chunk
        delta_text = chunk
        # Call streaming extraction
        delta_message = deepseek_tool_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )

        if delta_message and delta_message.tool_calls:
            for tool_call in delta_message.tool_calls:
                idx = tool_call.index if tool_call.index is not None else 0

                # Initialize state for new tool
                if idx not in tool_states:
                    tool_states[idx] = {
                        "id": None,
                        "name": None,
                        "arguments": "",
                        "type": None,
                    }

                # Update state
                if tool_call.id:
                    tool_states[idx]["id"] = tool_call.id

                if tool_call.type:
                    tool_states[idx]["type"] = tool_call.type

                if tool_call.function:
                    if tool_call.function.name:
                        tool_states[idx]["name"] = tool_call.function.name

                    if tool_call.function.arguments is not None:
                        tool_states[idx]["arguments"] += tool_call.function.arguments

        previous_text = current_text

    # Verify final state
    assert len(tool_states) == 2, f"Expected 2 tool calls, got {len(tool_states)}"

    # Verify first tool call
    state0 = tool_states[0]
    assert state0["id"] is not None
    assert state0["type"] == "function"
    assert state0["name"] == "get_weather"
    assert state0["arguments"] is not None
    arguments0 = json.loads(state0["arguments"])
    assert arguments0 == {"location": "Hangzhou", "date": "2024-01-16"}

    # Verify second tool call
    state1 = tool_states[1]
    assert state1["id"] is not None
    assert state1["id"] != state0["id"]
    assert state1["type"] == "function"
    assert state1["name"] == "get_weather"
    assert state1["arguments"] is not None
    arguments1 = json.loads(state1["arguments"])
    assert arguments1 == {"location": "Beijing", "date": "2024-01-16"}


def test_extract_tool_calls_streaming_incomplete_chunk_functions(
    deepseek_tool_parser,
    sample_tools,
):
    """Test streaming extraction of multiple function calls"""
    # Simulate streaming chunks for two function calls
    chunks = [
        "<｜DSML",
        "｜function_calls>",
        '\n<｜DSML｜invoke name="get_current_weather">',
        '\n<｜DSML｜parameter name="location" string="true">北京</｜DSML｜parameter>',
        '\n<｜DSML｜parameter name="time" string="true">2025-03-21</｜DSML｜parameter>',
        "\n</｜DSML｜invoke>",
        "</｜DSML｜function_calls>",
    ]

    request = ChatCompletionRequest(
        model="deepseek-v3",
        messages=[{"role": "user", "content": "What is the weather?"}],
        tools=sample_tools,
    )

    # Track accumulated state
    tool_states = {}
    previous_text = ""

    for chunk in chunks:
        current_text = previous_text + chunk
        delta_text = chunk
        # Call streaming extraction
        delta_message = deepseek_tool_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )

        if delta_message and delta_message.tool_calls:
            for tool_call in delta_message.tool_calls:
                idx = tool_call.index if tool_call.index is not None else 0

                # Initialize state for new tool
                if idx not in tool_states:
                    tool_states[idx] = {
                        "id": None,
                        "name": None,
                        "arguments": "",
                        "type": None,
                    }

                # Update state
                if tool_call.id:
                    tool_states[idx]["id"] = tool_call.id

                if tool_call.type:
                    tool_states[idx]["type"] = tool_call.type

                if tool_call.function:
                    if tool_call.function.name:
                        tool_states[idx]["name"] = tool_call.function.name

                    if tool_call.function.arguments is not None:
                        tool_states[idx]["arguments"] += tool_call.function.arguments

        previous_text = current_text

    # Verify final state
    assert len(tool_states) == 1, f"Expected 1 tool calls, got {len(tool_states)}"

    # Verify first tool call
    state0 = tool_states[0]
    assert state0["id"] is not None
    assert state0["type"] == "function"
    assert state0["name"] == "get_current_weather"
    assert state0["arguments"] is not None
    arguments0 = json.loads(state0["arguments"])
    assert arguments0 == {"location": "北京", "time": "2025-03-21"}


def test_extract_tool_calls_streaming_incomplete_chunk_function2(
    deepseek_tool_parser,
    sample_tools,
):
    """Test streaming extraction of multiple function calls"""
    # Simulate streaming chunks for two function calls
    chunks = [
        "<｜DSML｜function_calls>",
        '<｜DSML｜invoke name="get_current_weather">',
        '<｜DSML｜parameter name="location" string="true">北京</｜DSML｜parameter>',
        '<｜DSML｜parameter name="time" string="true">2025-03-21</｜DSML｜parameter>',
        "</｜DSML｜invoke>",
        '<｜DSML｜invoke name="get_current_weather">',
        '<｜DSML｜parameter name="location" string="true">北京</｜DSML｜parameter>',
        "</｜DSML｜invoke>",
        " </｜DSML｜function_calls>",
    ]

    request = ChatCompletionRequest(
        model="deepseek-v3",
        messages=[{"role": "user", "content": "What is the weather?"}],
        tools=sample_tools,
    )

    # Track accumulated state
    tool_states = {}
    previous_text = ""

    for chunk in chunks:
        current_text = previous_text + chunk
        delta_text = chunk
        # Call streaming extraction
        delta_message = deepseek_tool_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        print("aaa", delta_message)
        if delta_message and delta_message.tool_calls:
            for tool_call in delta_message.tool_calls:
                idx = tool_call.index if tool_call.index is not None else 0

                # Initialize state for new tool
                if idx not in tool_states:
                    tool_states[idx] = {
                        "id": None,
                        "name": None,
                        "arguments": "",
                        "type": None,
                    }

                # Update state
                if tool_call.id:
                    tool_states[idx]["id"] = tool_call.id

                if tool_call.type:
                    tool_states[idx]["type"] = tool_call.type

                if tool_call.function:
                    if tool_call.function.name:
                        tool_states[idx]["name"] = tool_call.function.name

                    if tool_call.function.arguments is not None:
                        tool_states[idx]["arguments"] += tool_call.function.arguments

        previous_text = current_text

    # Verify final state
    assert len(tool_states) == 2, f"Expected 2 tool calls, got {len(tool_states)}"

    # Verify first tool call
    state0 = tool_states[0]
    assert state0["id"] is not None
    assert state0["type"] == "function"
    assert state0["name"] == "get_current_weather"
    assert state0["arguments"] is not None
    arguments0 = json.loads(state0["arguments"])
    assert arguments0 == {"location": "北京", "time": "2025-03-21"}

    # Verify second tool call
    state0 = tool_states[1]
    assert state0["id"] is not None
    assert state0["type"] == "function"
    assert state0["name"] == "get_current_weather"
    assert state0["arguments"] is not None
    arguments0 = json.loads(state0["arguments"])
    assert arguments0 == {"location": "北京"}
