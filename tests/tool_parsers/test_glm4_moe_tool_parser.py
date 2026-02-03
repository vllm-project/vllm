# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import FunctionCall, ToolCall
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.glm4_moe_tool_parser import (
    Glm4MoeModelToolParser,
)

pytest.skip("skip glm4_moe parser test", allow_module_level=True)
# Use a common model that is likely to be available
MODEL = "zai-org/GLM-4.5"


@pytest.fixture(scope="module")
def glm4_moe_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def glm4_moe_tool_parser(glm4_moe_tokenizer):
    return Glm4MoeModelToolParser(glm4_moe_tokenizer)


def assert_tool_calls(
    actual_tool_calls: list[ToolCall], expected_tool_calls: list[ToolCall]
):
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(
        actual_tool_calls, expected_tool_calls
    ):
        assert isinstance(actual_tool_call.id, str)
        assert len(actual_tool_call.id) > 0

        assert actual_tool_call.type == "function"
        assert actual_tool_call.function.name == expected_tool_call.function.name
        # Compare arguments as JSON objects to handle formatting differences
        actual_args = json.loads(actual_tool_call.function.arguments)
        expected_args = json.loads(expected_tool_call.function.arguments)
        assert actual_args == expected_args


def test_extract_tool_calls_no_tools(glm4_moe_tool_parser):
    model_output = "This is a test"
    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output


@pytest.mark.parametrize(
    ids=[
        "single_tool_call",
        "multiple_tool_calls",
        "tool_call_with_content_before",
        "tool_call_with_mixed_args",
        "tool_call_with_chinese_content",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
            """<tool_call>get_current_weather
    <arg_key>city</arg_key>
    <arg_value>Dallas</arg_value>
    <arg_key>state</arg_key>
    <arg_value>TX</arg_value>
    <arg_key>unit</arg_key>
    <arg_value>fahrenheit</arg_value>
    </tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {
                                "city": "Dallas",
                                "state": "TX",
                                "unit": "fahrenheit",
                            }
                        ),
                    )
                )
            ],
            None,
        ),
        (
            """<tool_call>get_current_weather
    <arg_key>city</arg_key>
    <arg_value>Dallas</arg_value>
    <arg_key>state</arg_key>
    <arg_value>TX</arg_value>
    <arg_key>unit</arg_key>
    <arg_value>fahrenheit</arg_value>
    </tool_call>
    <tool_call>get_current_weather
    <arg_key>city</arg_key>
    <arg_value>Orlando</arg_value>
    <arg_key>state</arg_key>
    <arg_value>FL</arg_value>
    <arg_key>unit</arg_key>
    <arg_value>fahrenheit</arg_value>
    </tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {
                                "city": "Dallas",
                                "state": "TX",
                                "unit": "fahrenheit",
                            }
                        ),
                    )
                ),
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {
                                "city": "Orlando",
                                "state": "FL",
                                "unit": "fahrenheit",
                            }
                        ),
                    )
                ),
            ],
            None,
        ),
        (
            """I'll help you check the weather. <tool_call>get_current_weather
    <arg_key>city</arg_key>
    <arg_value>Seattle</arg_value>
    <arg_key>state</arg_key>
    <arg_value>WA</arg_value>
    <arg_key>unit</arg_key>
    <arg_value>celsius</arg_value>
    </tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {
                                "city": "Seattle",
                                "state": "WA",
                                "unit": "celsius",
                            }
                        ),
                    )
                )
            ],
            "I'll help you check the weather.",
        ),
        (
            """<tool_call>get_current_weather
    <arg_key>city</arg_key>
    <arg_value>New York</arg_value>
    <arg_key>state</arg_key>
    <arg_value>NY</arg_value>
    <arg_key>unit</arg_key>
    <arg_value>celsius</arg_value>
    </tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {
                                "city": "New York",
                                "state": "NY",
                                "unit": "celsius",
                            }
                        ),
                    )
                )
            ],
            None,
        ),
        (
            """I will help you get the weather.<tool_call>get_weather
    <arg_key>city</arg_key>
    <arg_value>Beijing</arg_value>
    <arg_key>date</arg_key>
    <arg_value>2025-08-01</arg_value>
    </tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_weather",
                        arguments=json.dumps(
                            {
                                "city": "Beijing",
                                "date": "2025-08-01",
                            }
                        ),
                    )
                )
            ],
            "I will help you get the weather.",
        ),
    ],
)
def test_extract_tool_calls(
    glm4_moe_tool_parser, model_output, expected_tool_calls, expected_content
):
    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
    assert extracted_tool_calls.tools_called
    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content


def test_extract_tool_calls_with_thinking_tags(glm4_moe_tool_parser):
    """Test tool extraction when thinking tags are present."""
    model_output = """<think>I want to get the weather.</think>

I will help you get the weather.
<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Beijing</arg_value>
<arg_key>date</arg_key>
<arg_value>2025-08-01</arg_value>
</tool_call>"""

    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 1
    assert extracted_tool_calls.tool_calls[0].function.name == "get_weather"

    expected_content = """<think>I want to get the weather.</think>

I will help you get the weather."""
    assert extracted_tool_calls.content == expected_content


def test_extract_tool_calls_malformed_xml(glm4_moe_tool_parser):
    """Test that malformed XML is handled gracefully."""
    model_output = """<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Seattle</arg_value>
<arg_key>incomplete_arg
<arg_value>value</arg_value>
</tool_call>"""

    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]

    # Should handle malformed XML gracefully
    # The parser should either extract what it can or return no tool calls
    # depending on how robust we want the parsing to be
    assert isinstance(extracted_tool_calls.tools_called, bool)
    assert isinstance(extracted_tool_calls.tool_calls, list)


def test_extract_tool_calls_empty_arguments(glm4_moe_tool_parser):
    """Test tool calls with no arguments."""
    model_output = """<tool_call>get_current_time
</tool_call>"""

    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 1
    assert extracted_tool_calls.tool_calls[0].function.name == "get_current_time"
    # Empty arguments should result in empty JSON object
    assert extracted_tool_calls.tool_calls[0].function.arguments == "{}"


def test_extract_tool_calls_mixed_content(glm4_moe_tool_parser):
    """Test extraction with mixed content and multiple tool calls."""
    model_output = """I will help you get the weather info.

<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Beijing</arg_value>
<arg_key>date</arg_key>
<arg_value>2025-08-01</arg_value>
</tool_call>

meaningwhile, I will also check the weather in Shanghai.

<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Shanghai</arg_value>
<arg_key>date</arg_key>
<arg_value>2025-08-01</arg_value>
</tool_call>"""

    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 2

    # Check first tool call
    assert extracted_tool_calls.tool_calls[0].function.name == "get_weather"
    args1 = json.loads(extracted_tool_calls.tool_calls[0].function.arguments)
    assert args1["city"] == "Beijing"
    assert args1["date"] == "2025-08-01"

    # Check second tool call
    assert extracted_tool_calls.tool_calls[1].function.name == "get_weather"
    args2 = json.loads(extracted_tool_calls.tool_calls[1].function.arguments)
    assert args2["city"] == "Shanghai"
    assert args2["date"] == "2025-08-01"

    # Content should be everything before the first tool call
    assert extracted_tool_calls.content == "I will help you get the weather info."


def test_streaming_basic_functionality(glm4_moe_tool_parser):
    """Test basic streaming functionality."""
    # Reset streaming state
    glm4_moe_tool_parser.current_tool_name_sent = False
    glm4_moe_tool_parser.prev_tool_call_arr = []
    glm4_moe_tool_parser.current_tool_id = -1
    glm4_moe_tool_parser.streamed_args_for_tool = []

    # Test with a simple tool call
    current_text = """<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Beijing</arg_value>
</tool_call>"""

    # Mock token IDs for testing
    tool_call_start_id = glm4_moe_tool_parser.tool_call_start_token_id or 12345
    tool_call_end_id = glm4_moe_tool_parser.tool_call_end_token_id or 12346

    result = glm4_moe_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=current_text,
        delta_text="</tool_call>",
        previous_token_ids=[],
        current_token_ids=[tool_call_start_id, tool_call_end_id],
        delta_token_ids=[tool_call_end_id],
        request=None,
    )

    # The result behavior depends on the streaming state
    # This test mainly ensures no exceptions are thrown
    assert result is None or hasattr(result, "tool_calls") or hasattr(result, "content")


def test_streaming_no_tool_calls(glm4_moe_tool_parser):
    """Test streaming when there are no tool calls."""
    current_text = "This is just regular text without any tool calls."

    result = glm4_moe_tool_parser.extract_tool_calls_streaming(
        previous_text="This is just regular text",
        current_text=current_text,
        delta_text=" without any tool calls.",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    # Should return the delta text as content
    assert result is not None
    assert hasattr(result, "content")
    assert result.content == " without any tool calls."


def test_streaming_with_content_before_tool_calls(glm4_moe_tool_parser):
    """Test streaming when there's content before tool calls."""
    # Reset streaming state
    glm4_moe_tool_parser.current_tool_name_sent = False
    glm4_moe_tool_parser.prev_tool_call_arr = []
    glm4_moe_tool_parser.current_tool_id = -1
    glm4_moe_tool_parser.streamed_args_for_tool = []

    current_text = "I will help you get the weather<tool_call>"

    result = glm4_moe_tool_parser.extract_tool_calls_streaming(
        previous_text="I will help you",
        current_text=current_text,
        delta_text="get the weather.<tool_call>",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    # Should return content when no tool call tokens are detected
    assert result is not None
    assert hasattr(result, "content")
    assert result.content == "get the weather.<tool_call>"


def test_extract_tool_calls_special_characters(glm4_moe_tool_parser):
    """Test tool calls with special characters and unicode."""
    model_output = """<tool_call>send_message
<arg_key>recipient</arg_key>
<arg_value>Amy</arg_value>
<arg_key>message</arg_key>
<arg_value>It is a nice day</arg_value>
<arg_key>priority</arg_key>
<arg_value>high</arg_value>
</tool_call>"""

    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 1
    assert extracted_tool_calls.tool_calls[0].function.name == "send_message"

    args = json.loads(extracted_tool_calls.tool_calls[0].function.arguments)
    assert args["recipient"] == "Amy"
    assert args["message"] == "It is a nice day"
    assert args["priority"] == "high"


def test_extract_tool_calls_incomplete_tool_call(glm4_moe_tool_parser):
    """Test incomplete tool calls (missing closing tag)."""
    model_output = """<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Beijing</arg_value>
<arg_key>date</arg_key>
<arg_value>2025-08-01</arg_value>"""

    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]

    # Incomplete tool calls should not be extracted
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output


def _reset_streaming_state(parser):
    """Helper to reset parser streaming state."""
    parser._buffer = ""
    parser._in_tool_call = False
    parser.current_tool_name_sent = False
    parser._current_tool_name = None
    parser._pending_key = None
    parser._streaming_string_value = False
    parser.prev_tool_call_arr = []
    parser.current_tool_id = -1
    parser.streamed_args_for_tool = []
    parser._tool_call_ids = []
    parser._args_started = []
    parser._args_closed = []
    parser._seen_keys = []


def test_streaming_incremental_string_value(glm4_moe_tool_parser):
    """Test incremental streaming of string argument values."""
    _reset_streaming_state(glm4_moe_tool_parser)

    # Simulate streaming a tool call character by character
    chunks = [
        "<tool_call>",
        "get_weather\n",
        "<arg_key>city</arg_key>",
        "<arg_value>",
        "Bei",
        "jing",
        "</arg_value>",
        "</tool_call>",
    ]

    collected_fragments = []
    for chunk in chunks:
        result = glm4_moe_tool_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="",
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )
        if result is not None and hasattr(result, "tool_calls") and result.tool_calls:
            for tc in result.tool_calls:
                if hasattr(tc, "function") and tc.function:
                    func = tc.function
                    if isinstance(func, dict):
                        if func.get("arguments"):
                            collected_fragments.append(func["arguments"])
                        if func.get("name"):
                            collected_fragments.append(f"name:{func['name']}")
                    else:
                        if func.arguments:
                            collected_fragments.append(func.arguments)
                        if func.name:
                            collected_fragments.append(f"name:{func.name}")

    # Verify we got incremental streaming of the argument value
    assert len(collected_fragments) > 0
    # The fragments should include the tool name and argument pieces
    combined = "".join(collected_fragments)
    assert "get_weather" in combined or "name:get_weather" in combined


def test_streaming_empty_tool_call(glm4_moe_tool_parser):
    """Test that empty tool calls don't cause infinite loops."""
    _reset_streaming_state(glm4_moe_tool_parser)

    # Empty tool call should be handled gracefully
    result = glm4_moe_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="",
        delta_text="<tool_call></tool_call>",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    # Should not hang and should return something (None or content)
    # The key is that this completes without hanging
    assert result is None or hasattr(result, "content") or hasattr(result, "tool_calls")
    # State should be properly reset
    assert glm4_moe_tool_parser.current_tool_id == -1


def test_streaming_prev_tool_call_arr_finalization(glm4_moe_tool_parser):
    """Test that prev_tool_call_arr contains parsed dict after tool call."""
    _reset_streaming_state(glm4_moe_tool_parser)

    # Stream a complete tool call
    chunks = [
        "<tool_call>get_weather\n",
        "<arg_key>city</arg_key>",
        "<arg_value>Beijing</arg_value>",
        "</tool_call>",
    ]

    for chunk in chunks:
        glm4_moe_tool_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="",
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )

    # After the tool call completes, prev_tool_call_arr should have parsed dict
    assert len(glm4_moe_tool_parser.prev_tool_call_arr) == 1
    tool_entry = glm4_moe_tool_parser.prev_tool_call_arr[0]
    assert tool_entry.get("name") == "get_weather"
    # arguments should be a dict, not a string
    args = tool_entry.get("arguments")
    assert isinstance(args, dict), f"Expected dict, got {type(args)}"
    assert args.get("city") == "Beijing"


def test_streaming_multiple_tool_calls_sequential(glm4_moe_tool_parser):
    """Test streaming multiple sequential tool calls."""
    _reset_streaming_state(glm4_moe_tool_parser)

    # Stream two tool calls
    chunks = [
        "<tool_call>get_weather\n",
        "<arg_key>city</arg_key>",
        "<arg_value>Beijing</arg_value>",
        "</tool_call>",
        "<tool_call>get_weather\n",
        "<arg_key>city</arg_key>",
        "<arg_value>Shanghai</arg_value>",
        "</tool_call>",
    ]

    for chunk in chunks:
        glm4_moe_tool_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="",
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )

    # Should have two tool calls in prev_tool_call_arr
    assert len(glm4_moe_tool_parser.prev_tool_call_arr) == 2
    assert glm4_moe_tool_parser.prev_tool_call_arr[0]["arguments"]["city"] == "Beijing"
    assert glm4_moe_tool_parser.prev_tool_call_arr[1]["arguments"]["city"] == "Shanghai"


def test_streaming_json_escape_in_string(glm4_moe_tool_parser):
    """Test that special characters in string values are properly escaped."""
    _reset_streaming_state(glm4_moe_tool_parser)

    # String with characters that need JSON escaping
    chunks = [
        "<tool_call>send_message\n",
        "<arg_key>message</arg_key>",
        '<arg_value>Hello "world"\nNew line</arg_value>',
        "</tool_call>",
    ]

    for chunk in chunks:
        glm4_moe_tool_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="",
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )

    # The streamed_args_for_tool should contain valid JSON
    assert len(glm4_moe_tool_parser.streamed_args_for_tool) == 1
    args_json = glm4_moe_tool_parser.streamed_args_for_tool[0]
    # Should be parseable as JSON
    parsed = json.loads(args_json)
    assert "message" in parsed
    # The value should preserve the special characters
    assert '"' in parsed["message"] or "world" in parsed["message"]


def test_streaming_long_content_incremental(glm4_moe_tool_parser):
    """Test incremental streaming of long content (Issue #32829).

    This is the core fix: for long string values like code (4000+ chars),
    the parser should stream incrementally rather than buffering until
    complete. This test verifies we get many fragments, not just 1-3.
    """
    _reset_streaming_state(glm4_moe_tool_parser)

    # Bubble sort example from Issue #32829 - realistic long content
    bubble_sort_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bubble Sort Implementation
"""

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

if __name__ == "__main__":
    test_arr = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original: {test_arr}")
    sorted_arr = bubble_sort(test_arr.copy())
    print(f"Sorted: {sorted_arr}")'''

    # Create a request with tool schema to enable string type detection
    # This is required for incremental streaming of string values
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "write_to_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                },
            }
        ],
    )

    # Simulate token-based streaming (special tags as single tokens)
    chunks = [
        "<tool_call>",
        "write_to_file\n",
        "<arg_key>file_path</arg_key>",
        "<arg_value>/tmp/bubble_sort.py</arg_value>",
        "<arg_key>content</arg_key>",
        "<arg_value>",
    ]
    # Add content line by line (realistic token streaming)
    for line in bubble_sort_code.split("\n"):
        chunks.append(line + "\n")
    chunks.append("</arg_value>")
    chunks.append("</tool_call>")

    # Count argument fragments
    fragment_count = 0
    for chunk in chunks:
        result = glm4_moe_tool_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="",
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        if result is not None and hasattr(result, "tool_calls") and result.tool_calls:
            for tc in result.tool_calls:
                if hasattr(tc, "function") and tc.function:
                    func = tc.function
                    args = (
                        func.get("arguments")
                        if isinstance(func, dict)
                        else getattr(func, "arguments", None)
                    )
                    if args:
                        fragment_count += 1

    # For true incremental streaming, we expect many fragments (10+)
    # Old buffered implementation would give only 1-3 fragments
    assert fragment_count >= 10, (
        f"Expected >=10 fragments for incremental streaming, got {fragment_count}"
    )

    # Verify final result is valid JSON
    assert len(glm4_moe_tool_parser.streamed_args_for_tool) == 1
    args_json = glm4_moe_tool_parser.streamed_args_for_tool[0]
    parsed = json.loads(args_json)
    assert parsed["file_path"] == "/tmp/bubble_sort.py"
    assert "def bubble_sort" in parsed["content"]


def test_extract_tool_calls_numeric_deserialization(glm4_moe_tool_parser):
    """Test that numeric arguments are deserialized as numbers, not strings."""
    model_output = """<tool_call>calculate
<arg_key>operation</arg_key>
<arg_value>add</arg_value>
<arg_key>a</arg_key>
<arg_value>42</arg_value>
<arg_key>b</arg_key>
<arg_value>3.14</arg_value>
<arg_key>enabled</arg_key>
<arg_value>true</arg_value>
</tool_call>"""

    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 1

    args = json.loads(extracted_tool_calls.tool_calls[0].function.arguments)

    # String should remain string
    assert args["operation"] == "add"
    assert isinstance(args["operation"], str)

    # Integer should be deserialized as int
    assert args["a"] == 42
    assert isinstance(args["a"], int)

    # Float should be deserialized as float
    assert args["b"] == 3.14
    assert isinstance(args["b"], float)

    # Boolean should be deserialized as bool
    assert args["enabled"] is True
    assert isinstance(args["enabled"], bool)
