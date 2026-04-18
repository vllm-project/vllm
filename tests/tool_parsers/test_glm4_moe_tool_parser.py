# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import Mock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.entrypoints.openai.engine.protocol import FunctionCall, ToolCall
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.glm4_moe_tool_parser import (
    Glm4MoeModelToolParser,
)

# Use a common model that is likely to be available
MODEL = "zai-org/GLM-4.5"


@pytest.fixture(scope="module")
def glm4_moe_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def sample_tools():
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={"city": {"type": "string"}},
            ),
        ),
    ]


@pytest.fixture
def glm4_moe_tool_parser(glm4_moe_tokenizer, sample_tools):
    return Glm4MoeModelToolParser(glm4_moe_tokenizer, tools=sample_tools)


@pytest.fixture
def mock_request(sample_tools) -> ChatCompletionRequest:
    request = Mock(spec=ChatCompletionRequest)
    request.tools = sample_tools
    return request


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


def test_extract_tool_calls_no_tools(glm4_moe_tool_parser, mock_request):
    model_output = "This is a test"
    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=mock_request
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
            "I'll help you check the weather. ",
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
    glm4_moe_tool_parser,
    mock_request,
    model_output,
    expected_tool_calls,
    expected_content,
):
    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=mock_request
    )  # type: ignore[arg-type]
    assert extracted_tool_calls.tools_called
    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content


def test_extract_tool_calls_with_thinking_tags(glm4_moe_tool_parser, mock_request):
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
        model_output, request=mock_request
    )  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 1
    assert extracted_tool_calls.tool_calls[0].function.name == "get_weather"

    expected_content = """<think>I want to get the weather.</think>

I will help you get the weather.
"""
    assert extracted_tool_calls.content == expected_content


def test_extract_tool_calls_malformed_xml(glm4_moe_tool_parser, mock_request):
    """Test that malformed XML is handled gracefully."""
    model_output = """<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Seattle</arg_value>
<arg_key>incomplete_arg
<arg_value>value</arg_value>
</tool_call>"""

    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=mock_request
    )  # type: ignore[arg-type]

    # Should handle malformed XML gracefully
    # The parser should either extract what it can or return no tool calls
    # depending on how robust we want the parsing to be
    assert isinstance(extracted_tool_calls.tools_called, bool)
    assert isinstance(extracted_tool_calls.tool_calls, list)


def test_extract_tool_calls_empty_arguments(glm4_moe_tool_parser, mock_request):
    """Test tool calls with no arguments."""
    model_output = """<tool_call>get_current_time
</tool_call>"""

    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=mock_request
    )  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 1
    assert extracted_tool_calls.tool_calls[0].function.name == "get_current_time"
    # Empty arguments should result in empty JSON object
    assert extracted_tool_calls.tool_calls[0].function.arguments == "{}"


def test_extract_tool_calls_mixed_content(glm4_moe_tool_parser, mock_request):
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
        model_output, request=mock_request
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
    assert extracted_tool_calls.content == "I will help you get the weather info.\n\n"


def test_streaming_basic_functionality(glm4_moe_tool_parser, mock_request):
    """Test basic streaming functionality."""
    _reset_streaming_state(glm4_moe_tool_parser)

    current_text = """<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Beijing</arg_value>
</tool_call>"""

    result = glm4_moe_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=current_text,
        delta_text=current_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )

    # Should return tool call with name and arguments in one shot
    assert result is not None
    assert result.tool_calls is not None
    assert len(result.tool_calls) >= 1


def test_streaming_no_tool_calls(glm4_moe_tool_parser, mock_request):
    """Test streaming when there are no tool calls."""
    _reset_streaming_state(glm4_moe_tool_parser)

    current_text = "This is just regular text without any tool calls."

    result = glm4_moe_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=current_text,
        delta_text=current_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )

    # Should return content
    assert result is not None
    assert result.content == current_text


def test_streaming_with_content_before_tool_calls(glm4_moe_tool_parser, mock_request):
    """Test streaming when there's content before tool calls."""
    _reset_streaming_state(glm4_moe_tool_parser)

    current_text = "I will help you get the weather.<tool_call>"

    result = glm4_moe_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=current_text,
        delta_text=current_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )

    # Should return content before the <tool_call> tag
    assert result is not None
    assert result.content == "I will help you get the weather."


def test_extract_tool_calls_special_characters(glm4_moe_tool_parser, mock_request):
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
        model_output, request=mock_request
    )  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 1
    assert extracted_tool_calls.tool_calls[0].function.name == "send_message"

    args = json.loads(extracted_tool_calls.tool_calls[0].function.arguments)
    assert args["recipient"] == "Amy"
    assert args["message"] == "It is a nice day"
    assert args["priority"] == "high"


def test_extract_tool_calls_incomplete_tool_call(glm4_moe_tool_parser, mock_request):
    """Test incomplete tool calls (missing closing tag)."""
    model_output = """<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Beijing</arg_value>
<arg_key>date</arg_key>
<arg_value>2025-08-01</arg_value>"""

    extracted_tool_calls = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=mock_request
    )  # type: ignore[arg-type]

    # Incomplete tool calls should not be extracted
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output


def _reset_streaming_state(parser):
    """Helper to reset parser streaming state."""
    parser.current_tool_name_sent = False
    parser.prev_tool_call_arr = []
    parser.current_tool_id = -1
    parser.streamed_args_for_tool = []
    parser._tool_call_ids = []
    parser._sent_content_idx = 0


def test_streaming_incremental_string_value(glm4_moe_tool_parser, mock_request):
    """Test incremental streaming of string argument values."""
    _reset_streaming_state(glm4_moe_tool_parser)

    # Simulate streaming a tool call chunk by chunk
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
    current_text = ""
    for chunk in chunks:
        current_text += chunk
        result = glm4_moe_tool_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=mock_request,
        )
        if result is not None and result.tool_calls:
            for tc in result.tool_calls:
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


def test_streaming_empty_tool_call(glm4_moe_tool_parser, mock_request):
    """Test that empty tool calls don't cause infinite loops."""
    _reset_streaming_state(glm4_moe_tool_parser)

    current_text = "<tool_call></tool_call>"
    result = glm4_moe_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=current_text,
        delta_text=current_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )

    # Should not hang and should return something (None or content)
    # The key is that this completes without hanging
    assert result is None or hasattr(result, "content") or hasattr(result, "tool_calls")


def test_streaming_prev_tool_call_arr_updates(glm4_moe_tool_parser, mock_request):
    """Test that prev_tool_call_arr is populated incrementally."""
    _reset_streaming_state(glm4_moe_tool_parser)

    chunks = [
        "<tool_call>get_weather\n",
        "<arg_key>city</arg_key>",
        "<arg_value>Beijing</arg_value>",
        "</tool_call>",
    ]

    current_text = ""
    for chunk in chunks:
        current_text += chunk
        glm4_moe_tool_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=mock_request,
        )

    # After the tool call completes, prev_tool_call_arr should be populated
    assert len(glm4_moe_tool_parser.prev_tool_call_arr) == 1
    tool_entry = glm4_moe_tool_parser.prev_tool_call_arr[0]
    assert tool_entry.get("name") == "get_weather"

    # arguments is a JSON string in the re-parse approach
    args_str = tool_entry.get("arguments")
    assert isinstance(args_str, str), f"Expected str, got {type(args_str)}"
    parsed = json.loads(args_str)
    assert parsed["city"] == "Beijing"

    # streamed_args_for_tool should match prev_tool_call_arr arguments
    streamed = glm4_moe_tool_parser.streamed_args_for_tool[0]
    assert streamed == args_str


def test_streaming_multiple_tool_calls_sequential(glm4_moe_tool_parser, mock_request):
    """Test streaming multiple sequential tool calls."""
    _reset_streaming_state(glm4_moe_tool_parser)

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

    current_text = ""
    for chunk in chunks:
        current_text += chunk
        glm4_moe_tool_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=mock_request,
        )

    # Should have two tool calls in prev_tool_call_arr
    assert len(glm4_moe_tool_parser.prev_tool_call_arr) == 2
    args0 = json.loads(glm4_moe_tool_parser.prev_tool_call_arr[0]["arguments"])
    args1 = json.loads(glm4_moe_tool_parser.prev_tool_call_arr[1]["arguments"])
    assert args0["city"] == "Beijing"
    assert args1["city"] == "Shanghai"


def test_streaming_json_escape_in_string(glm4_moe_tool_parser, mock_request):
    """Test that special characters in string values are properly escaped."""
    _reset_streaming_state(glm4_moe_tool_parser)

    chunks = [
        "<tool_call>send_message\n",
        "<arg_key>message</arg_key>",
        '<arg_value>Hello "world"\nNew line</arg_value>',
        "</tool_call>",
    ]

    current_text = ""
    for chunk in chunks:
        current_text += chunk
        glm4_moe_tool_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=mock_request,
        )

    # The streamed_args_for_tool should contain valid JSON
    assert len(glm4_moe_tool_parser.streamed_args_for_tool) == 1
    args_json = glm4_moe_tool_parser.streamed_args_for_tool[0]
    parsed = json.loads(args_json)
    assert "message" in parsed
    assert '"' in parsed["message"] or "world" in parsed["message"]


def test_streaming_long_content_incremental(glm4_moe_tokenizer):
    """Test incremental streaming of long content (Issue #32829).

    This is the core fix: for long string values like code (4000+ chars),
    the parser should stream incrementally rather than buffering until
    complete. This test verifies we get many fragments, not just 1-3.
    """

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

    # Create tools with schema to enable string type detection
    # This is required for incremental streaming of string values
    tools = [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="write_to_file",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            ),
        ),
    ]
    glm4_moe_tool_parser = Glm4MoeModelToolParser(glm4_moe_tokenizer, tools=tools)
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[],
        tools=tools,
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
    current_text = ""
    for chunk in chunks:
        current_text += chunk
        result = glm4_moe_tool_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        if result is not None and result.tool_calls:
            for tc in result.tool_calls:
                func = tc.function
                if isinstance(func, dict):
                    args = func.get("arguments")
                else:
                    args = getattr(func, "arguments", None)
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


def test_extract_tool_calls_numeric_deserialization(glm4_moe_tool_parser, mock_request):
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
        model_output, request=mock_request
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


def test_zero_argument_tool_call(glm4_moe_tool_parser, mock_request):
    """Regression: zero-argument tool call crash (PR #32321)."""
    model_output = """<tool_call>get_time
</tool_call>"""

    extracted = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=mock_request
    )  # type: ignore[arg-type]

    assert extracted.tools_called
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].function.name == "get_time"
    args = json.loads(extracted.tool_calls[0].function.arguments)
    assert args == {}


def test_malformed_tool_call_no_regex_match(glm4_moe_tool_parser, mock_request):
    """Regression: malformed tool_call with no regex match (PR #32321)."""
    model_output = "<tool_call>   </tool_call>"

    extracted = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=mock_request
    )  # type: ignore[arg-type]

    assert extracted.tools_called is False
    assert extracted.tool_calls == []


def test_delimiter_preserved_transformers_5x(glm4_moe_tool_parser):
    """Regression: adjust_request sets skip_special_tokens=False (PR #31622)."""
    # Tools enabled
    request_with_tools = ChatCompletionRequest(
        model=MODEL,
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ],
    )  # type: ignore
    adjusted = glm4_moe_tool_parser.adjust_request(request_with_tools)
    assert adjusted.skip_special_tokens is False

    # tool_choice="none"
    request_no_choice = ChatCompletionRequest(
        model=MODEL,
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ],
        tool_choice="none",
    )  # type: ignore
    adjusted_none = glm4_moe_tool_parser.adjust_request(request_no_choice)
    assert adjusted_none.skip_special_tokens is True

    # No tools at all
    request_no_tools = ChatCompletionRequest(
        model=MODEL,
        messages=[],
    )  # type: ignore
    adjusted_empty = glm4_moe_tool_parser.adjust_request(request_no_tools)
    assert adjusted_empty.skip_special_tokens is True


def test_unicode_characters_preserved(glm4_moe_tool_parser, mock_request):
    """Regression: Unicode chars must not be escaped to \\uXXXX (PR #30920)."""
    model_output = """<tool_call>send_message
<arg_key>greeting</arg_key>
<arg_value>你好世界</arg_value>
<arg_key>emoji</arg_key>
<arg_value>🎉</arg_value>
</tool_call>"""

    extracted = glm4_moe_tool_parser.extract_tool_calls(
        model_output, request=mock_request
    )  # type: ignore[arg-type]

    assert extracted.tools_called
    assert len(extracted.tool_calls) == 1

    raw_args = extracted.tool_calls[0].function.arguments
    assert "你好世界" in raw_args
    assert "🎉" in raw_args
    assert "\\u4f60" not in raw_args
    parsed_args = json.loads(raw_args)
    assert parsed_args["greeting"] == "你好世界"
    assert parsed_args["emoji"] == "🎉"


def test_streaming_multi_token_chunks(glm4_moe_tool_parser, mock_request):
    """Test that multi-token chunks (stream_interval > 1) are handled correctly.

    With stream_interval > 1 or MTP, multiple XML tags arrive in one delta.
    The old buffer-based parser could only return one delta per call, losing
    data on the final output. The re-parse approach handles this correctly.
    """
    _reset_streaming_state(glm4_moe_tool_parser)

    # Simulate stream_interval=3: chunks contain multiple XML tags
    chunks = [
        "<tool_call>get_weather\n<arg_key>city</arg_key><arg_value>Bei",
        "jing</arg_value>",
        "</tool_call>",
    ]

    current_text = ""
    for chunk in chunks:
        current_text += chunk
        glm4_moe_tool_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=mock_request,
        )

    # All data should be captured despite multi-token chunks
    assert len(glm4_moe_tool_parser.prev_tool_call_arr) == 1
    args = json.loads(glm4_moe_tool_parser.streamed_args_for_tool[0])
    assert args["city"] == "Beijing"


def test_streaming_entire_tool_call_at_once(glm4_moe_tool_parser, mock_request):
    """Test that a complete tool call arriving in one delta works.

    This simulates the extreme MTP case where all tokens arrive at once.
    """
    _reset_streaming_state(glm4_moe_tool_parser)

    full_text = (
        "<tool_call>get_weather\n"
        "<arg_key>city</arg_key>"
        "<arg_value>Beijing</arg_value>"
        "</tool_call>"
    )

    result = glm4_moe_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=full_text,
        delta_text=full_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )

    # Should emit tool call with complete arguments in one shot
    assert result is not None
    assert result.tool_calls is not None

    # Verify final state
    assert len(glm4_moe_tool_parser.prev_tool_call_arr) == 1
    args = json.loads(glm4_moe_tool_parser.streamed_args_for_tool[0])
    assert args["city"] == "Beijing"


def test_streaming_content_between_tool_calls_multi_token(
    glm4_moe_tool_parser, mock_request
):
    """Test content between tool calls with multi-token chunks."""
    _reset_streaming_state(glm4_moe_tool_parser)

    # Deliver everything at once — worst case for the old buffer parser
    full_text = (
        "I will check.\n"
        "<tool_call>get_weather\n"
        "<arg_key>city</arg_key>"
        "<arg_value>Beijing</arg_value>"
        "</tool_call>"
        "\nAlso Shanghai.\n"
        "<tool_call>get_weather\n"
        "<arg_key>city</arg_key>"
        "<arg_value>Shanghai</arg_value>"
        "</tool_call>"
    )

    # First call with partial text (content only)
    partial = "I will check.\n"
    result1 = glm4_moe_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=partial,
        delta_text=partial,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )
    assert result1 is not None
    assert result1.content == "I will check.\n"

    # Second call with everything
    glm4_moe_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=full_text,
        delta_text=full_text[len(partial) :],
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )

    # Should have both tool calls
    assert len(glm4_moe_tool_parser.prev_tool_call_arr) == 2
    args0 = json.loads(glm4_moe_tool_parser.prev_tool_call_arr[0]["arguments"])
    args1 = json.loads(glm4_moe_tool_parser.prev_tool_call_arr[1]["arguments"])
    assert args0["city"] == "Beijing"
    assert args1["city"] == "Shanghai"


def test_streaming_multi_token_with_multiple_args(glm4_moe_tokenizer):
    """Test multi-token streaming with multiple arguments of mixed types."""
    tools = [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="calculate",
                parameters={
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string"},
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                },
            ),
        ),
    ]
    parser = Glm4MoeModelToolParser(glm4_moe_tokenizer, tools=tools)
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[],
        tools=tools,
    )

    # All arguments arrive in two big chunks (simulates stream_interval=5)
    chunks = [
        "<tool_call>calculate\n<arg_key>operation</arg_key><arg_value>add</arg_value><arg_key>a</arg_key>",
        "<arg_value>42</arg_value><arg_key>b</arg_key><arg_value>3.14</arg_value></tool_call>",
    ]

    current_text = ""
    for chunk in chunks:
        current_text += chunk
        parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )

    args = json.loads(parser.streamed_args_for_tool[0])
    assert args["operation"] == "add"
    assert args["a"] == 42
    assert args["b"] == 3.14


def _simulate_streaming(tokenizer, parser, request, text, stream_interval=1):
    """Simulate streaming with a given stream_interval.

    Tokens are batched into chunks of ``stream_interval`` tokens,
    mimicking how the output processor delivers them.
    Returns a list of non-None DeltaMessages.
    """
    tokens = tokenizer.encode(text)
    previous_text = ""
    deltas = []
    for i in range(0, len(tokens), stream_interval):
        chunk_ids = tokens[i : i + stream_interval]
        delta_text = tokenizer.decode(chunk_ids)
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=chunk_ids,
            request=request,
        )
        previous_text = current_text
        if delta is not None:
            deltas.append(delta)
    return deltas


def _collect_from_deltas(deltas):
    """Reconstruct tool call names/args and content from a delta stream."""
    tools: dict[int, dict] = {}
    content_parts: list[str] = []
    for d in deltas:
        if d.content:
            content_parts.append(d.content)
        if d.tool_calls:
            for tc in d.tool_calls:
                func = tc.function
                if isinstance(func, dict):
                    name = func.get("name")
                    args = func.get("arguments")
                else:
                    name = getattr(func, "name", None)
                    args = getattr(func, "arguments", None)
                idx = tc.index
                if idx not in tools:
                    tools[idx] = {"name": None, "args_fragments": []}
                if name:
                    tools[idx]["name"] = name
                if args:
                    tools[idx]["args_fragments"].append(args)
    return content_parts, tools


@pytest.mark.parametrize("stream_interval", [1, 2, 3, 5, 8])
def test_stream_interval_single_tool_call(glm4_moe_tokenizer, stream_interval):
    """Tool call streaming produces correct name + args at any interval."""
    tools = [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            ),
        ),
    ]
    parser = Glm4MoeModelToolParser(glm4_moe_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    text = (
        "<tool_call>get_weather\n"
        "<arg_key>city</arg_key>"
        "<arg_value>Beijing</arg_value>"
        "</tool_call>"
    )

    deltas = _simulate_streaming(
        glm4_moe_tokenizer, parser, request, text, stream_interval
    )
    _, tools_found = _collect_from_deltas(deltas)

    assert 0 in tools_found
    assert tools_found[0]["name"] == "get_weather"
    args_json = "".join(tools_found[0]["args_fragments"])
    parsed = json.loads(args_json)
    assert parsed == {"city": "Beijing"}


@pytest.mark.parametrize("stream_interval", [1, 2, 3, 5, 8])
def test_stream_interval_multiple_tool_calls(glm4_moe_tokenizer, stream_interval):
    """Multiple sequential tool calls with correct indices at any interval."""
    tools = [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            ),
        ),
    ]
    parser = Glm4MoeModelToolParser(glm4_moe_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    text = (
        "<tool_call>get_weather\n"
        "<arg_key>city</arg_key>"
        "<arg_value>Beijing</arg_value>"
        "</tool_call>"
        "<tool_call>get_weather\n"
        "<arg_key>city</arg_key>"
        "<arg_value>Shanghai</arg_value>"
        "</tool_call>"
    )

    deltas = _simulate_streaming(
        glm4_moe_tokenizer, parser, request, text, stream_interval
    )
    _, tools_found = _collect_from_deltas(deltas)

    assert 0 in tools_found and 1 in tools_found
    args0 = json.loads("".join(tools_found[0]["args_fragments"]))
    args1 = json.loads("".join(tools_found[1]["args_fragments"]))
    assert args0 == {"city": "Beijing"}
    assert args1 == {"city": "Shanghai"}


@pytest.mark.parametrize("stream_interval", [1, 2, 3, 5, 8])
def test_stream_interval_content_then_tool_call(glm4_moe_tokenizer, stream_interval):
    """Content before a tool call is fully emitted before tool deltas."""
    tools = [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            ),
        ),
    ]
    parser = Glm4MoeModelToolParser(glm4_moe_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    text = (
        "I will check the weather for you.\n"
        "<tool_call>get_weather\n"
        "<arg_key>city</arg_key>"
        "<arg_value>Beijing</arg_value>"
        "</tool_call>"
    )

    deltas = _simulate_streaming(
        glm4_moe_tokenizer, parser, request, text, stream_interval
    )
    content_parts, tools_found = _collect_from_deltas(deltas)

    # Content must be present and precede tool calls
    full_content = "".join(content_parts)
    assert "I will check the weather" in full_content

    # Tool call must be correct
    assert 0 in tools_found
    assert tools_found[0]["name"] == "get_weather"
    args = json.loads("".join(tools_found[0]["args_fragments"]))
    assert args == {"city": "Beijing"}


def test_stream_interval_extreme_single_chunk(glm4_moe_tokenizer):
    """Extreme MTP: entire output arrives in one chunk (interval=9999)."""
    tools = [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            ),
        ),
    ]
    parser = Glm4MoeModelToolParser(glm4_moe_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    text = (
        "Here is the weather.\n"
        "<tool_call>get_weather\n"
        "<arg_key>city</arg_key>"
        "<arg_value>Beijing</arg_value>"
        "</tool_call>"
    )

    deltas = _simulate_streaming(
        glm4_moe_tokenizer, parser, request, text, stream_interval=9999
    )
    content_parts, tools_found = _collect_from_deltas(deltas)

    assert "Here is the weather" in "".join(content_parts)
    assert 0 in tools_found
    assert tools_found[0]["name"] == "get_weather"
    args = json.loads("".join(tools_found[0]["args_fragments"]))
    assert args == {"city": "Beijing"}


@pytest.mark.parametrize("stream_interval", [1, 2, 5])
def test_stream_interval_content_between_tool_calls(
    glm4_moe_tokenizer, stream_interval
):
    """Content between tool calls must be emitted, not silently dropped."""
    tools = [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            ),
        ),
    ]
    parser = Glm4MoeModelToolParser(glm4_moe_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    text = (
        "Checking Beijing.\n"
        "<tool_call>get_weather\n"
        "<arg_key>city</arg_key>"
        "<arg_value>Beijing</arg_value>"
        "</tool_call>"
        "\nAlso Shanghai.\n"
        "<tool_call>get_weather\n"
        "<arg_key>city</arg_key>"
        "<arg_value>Shanghai</arg_value>"
        "</tool_call>"
    )

    deltas = _simulate_streaming(
        glm4_moe_tokenizer, parser, request, text, stream_interval
    )
    content_parts, tools_found = _collect_from_deltas(deltas)

    full_content = "".join(content_parts)
    # Both prefix and inter-tool-call content must appear
    assert "Checking Beijing" in full_content
    assert "Also Shanghai" in full_content

    # Both tool calls must be correct
    assert 0 in tools_found and 1 in tools_found
    args0 = json.loads("".join(tools_found[0]["args_fragments"]))
    args1 = json.loads("".join(tools_found[1]["args_fragments"]))
    assert args0 == {"city": "Beijing"}
    assert args1 == {"city": "Shanghai"}


# --- Regression tests for streaming argument corruption -------------------
# Prior to this fix, Optional[str] schemas (rendered as anyOf / list-type by
# Pydantic) caused `_is_string_type` to return False.  The streaming partial
# path would then emit the bare value (no quotes), and the completed-pair
# path would emit `json.dumps(...)` (with quotes), causing the length-based
# diff to produce garbage such as `{"sender": Smithh"}` — missing opening
# quote, last char duplicated.  Non-string values (arrays, etc.) suffered
# the same class of bug when `json.dumps` normalized whitespace.


@pytest.mark.parametrize(
    "schema",
    [
        {"anyOf": [{"type": "string"}, {"type": "null"}]},
        {"oneOf": [{"type": "string"}, {"type": "null"}]},
        {"type": ["string", "null"]},
    ],
    ids=["anyOf_nullable_string", "oneOf_nullable_string", "type_list_string"],
)
@pytest.mark.parametrize("stream_interval", [1, 2, 3, 5, 8])
def test_streaming_optional_string_not_corrupted(
    glm4_moe_tokenizer, schema, stream_interval
):
    """Optional[str] fields must be detected as string types at any
    stream_interval / real-tokenizer boundary, and streamed as properly
    quoted JSON (no missing opening quote, no duplicated last char)."""
    tools = [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="search_email",
                parameters={
                    "type": "object",
                    "properties": {"sender": schema},
                },
            ),
        ),
    ]
    parser = Glm4MoeModelToolParser(glm4_moe_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    text = (
        "<tool_call>search_email\n"
        "<arg_key>sender</arg_key>"
        "<arg_value>Smith</arg_value>"
        "</tool_call>"
    )

    deltas = _simulate_streaming(
        glm4_moe_tokenizer, parser, request, text, stream_interval
    )
    _, tools_found = _collect_from_deltas(deltas)

    assert 0 in tools_found
    args = json.loads("".join(tools_found[0]["args_fragments"]))
    assert args == {"sender": "Smith"}


@pytest.mark.parametrize("stream_interval", [1, 2, 3, 5, 8])
def test_streaming_array_arg_preserves_raw_json(glm4_moe_tokenizer, stream_interval):
    """Non-string values (arrays/objects) must stream through without the
    whitespace-divergence corruption (``["Acme","Beta"]]`` pattern) that
    used to result from json.dumps normalizing the partial path's raw
    output (`[a,b]`) into a differently-spaced complete-path render
    (`[a, b]`).  Verified across real tokenizer boundaries."""
    tools = [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="search_email",
                parameters={
                    "type": "object",
                    "properties": {"companies": {"type": "array"}},
                },
            ),
        ),
    ]
    parser = Glm4MoeModelToolParser(glm4_moe_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    text = (
        "<tool_call>search_email\n"
        "<arg_key>companies</arg_key>"
        '<arg_value>["Acme","Beta"]</arg_value>'
        "</tool_call>"
    )

    deltas = _simulate_streaming(
        glm4_moe_tokenizer, parser, request, text, stream_interval
    )
    _, tools_found = _collect_from_deltas(deltas)

    assert 0 in tools_found
    args = json.loads("".join(tools_found[0]["args_fragments"]))
    assert args == {"companies": ["Acme", "Beta"]}


@pytest.mark.parametrize("stream_interval", [1, 3, 8])
def test_streaming_large_object_arg_streams_incrementally(
    glm4_moe_tokenizer, stream_interval
):
    """A non-string (object) arg must stream *incrementally* — the client
    should receive at least one args delta before the closing tag, not be
    held until </arg_value>.  This guards against a throughput regression
    where non-string partials would otherwise be suppressed."""
    tools = [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="do_thing",
                parameters={
                    "type": "object",
                    "properties": {"payload": {"type": "object"}},
                },
            ),
        ),
    ]
    parser = Glm4MoeModelToolParser(glm4_moe_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    # Use a body long enough to span multiple tokenizer boundaries.
    body = '{"a":1,"b":[1,2,3,4,5],"c":"hello world","d":true,"e":null}'
    text = (
        "<tool_call>do_thing\n"
        "<arg_key>payload</arg_key>"
        f"<arg_value>{body}</arg_value>"
        "</tool_call>"
    )

    deltas = _simulate_streaming(
        glm4_moe_tokenizer, parser, request, text, stream_interval
    )
    _, tools_found = _collect_from_deltas(deltas)

    assert 0 in tools_found
    fragments = tools_found[0]["args_fragments"]
    # Final rendering must parse.  The arguments object wraps the body
    # under the declared "payload" key.
    args = json.loads("".join(fragments))
    assert args == {"payload": json.loads(body)}

    # At least one fragment must arrive before the trailing `}` that
    # closes the outer arguments object — otherwise the client is held
    # hostage to the </arg_value> tag for non-string args.  (stream_interval=8
    # may legitimately batch everything together if the whole tool call
    # fits in fewer than 8 tokens, so only assert for interval=1.)
    if stream_interval == 1:
        combined = "".join(fragments)
        assert combined.endswith("}")
        assert len(fragments) >= 2, (
            f"expected non-string arg to stream incrementally, got fragments="
            f"{fragments!r}"
        )


def test_extract_tool_calls_optional_string_preserves_literal(
    glm4_moe_tool_parser,
):
    """Non-streaming behavior-change test for ``extract_tool_calls``:
    Optional[str] fields now preserve the model's emitted text as a
    string (e.g. "true"), rather than coercing it via ``_deserialize``
    into a bool/int.  Pre-fix, ``_is_string_type`` returned False for
    Optional[str], sending the value through ``ast.literal_eval`` /
    ``json.loads`` and silently converting ``"true"`` → ``True``.  The
    new behavior matches the declared schema type (string)."""
    tools = [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="note_flag",
                parameters={
                    "type": "object",
                    "properties": {
                        "label": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "null"},
                            ]
                        },
                    },
                },
            ),
        ),
    ]
    parser = Glm4MoeModelToolParser(glm4_moe_tool_parser.model_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    model_output = (
        "<tool_call>note_flag\n"
        "<arg_key>label</arg_key>"
        "<arg_value>true</arg_value>"
        "</tool_call>"
    )
    result = parser.extract_tool_calls(model_output, request=request)
    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args == {"label": "true"}, (
        "Optional[str] value must remain a string, not be coerced to bool"
    )


@pytest.mark.parametrize("stream_interval", [1, 2, 3, 5, 8])
def test_streaming_undeclared_arg_defaults_to_string(
    glm4_moe_tokenizer, stream_interval
):
    """A tool whose schema declares some args but not others must still
    stream the undeclared args correctly.  Reproduces the production
    divergence-guard warning seen when GLM-5.1 emits text fields
    (e.g. edit-tool ``new_text``) that the tool schema did not list.

    Pre-fix: ``_is_string_type`` returns False for the undeclared arg,
    the non-string partial path emits bare text, the complete path
    wraps with ``json.dumps``, the prefix-divergence guard fires, and
    the client never receives the trailing ``"`` / ``}`` — leading to
    ``json.loads`` failure on the concatenated deltas.

    Post-fix: undeclared args default to string-type (matching
    minimax_m2 / qwen3xml parsers), partial and complete renderings
    stay prefix-consistent, client gets valid JSON.
    """
    tools = [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="edit_memory",
                parameters={
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "target": {"type": "string"},
                        # new_text intentionally undeclared
                    },
                },
            ),
        ),
    ]
    parser = Glm4MoeModelToolParser(glm4_moe_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    text = (
        "<tool_call>edit_memory\n"
        "<arg_key>action</arg_key><arg_value>replace</arg_value>"
        "<arg_key>target</arg_key><arg_value>memory</arg_value>"
        "<arg_key>new_text</arg_key>"
        "<arg_value>A long body of replacement text.</arg_value>"
        "</tool_call>"
    )

    deltas = _simulate_streaming(
        glm4_moe_tokenizer, parser, request, text, stream_interval
    )
    _, tools_found = _collect_from_deltas(deltas)
    assert 0 in tools_found
    args = json.loads("".join(tools_found[0]["args_fragments"]))
    assert args == {
        "action": "replace",
        "target": "memory",
        "new_text": "A long body of replacement text.",
    }
