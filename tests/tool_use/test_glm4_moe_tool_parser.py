# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import json

import pytest

from vllm.entrypoints.openai.protocol import FunctionCall, ToolCall
from vllm.entrypoints.openai.tool_parsers.glm4_moe_tool_parser import (
    Glm4MoeModelToolParser,
)
from vllm.transformers_utils.tokenizer import get_tokenizer

pytestmark = pytest.mark.cpu_test

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
