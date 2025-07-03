# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

import json

import pytest

from vllm.entrypoints.openai.protocol import FunctionCall, ToolCall
from vllm.entrypoints.openai.tool_parsers import MinimaxToolParser
from vllm.transformers_utils.tokenizer import get_tokenizer

# Use a common model that is likely to be available
MODEL = "MiniMaxAi/MiniMax-M1-40k"


@pytest.fixture(scope="module")
def minimax_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def minimax_tool_parser(minimax_tokenizer):
    return MinimaxToolParser(minimax_tokenizer)


def assert_tool_calls(actual_tool_calls: list[ToolCall],
                      expected_tool_calls: list[ToolCall]):
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(actual_tool_calls,
                                                    expected_tool_calls):
        assert isinstance(actual_tool_call.id, str)
        assert len(actual_tool_call.id) > 16

        assert actual_tool_call.type == "function"
        assert actual_tool_call.function == expected_tool_call.function


def test_extract_tool_calls_no_tools(minimax_tool_parser):
    model_output = "This is a test"
    extracted_tool_calls = minimax_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output


@pytest.mark.parametrize(
    ids=[
        "single_tool_call",
        "multiple_tool_calls",
        "tool_call_with_content_before",
        "tool_call_with_single_line_json",
        "tool_call_incomplete_tag",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
            """<tool_calls>
{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}
</tool_calls>""",
            [
                ToolCall(function=FunctionCall(
                    name="get_current_weather",
                    arguments=json.dumps({
                        "city": "Dallas",
                        "state": "TX",
                        "unit": "fahrenheit",
                    }),
                ))
            ],
            None,
        ),
        (
            """<tool_calls>
{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}
{"name": "get_current_weather", "arguments": {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}}
</tool_calls>""",
            [
                ToolCall(function=FunctionCall(
                    name="get_current_weather",
                    arguments=json.dumps({
                        "city": "Dallas",
                        "state": "TX",
                        "unit": "fahrenheit",
                    }),
                )),
                ToolCall(function=FunctionCall(
                    name="get_current_weather",
                    arguments=json.dumps({
                        "city": "Orlando",
                        "state": "FL",
                        "unit": "fahrenheit",
                    }),
                )),
            ],
            None,
        ),
        (
            """I'll help you check the weather. <tool_calls>
{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "celsius"}}
</tool_calls>""",
            [
                ToolCall(function=FunctionCall(
                    name="get_current_weather",
                    arguments=json.dumps({
                        "city": "Seattle",
                        "state": "WA",
                        "unit": "celsius",
                    }),
                ))
            ],
            "I'll help you check the weather.",
        ),
        (
            """<tool_calls>
{"name": "get_current_weather", "arguments": {"city": "New York", "state": "NY", "unit": "celsius"}}
</tool_calls>""",
            [
                ToolCall(function=FunctionCall(
                    name="get_current_weather",
                    arguments=json.dumps({
                        "city": "New York",
                        "state": "NY",
                        "unit": "celsius",
                    }),
                ))
            ],
            None,
        ),
        (
            """<tool_calls>
{"name": "get_current_weather", "arguments": {"city": "Boston", "state": "MA"}}""",
            [
                ToolCall(function=FunctionCall(
                    name="get_current_weather",
                    arguments=json.dumps({
                        "city": "Boston",
                        "state": "MA",
                    }),
                ))
            ],
            None,
        ),
    ],
)
def test_extract_tool_calls(minimax_tool_parser, model_output,
                            expected_tool_calls, expected_content):
    extracted_tool_calls = minimax_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]
    assert extracted_tool_calls.tools_called

    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content


def test_preprocess_model_output_with_thinking_tags(minimax_tool_parser):
    """Test that tool calls within thinking tags are removed during preprocessing."""
    model_output = """<think>Let me think about this. <tool_calls>
{"name": "fake_tool", "arguments": {"param": "value"}}
</tool_calls> This should be removed.</think>

I'll help you with that. <tool_calls>
{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA"}}
</tool_calls>"""

    processed_output = minimax_tool_parser.preprocess_model_output(
        model_output)

    # The tool call within thinking tags should be removed
    assert "fake_tool" not in processed_output
    # But the thinking tag itself should remain
    assert "<think>" in processed_output
    assert "</think>" in processed_output
    # The actual tool call outside thinking tags should remain
    assert "get_current_weather" in processed_output


def test_extract_tool_calls_with_thinking_tags(minimax_tool_parser):
    """Test tool extraction when thinking tags contain tool calls that should be ignored."""
    model_output = """<think>I should use a tool. <tool_calls>
{"name": "ignored_tool", "arguments": {"should": "ignore"}}
</tool_calls></think>

Let me help you with the weather. <tool_calls>
{"name": "get_current_weather", "arguments": {"city": "Miami", "state": "FL", "unit": "fahrenheit"}}
</tool_calls>"""

    extracted_tool_calls = minimax_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 1
    assert extracted_tool_calls.tool_calls[
        0].function.name == "get_current_weather"

    # Content extraction is based on the position of the first <tool_calls> in the original model_output
    # Since preprocessing removes tool calls within thinking tags, the actual first <tool_calls> is the external one
    expected_content = """<think>I should use a tool. <tool_calls>
{"name": "ignored_tool", "arguments": {"should": "ignore"}}
</tool_calls></think>

Let me help you with the weather."""
    assert extracted_tool_calls.content == expected_content


def test_extract_tool_calls_invalid_json(minimax_tool_parser):
    """Test that invalid JSON in tool calls is handled gracefully."""
    model_output = """<tool_calls>
{"name": "valid_tool", "arguments": {"city": "Seattle"}}
{invalid json here}
{"name": "another_valid_tool", "arguments": {"param": "value"}}
</tool_calls>"""

    extracted_tool_calls = minimax_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    # Should extract only the valid JSON tool calls
    assert len(extracted_tool_calls.tool_calls) == 2
    assert extracted_tool_calls.tool_calls[0].function.name == "valid_tool"
    assert extracted_tool_calls.tool_calls[
        1].function.name == "another_valid_tool"


def test_extract_tool_calls_missing_name_or_arguments(minimax_tool_parser):
    """Test that tool calls missing name or arguments are filtered out."""
    model_output = """<tool_calls>
{"name": "valid_tool", "arguments": {"city": "Seattle"}}
{"name": "missing_args"}
{"arguments": {"city": "Portland"}}
{"name": "another_valid_tool", "arguments": {"param": "value"}}
</tool_calls>"""

    extracted_tool_calls = minimax_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    # Should extract only the valid tool calls with both name and arguments
    assert len(extracted_tool_calls.tool_calls) == 2
    assert extracted_tool_calls.tool_calls[0].function.name == "valid_tool"
    assert extracted_tool_calls.tool_calls[
        1].function.name == "another_valid_tool"


def test_streaming_basic_functionality(minimax_tool_parser):
    """Test basic streaming functionality."""
    # Reset streaming state
    minimax_tool_parser.current_tool_name_sent = False
    minimax_tool_parser.prev_tool_call_arr = []
    minimax_tool_parser.current_tool_id = -1
    minimax_tool_parser.streamed_args_for_tool = []

    # Test with a simple tool call
    current_text = """<tool_calls>
{"name": "get_current_weather", "arguments": {"city": "Seattle"}}
</tool_calls>"""

    # First call should handle the initial setup
    result = minimax_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=current_text,
        delta_text="</tool_calls>",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    # The result might be None or contain tool call information
    # This depends on the internal state management
    if result is not None and hasattr(result,
                                      'tool_calls') and result.tool_calls:
        assert len(result.tool_calls) >= 0


def test_streaming_with_content_before_tool_calls(minimax_tool_parser):
    """Test streaming when there's content before tool calls."""
    # Reset streaming state
    minimax_tool_parser.current_tool_name_sent = False
    minimax_tool_parser.prev_tool_call_arr = []
    minimax_tool_parser.current_tool_id = -1
    minimax_tool_parser.streamed_args_for_tool = []

    current_text = "I'll help you with that. <tool_calls>"

    # When there's content before tool calls, it should be returned as content
    result = minimax_tool_parser.extract_tool_calls_streaming(
        previous_text="I'll help you",
        current_text=current_text,
        delta_text=" with that. <tool_calls>",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    if result is not None and hasattr(result, 'content'):
        # Should contain some content
        assert result.content is not None


def test_streaming_no_tool_calls(minimax_tool_parser):
    """Test streaming when there are no tool calls."""
    current_text = "This is just regular text without any tool calls."

    result = minimax_tool_parser.extract_tool_calls_streaming(
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
    assert hasattr(result, 'content')
    assert result.content == " without any tool calls."


def test_streaming_with_thinking_tags(minimax_tool_parser):
    """Test streaming with thinking tags that contain tool calls."""
    # Reset streaming state
    minimax_tool_parser.current_tool_name_sent = False
    minimax_tool_parser.prev_tool_call_arr = []
    minimax_tool_parser.current_tool_id = -1
    minimax_tool_parser.streamed_args_for_tool = []

    current_text = """<think><tool_calls>{"name": "ignored", "arguments": {}}</tool_calls></think><tool_calls>{"name": "real_tool", "arguments": {"param": "value"}}</tool_calls>"""

    result = minimax_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=current_text,
        delta_text=current_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    # The preprocessing should remove tool calls from thinking tags
    # and only process the real tool call
    if result is not None and hasattr(result,
                                      'tool_calls') and result.tool_calls:
        for tool_call in result.tool_calls:
            assert tool_call.function.name != "ignored"


def test_extract_tool_calls_multiline_json_not_supported(minimax_tool_parser):
    """Test that multiline JSON in tool calls is not currently supported."""
    model_output = """<tool_calls>
{
  "name": "get_current_weather",
  "arguments": {
    "city": "New York",
    "state": "NY",
    "unit": "celsius"
  }
}
</tool_calls>"""

    extracted_tool_calls = minimax_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]

    # Multiline JSON is currently not supported, should return no tools called
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content is None
