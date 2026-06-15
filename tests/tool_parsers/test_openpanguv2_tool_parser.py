# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for OpenPanguV2ToolParser."""

import json
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.openpangu_v2_tool_parser import OpenPanguV2ToolParser

TC_START = "<|tool_call_start|>"
TC_END = "<|tool_call_end|>"


@pytest.fixture(scope="module")
def parser():
    """Create a OpenPanguV2ToolParser instance."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_vocab.return_value = {
        TC_START: 104,
        TC_END: 105,
    }
    mock_tokenizer.tokenizer = mock_tokenizer
    parser = OpenPanguV2ToolParser(mock_tokenizer)
    return parser


@pytest.fixture(scope="module")
def req():
    """Create a mock ChatCompletionRequest."""
    req = MagicMock(spec=ChatCompletionRequest)
    return req


def build_tool_call(tool_calls: list[dict]) -> str:
    """
    Build Pangu tool call string from list of tool dicts.

    Args:
        tool_calls: List of tool call dictionaries with 'name' and 'arguments' keys

    Returns:
        Formatted string with tool call tokens enclosing JSON array
    """
    json_str = json.dumps(tool_calls, ensure_ascii=False)
    return f"{TC_START}{json_str}{TC_END}"


def test_parser_registered():
    """Test that OpenPanguV2ToolParser is registered in ToolParserManager."""
    # This test may need adjustment based on actual registration
    # Uncomment and modify when parser is properly registered
    assert ToolParserManager.get_tool_parser("openpangu_v2") is OpenPanguV2ToolParser


# Non-streaming extraction tests


def test_extract_tool_calls_no_tool_call(parser, req):
    """Test extraction when no tool call tokens are present."""
    model_output = "Hello, I am an AI assistant."

    result = parser.extract_tool_calls(model_output, req)

    assert not result.tools_called
    assert result.content == model_output
    assert result.tool_calls == []


def test_extract_tool_calls_single_tool(parser, req):
    """Test extraction of a single tool call."""
    tool_calls = [{"name": "get_weather", "arguments": {"city": "Beijing"}}]
    model_output = "Thought: I need to check weather." + build_tool_call(tool_calls)

    result = parser.extract_tool_calls(model_output, req)

    assert result.tools_called
    assert result.content == "Thought: I need to check weather."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "get_weather"
    assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Beijing"}


def test_extract_tool_calls_multiple_tools(parser, req):
    """Test extraction of multiple tool calls in one response."""
    tools = [
        {"name": "get_weather", "arguments": {"city": "Beijing"}},
        {"name": "search", "arguments": {"query": "weather forecast"}},
    ]
    model_output = "Let me help you." + build_tool_call(tools)

    result = parser.extract_tool_calls(model_output, req)

    assert result.tools_called
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].function.name == "get_weather"
    assert result.tool_calls[1].function.name == "search"
    assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Beijing"}
    assert json.loads(result.tool_calls[1].function.arguments) == {
        "query": "weather forecast"
    }


def test_extract_tool_calls_invalid_single_object(parser, req):
    """Test that single object (not array) is rejected."""
    # Invalid: single object instead of array
    model_output = (
        "Thought: I need to check weather."
        '<|tool_call_start|>{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        "<|tool_call_end|>"
    )

    result = parser.extract_tool_calls(model_output, req)

    assert not result.tools_called
    assert result.content == model_output
    assert result.tool_calls == []


def test_extract_tool_calls_malformed_json(parser, req):
    """Test handling of malformed JSON in tool call."""
    # Missing closing brace
    model_output = (
        '<|tool_call_start|>[{"name": "send_email", "arguments": {"user": "ALAN"}}'
        "<|tool_call_end|>"
    )

    result = parser.extract_tool_calls(model_output, req)

    assert not result.tools_called
    assert result.tool_calls == []
    assert result.content == model_output


def test_extract_tool_calls_json_parse_error(parser, req):
    """Test handling of JSON syntax errors."""
    # Invalid JSON syntax (double colon)
    model_output = '<|tool_call_start|>[{"name":: "error"}]<|tool_call_end|>'

    result = parser.extract_tool_calls(model_output, req)

    assert not result.tools_called
    assert result.content == model_output


def test_extract_tool_calls_only_start_token(parser, req):
    """Test when only start token is present without end token."""
    model_output = '<|tool_call_start|>[{"name": "incomplete"}'

    result = parser.extract_tool_calls(model_output, req)

    # Should treat as no tool call since end token is missing
    assert not result.tools_called
    assert result.content == model_output


# Streaming extraction tests


def test_extract_tool_calls_streaming_with_basic_returns(parser, req):
    """Test basic return cases in streaming mode."""

    # Case 1: Only tool call end token ID - should return None
    result = parser.extract_tool_calls_streaming("", "", "", [], [], [105], req)
    assert result is None

    # Case 2: No tool call start token in text - should return content
    result = parser.extract_tool_calls_streaming("", "Hello", "Hello", [], [], [1], req)
    assert result.content == "Hello"

    # Case 3: Only tool call start token ID - should return None
    result = parser.extract_tool_calls_streaming(
        "", "<|tool_call_start|>", "<|tool_call_start|>", [], [], [104], req
    )
    assert result is None


def test_extract_tool_calls_streaming_with_tool_arguments(parser, req):
    """est streaming tool call arguments with pre-sent tool name."""
    parser.current_tool_id = 0
    parser.current_tool_name_sent = True
    parser.streamed_args_for_tool = [""]
    parser.prev_tool_call_arr = [{"name": "get_weather"}]

    curr = (
        '<|tool_call_start|>[{"name": "get_weather", "arguments": {"city": "Beijing"}}]'
    )
    res = parser.extract_tool_calls_streaming(
        "", curr, '{"city": "Beijing"}}', [], [], [], req
    )

    assert "Beijing" in res.tool_calls[0].function.arguments
    assert "Beijing" in parser.streamed_args_for_tool[0]


def test_extract_tool_calls_streaming_with_text_before_start_token(parser, req):
    """Test streaming with text before tool call start token."""
    delta = "Thought: I should use a tool.<|tool_call_start|>"

    result = parser.extract_tool_calls_streaming(
        "", delta, delta, [], [], [1, 104], req
    )

    assert result.content == "Thought: I should use a tool."


def test_streaming_with_end_token(parser, req):
    """Test streaming behavior when end token appears."""

    # Text after end token
    curr = "<|tool_call_start|>[...]<|tool_call_end|> End text"
    delta = " End text"

    result = parser.extract_tool_calls_streaming("", curr, delta, [], [], [], req)

    assert result is None


def test_extract_tool_calls_streaming_whit_exception(parser, req):
    """est streaming extraction with incomplete JSON input."""
    curr = '<|tool_call_start|>[{"name": "get_weather"'
    res = parser.extract_tool_calls_streaming(
        "<|tool_call_start|>",
        curr,
        '[{"name": "get_weather"',
        [104],
        [104, 2],
        [2],
        req,
    )

    assert res is None


def test_extract_tool_calls_streaming_with_new_tool_registration(parser, req):
    """Test streaming tool registration:
    state transition from -1 to 0, then name sending."""
    # Initialize parser state
    parser.current_tool_id = -1
    parser.current_tool_name_sent = False
    parser.streamed_args_for_tool = []

    # Construct a fragment parseable by partial_json_parser
    curr = '<|tool_call_start|>[{"name": "get_weather"}'

    # First call: current_tool_id change from -1 to 0
    parser.extract_tool_calls_streaming(
        previous_text="<|tool_call_start|>",
        current_text=curr,
        delta_text='[{"name": "get_weather"}',
        previous_token_ids=[104],
        current_token_ids=[104, 1],
        delta_token_ids=[1],
        request=req,
    )

    assert parser.current_tool_id == 0

    # Second call: simulate state update and send tool name
    res_name = parser.extract_tool_calls_streaming(
        previous_text=curr,
        current_text=curr,  # 文本没变，但状态变了
        delta_text="",
        previous_token_ids=[104, 1],
        current_token_ids=[104, 1],
        delta_token_ids=[],
        request=req,
    )

    assert parser.current_tool_name_sent
    assert res_name is not None
    assert res_name.tool_calls[0].function.name == "get_weather"
    assert res_name.tool_calls[0].index == 0


def test_extract_tool_calls_streaming_with_multiple_tools_transition(parser, req):
    """Test streaming with multiple tools transition"""
    prev_args = {"city": "BJ"}
    full_args_json = json.dumps(prev_args, ensure_ascii=False)

    parser.current_tool_id = 0
    parser.current_tool_name_sent = True
    parser.streamed_args_for_tool = [full_args_json]
    parser.prev_tool_call_arr = [{"name": "t1", "arguments": prev_args}]

    curr = (
        '<|tool_call_start|>[{"name": "t1", "arguments": {"city": "BJ"}}, '
        '{"name": "t2"}]'
    )

    res = parser.extract_tool_calls_streaming(
        "", curr, ', {"name": "t2"}]', [], [], [], req
    )

    assert parser.current_tool_id == 1
    assert not parser.current_tool_name_sent
    assert not res.tool_calls[0].id

    res_next = parser.extract_tool_calls_streaming(curr, curr, "", [], [], [], req)

    assert res_next is not None
    assert res_next.tool_calls[0].function.name == "t2"
    assert parser.current_tool_name_sent


def test_extract_tool_calls_streaming_does_not_emit_empty_arguments(parser, req):
    """Test that partial JSON empty arguments are not emitted prematurely."""
    parser.current_tool_id = -1
    parser.current_tool_name_sent = False
    parser.streamed_args_for_tool = []

    partial = '<|tool_call_start|>[{"name": "get_nums", "arguments": {}}]'

    res = parser.extract_tool_calls_streaming(
        previous_text="<|tool_call_start|>",
        current_text=partial,
        delta_text='[{"name": "get_nums", "arguments": {}',
        previous_token_ids=[104],
        current_token_ids=[104, 1],
        delta_token_ids=[1],
        request=req,
    )

    assert not res
    assert parser.current_tool_id == 0
    assert parser.streamed_args_for_tool == [""]

    res_name = parser.extract_tool_calls_streaming(
        previous_text=partial,
        current_text=partial,
        delta_text="",
        previous_token_ids=[104, 1],
        current_token_ids=[104, 1],
        delta_token_ids=[],
        request=req,
    )

    assert res_name
    assert res_name.tool_calls[0].function.name == "get_nums"
    assert parser.streamed_args_for_tool[0] == "{}"


def test_extract_tool_calls_streaming_func_name_with_empty_dict(parser, req):
    """Test streaming extraction of tool call with function name and empty arguments."""
    delta_tokens = [
        "<|tool_call_start|>[",
        '{"name": ',
        '"get_',
        'weather", "arguments": {}} ',
        "]<|tool_call_end|>",
    ]

    all_token_ids = [
        [104, 228],
        [45920, 13],
        [89108, 89216, 24440, 84063],
        [93166, 88870, 89216, 6731],
        [45932, 105],
    ]
    previous_texts = [""]
    all_previous_token_ids: list[list[int]] = [[]]
    for idx, delta_text in enumerate(delta_tokens):
        delta_token_ids = all_token_ids[idx]
        previous_text = previous_texts[0]
        previous_token_ids = all_previous_token_ids[0]
        current_text = previous_text + delta_text
        current_token_ids = previous_token_ids + list(delta_token_ids)

        delta_message = parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            req,
        )
        previous_texts[0] = current_text
        all_previous_token_ids[0] = current_token_ids

        if delta_message and delta_message.tool_calls:
            name = delta_message.tool_calls[0].function.name
            args = delta_message.tool_calls[0].function.arguments
            if name and args:
                assert name == "get_weather"
                assert json.loads(args) == {}


def test_extract_tool_calls_streaming_empty_dict(parser, req):
    """Test streaming extraction of tool call with function name and empty arguments."""
    delta_tokens = [
        "<|tool_call_start|>",
        "[",
        '{"',
        'name": ',
        '"get_',
        'weather", ',
        '"arguments": {',
        "}",
        "} ",
        "]",
        "<|tool_call_end|>",
    ]

    all_token_ids = [
        [104],
        *[[i] for i in range(len(delta_tokens) - 2)],
        [105],
    ]

    previous_texts = [""]
    all_previous_token_ids: list[list[int]] = [[]]
    for idx, delta_text in enumerate(delta_tokens):
        delta_token_ids = all_token_ids[idx]
        previous_text = previous_texts[0]
        previous_token_ids = all_previous_token_ids[0]
        current_text = previous_text + delta_text
        current_token_ids = previous_token_ids + list(delta_token_ids)

        delta_message = parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            req,
        )
        previous_texts[0] = current_text
        all_previous_token_ids[0] = current_token_ids

        if delta_message and delta_message.tool_calls:
            name = delta_message.tool_calls[0].function.name
            args = delta_message.tool_calls[0].function.arguments
            if name:
                assert name == "get_weather"
            if args:
                assert json.loads(args) == {}
