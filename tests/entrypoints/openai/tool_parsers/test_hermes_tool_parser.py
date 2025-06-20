# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

from tests.entrypoints.openai.tool_parsers.utils import (
    run_tool_extraction, run_tool_extraction_streaming)
from vllm.entrypoints.openai.protocol import FunctionCall
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager

# Test cases for the Hermes-2-Pro format
SIMPLE_FUNCTION_OUTPUT = ('<tool_call>{"name": "get_weather", "arguments": '
                          '{"city": "San Francisco", "metric": "celsius"}}'
                          '</tool_call>')
SIMPLE_FUNCTION_CALL = FunctionCall(
    name="get_weather",
    arguments='{"city": "San Francisco", "metric": "celsius"}',
)

MORE_TYPES_FUNCTION_OUTPUT = (
    '<tool_call>{"name": "register_user", "arguments": {'
    '"name": "John Doe", "age": 37, "address": {"city": "San Francisco", '
    '"state": "CA"}, "role": null, "passed_test": true, "aliases": '
    '["John", "Johnny"]}}</tool_call>')
MORE_TYPES_FUNCTION_CALL = FunctionCall(
    name="register_user",
    arguments='{"name": "John Doe", "age": 37, "address": {"city": '
    '"San Francisco", "state": "CA"}, "role": null, "passed_test": true, '
    '"aliases": ["John", "Johnny"]}',
)

PARAMETERLESS_FUNCTION_OUTPUT = ('<tool_call>{"name": "get_weather", '
                                 '"arguments": {}}</tool_call>')
PARAMETERLESS_FUNCTION_CALL = FunctionCall(
    name="get_weather",
    arguments='{}',
)

EMPTY_DICT_FUNCTION_OUTPUT = ('<tool_call>{"name": "do_something_cool", '
                              '"arguments": {"additional_data": {}}}'
                              '</tool_call>')
EMPTY_DICT_FUNCTION_CALL = FunctionCall(
    name="do_something_cool",
    arguments='{"additional_data": {}}',
)

EMPTY_LIST_FUNCTION_OUTPUT = ('<tool_call>{"name": "do_something_cool", '
                              '"arguments": {"steps": []}}</tool_call>')
EMPTY_LIST_FUNCTION_CALL = FunctionCall(
    name="do_something_cool",
    arguments='{"steps": []}',
)

ESCAPED_STRING_FUNCTION_OUTPUT = (
    r'<tool_call>{"name": "get_weather", "arguments": {"city": '
    r'"Martha\'s Vineyard", "metric": "\"cool units\""}}</tool_call>')
ESCAPED_STRING_FUNCTION_CALL = FunctionCall(
    name="get_weather",
    arguments='{"city": "Martha\'s Vineyard", "metric": "\\"cool units\\""}',
)

TEXT_AND_TOOL_CALL_OUTPUT = ("Today's weather is nice. " +
                             SIMPLE_FUNCTION_OUTPUT)
PARALLEL_CALLS_OUTPUT = (SIMPLE_FUNCTION_OUTPUT +
                         MORE_TYPES_FUNCTION_OUTPUT)


@pytest.fixture
def mock_tokenizer():
    """Provides a mock tokenizer for the Hermes2ProToolParser."""
    tokenizer = MagicMock()
    # Simulate token splitting to test the tool_call_delta_buffer logic.
    tokenizer.encode.side_effect = lambda text, add_special_tokens=False: [
        ord(c) for c in text
    ]
    tokenizer.decode.side_effect = lambda token_ids: "".join(
        [chr(c) for c in token_ids])
    return tokenizer


@pytest.mark.parametrize("streaming", [True, False])
def test_no_tool_call(streaming: bool, mock_tokenizer: MagicMock):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("hermes")(
        mock_tokenizer)
    model_output = "How can I help you today?"

    content, tool_calls = run_tool_extraction(tool_parser,
                                              model_output,
                                              streaming=streaming)

    assert content == model_output
    assert len(tool_calls) == 0


TEST_CASES = [
    pytest.param(True,
                 SIMPLE_FUNCTION_OUTPUT, [SIMPLE_FUNCTION_CALL],
                 None,
                 id="simple_streaming"),
    pytest.param(False,
                 SIMPLE_FUNCTION_OUTPUT, [SIMPLE_FUNCTION_CALL],
                 None,
                 id="simple_nonstreaming"),
    pytest.param(True,
                 MORE_TYPES_FUNCTION_OUTPUT, [MORE_TYPES_FUNCTION_CALL],
                 None,
                 id="more_types_streaming"),
    pytest.param(False,
                 MORE_TYPES_FUNCTION_OUTPUT, [MORE_TYPES_FUNCTION_CALL],
                 None,
                 id="more_types_nonstreaming"),
    pytest.param(True,
                 PARAMETERLESS_FUNCTION_OUTPUT, [PARAMETERLESS_FUNCTION_CALL],
                 None,
                 id="parameterless_streaming"),
    pytest.param(False,
                 PARAMETERLESS_FUNCTION_OUTPUT, [PARAMETERLESS_FUNCTION_CALL],
                 None,
                 id="parameterless_nonstreaming"),
    pytest.param(True,
                 EMPTY_DICT_FUNCTION_OUTPUT, [EMPTY_DICT_FUNCTION_CALL],
                 None,
                 id="empty_dict_streaming"),
    pytest.param(False,
                 EMPTY_DICT_FUNCTION_OUTPUT, [EMPTY_DICT_FUNCTION_CALL],
                 None,
                 id="empty_dict_nonstreaming"),
    pytest.param(True,
                 EMPTY_LIST_FUNCTION_OUTPUT, [EMPTY_LIST_FUNCTION_CALL],
                 None,
                 id="empty_list_streaming"),
    pytest.param(False,
                 EMPTY_LIST_FUNCTION_OUTPUT, [EMPTY_LIST_FUNCTION_CALL],
                 None,
                 id="empty_list_nonstreaming"),
    pytest.param(True,
                 ESCAPED_STRING_FUNCTION_OUTPUT,
                 [ESCAPED_STRING_FUNCTION_CALL],
                 None,
                 id="escaped_string_streaming"),
    pytest.param(False,
                 ESCAPED_STRING_FUNCTION_OUTPUT,
                 [ESCAPED_STRING_FUNCTION_CALL],
                 None,
                 id="escaped_string_nonstreaming"),
    pytest.param(True,
                 PARALLEL_CALLS_OUTPUT,
                 [SIMPLE_FUNCTION_CALL, MORE_TYPES_FUNCTION_CALL],
                 None,
                 id="parallel_calls_streaming"),
    pytest.param(False,
                 PARALLEL_CALLS_OUTPUT,
                 [SIMPLE_FUNCTION_CALL, MORE_TYPES_FUNCTION_CALL],
                 None,
                 id="parallel_calls_nonstreaming"),
    pytest.param(True,
                 TEXT_AND_TOOL_CALL_OUTPUT, [SIMPLE_FUNCTION_CALL],
                 "Today's weather is nice. ",
                 id="text_and_tool_call_streaming"),
    pytest.param(False,
                 TEXT_AND_TOOL_CALL_OUTPUT, [SIMPLE_FUNCTION_CALL],
                 "Today's weather is nice. ",
                 id="text_and_tool_call_nonstreaming"),
]


@pytest.mark.parametrize(
    "streaming, model_output, expected_tool_calls, expected_content",
    TEST_CASES)
def test_tool_call(streaming: bool, model_output: str,
                   expected_tool_calls: list[FunctionCall],
                   expected_content: str, mock_tokenizer: MagicMock):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("hermes")(
        mock_tokenizer)

    content, tool_calls = run_tool_extraction(tool_parser,
                                              model_output,
                                              streaming=streaming)

    assert content == expected_content
    assert len(tool_calls) == len(expected_tool_calls)
    for actual, expected in zip(tool_calls, expected_tool_calls):
        assert actual.type == "function"
        assert actual.function == expected


def test_streaming_tool_call_with_fragmented_tags(mock_tokenizer: MagicMock):
    """
    Tests streaming when the <tool_call> and </tool_call> tags are split
    across multiple deltas.
    """
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("hermes")(
        mock_tokenizer)
    # Split <tool_call> into "<", "tool", "_", "call", ">"
    model_output_deltas = [
        "<", "tool", "_", "call", ">",
        '{"name": "get_weather", "arguments": {"city": "SF"}}',
        "<", "/", "tool", "_", "call", ">"
    ]

    reconstructor = run_tool_extraction_streaming(
        tool_parser, model_output_deltas, assert_one_tool_per_delta=False)

    assert reconstructor.other_content == ""
    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function == FunctionCall(
        name="get_weather", arguments='{"city": "SF"}')


def test_streaming_tool_call_with_large_steps(mock_tokenizer: MagicMock):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("hermes")(
        mock_tokenizer)
    model_output_deltas = [
        '<tool_call>{"name": "get_weather", "arguments": {"city": "San',
        ' Francisco", "metric": "celsius"}}</tool_call>',
        '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>',
        '<tool_call>{"name": "do_something_cool", "arguments": {"steps": []}}'
        '</tool_call>',
    ]

    reconstructor = run_tool_extraction_streaming(
        tool_parser, model_output_deltas, assert_one_tool_per_delta=False)

    assert reconstructor.other_content == ""
    assert len(reconstructor.tool_calls) == 3
    assert reconstructor.tool_calls[0].function == SIMPLE_FUNCTION_CALL
    assert reconstructor.tool_calls[1].function == PARAMETERLESS_FUNCTION_CALL
    assert reconstructor.tool_calls[2].function == EMPTY_LIST_FUNCTION_CALL


@pytest.mark.parametrize("streaming", [False])
def test_regex_timeout_handling(streaming: bool, mock_tokenizer: MagicMock):
    """test regex timeout is handled gracefully"""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("hermes")(
        mock_tokenizer)

    fake_problematic_input = "hello world<tool_call>" + '{"a":' * 100

    # create a mock regex that raises TimeoutError
    mock_regex = MagicMock()
    mock_regex.findall.side_effect = TimeoutError("Regex timeout")

    with patch.object(tool_parser, 'tool_call_regex', mock_regex):
        content, tool_calls = run_tool_extraction(tool_parser,
                                                  fake_problematic_input,
                                                  streaming=streaming)

        # should treat as regular text when regex times out
        assert content == fake_problematic_input
        assert len(tool_calls) == 0
        mock_regex.findall.assert_called_once()


def test_streaming_with_mixed_content(mock_tokenizer: MagicMock):
    """Tests streaming with a mix of text and tool calls."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("hermes")(
        mock_tokenizer)
    model_output_deltas = [
        "Thinking... ",
        "I should call a tool. ",
        "<tool_call>",
        '{"name": "get_weather", ',
        '"arguments": {"city": "Seoul"}}',
        "</tool_call>",
        " The tool call is complete."
    ]

    reconstructor = run_tool_extraction_streaming(
        tool_parser, model_output_deltas, assert_one_tool_per_delta=False)

    expected_content = "Thinking... I should call a tool.  The tool call is complete."
    assert reconstructor.other_content == expected_content
    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function == FunctionCall(
        name="get_weather", arguments='{"city": "Seoul"}')