# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.tool_parsers.minimax_m2_tool_parser import (
    MinimaxM2ToolParser,
)

pytestmark = pytest.mark.cpu_test


class FakeTokenizer:
    """Minimal fake tokenizer that exposes the attributes used by the
    parser: a truthy model_tokenizer marker and a vocab mapping for the
    special tokens.
    """

    def __init__(self):
        self.model_tokenizer = True
        # The parser will look up start/end tokens by their literal strings
        self.vocab = {
            "<minimax:tool_call>": 1,
            "</minimax:tool_call>": 2,
        }

    def get_vocab(self):
        return self.vocab


@pytest.fixture
def minimax_m2_tool_parser():
    return MinimaxM2ToolParser(FakeTokenizer())


def test_extract_tool_calls_streaming_incremental(minimax_m2_tool_parser):
    parser = minimax_m2_tool_parser
    parser._reset_streaming_state()
    chunks = [
        "<minimax:tool_call>",
        '<invoke name="get_weather">',
        '<parameter name="city">',
        "Seattle</parameter>",
        "</invoke></minimax:tool_call>",
    ]
    previous = ""
    for chunk in chunks:
        current = previous + chunk
        delta = chunk
        parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )
        previous = current

    assert len(parser.prev_tool_call_arr) == 1
    entry = parser.prev_tool_call_arr[0]

    assert entry["name"] == "get_weather"
    args = entry["arguments"]
    assert args["city"] == "Seattle"


def test_streaming_minimax_m2_multiple_invokes(minimax_m2_tool_parser):
    parser = minimax_m2_tool_parser
    parser._reset_streaming_state()

    chunks = [
        "<minimax:tool_call>",
        '<invoke name="search_web">',
        '<parameter name="query_tag">',
        '["technology", "events"]</parameter>',
        '<parameter name="query_list">',
        '["OpenAI", "latest", "release"]</parameter>',
        "</invoke>",
        '<invoke name="search_web">',
        '<parameter name="query_tag">',
        '["technology", "events"]</parameter>',
        '<parameter name="query_list">',
        '["Gemini", "latest", "release"]</parameter>',
        "</invoke>",
        "</minimax:tool_call>",
    ]
    previous = ""
    for chunk in chunks:
        current = previous + chunk
        delta = chunk
        parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )
        previous = current

    assert len(parser.prev_tool_call_arr) == 2

    for entry, expect_model in zip(parser.prev_tool_call_arr, ["OpenAI", "Gemini"]):
        assert entry["name"] == "search_web"
        args = json.dumps(entry["arguments"])
        assert "technology" in args and "events" in args
        assert expect_model in args

    # check streamed_args_for_tool for serving_chat.py
    for index in range(2):
        expected_call = parser.prev_tool_call_arr[index].get("arguments", {})
        expected_call = json.dumps(expected_call)
        actual_call = parser.streamed_args_for_tool[index]
        assert expected_call == actual_call


def test_streaming_with_extra_attributes_in_tags(minimax_m2_tool_parser):
    """Test that parsing handles extra attributes in tags gracefully.

    This tests the fix for issue #32827 where descriptions containing
    parentheses (e.g., "(e.g. ls -la)") could cause the model to echo
    these back as additional attributes in the XML tags.
    """
    parser = minimax_m2_tool_parser
    parser._reset_streaming_state()

    # Simulate model output where it echoes description as extra attribute
    # This could happen when tool description contains "(e.g. ls -la)"
    chunks = [
        "<minimax:tool_call>",
        '<invoke name="execute_command" description="Run a command">',
        '<parameter name="command" type="string">',
        "ls -la</parameter>",
        "</invoke></minimax:tool_call>",
    ]
    previous = ""
    for chunk in chunks:
        current = previous + chunk
        delta = chunk
        parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )
        previous = current

    assert len(parser.prev_tool_call_arr) == 1
    entry = parser.prev_tool_call_arr[0]

    # Should correctly extract function name, ignoring extra attributes
    assert entry["name"] == "execute_command"
    args = entry["arguments"]
    # Should correctly extract parameter name, ignoring extra attributes
    assert "command" in args
    assert args["command"] == "ls -la"


def test_streaming_with_parentheses_in_value(minimax_m2_tool_parser):
    """Test that parentheses in parameter values are handled correctly.

    This is related to issue #32827 - ensure values with parentheses
    like "(e.g. ls -la)" don't break parsing.
    """
    parser = minimax_m2_tool_parser
    parser._reset_streaming_state()

    chunks = [
        "<minimax:tool_call>",
        '<invoke name="help">',
        '<parameter name="topic">',
        "commands (e.g. ls -la, cat file.txt)</parameter>",
        "</invoke></minimax:tool_call>",
    ]
    previous = ""
    for chunk in chunks:
        current = previous + chunk
        delta = chunk
        parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )
        previous = current

    assert len(parser.prev_tool_call_arr) == 1
    entry = parser.prev_tool_call_arr[0]

    assert entry["name"] == "help"
    args = entry["arguments"]
    assert args["topic"] == "commands (e.g. ls -la, cat file.txt)"


def test_non_streaming_with_extra_attributes(minimax_m2_tool_parser):
    """Test non-streaming parsing with extra attributes in tags."""
    parser = minimax_m2_tool_parser

    # Create a mock request
    class MockRequest:
        tools = None

    model_output = (
        "<minimax:tool_call>"
        '<invoke name="test_func" extra="ignored">'
        '<parameter name="arg1" type="string">value1</parameter>'
        '<parameter name="arg2" description="(e.g. example)">value2</parameter>'
        "</invoke>"
        "</minimax:tool_call>"
    )

    result = parser.extract_tool_calls(model_output, MockRequest())

    assert result.tools_called is True
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert tool_call.function.name == "test_func"

    args = json.loads(tool_call.function.arguments)
    assert args["arg1"] == "value1"
    assert args["arg2"] == "value2"
