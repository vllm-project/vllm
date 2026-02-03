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
