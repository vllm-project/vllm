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


def _collect_args_fragments(parser, chunks, request=None):
    """Feed *chunks* through the streaming parser and return the concatenated
    argument fragments for each tool call index.

    Each element in *chunks* is either:
    - a ``str``: used as ``delta_text`` with matching ``current_text``
    - a ``(current_text, delta_text, delta_token_ids)`` tuple for full control
    """
    previous = ""
    args_by_index: dict[int, str] = {}
    for chunk in chunks:
        if isinstance(chunk, tuple):
            current, delta, delta_ids = chunk
        else:
            current = previous + chunk
            delta = chunk
            delta_ids = []

        result = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=delta_ids,
            request=request,
        )
        if result and result.tool_calls:
            for tc in result.tool_calls:
                if tc.function and tc.function.arguments is not None:
                    args_by_index.setdefault(tc.index, "")
                    args_by_index[tc.index] += tc.function.arguments

        previous = current

    return args_by_index


def test_streaming_stream_interval_gt1(minimax_m2_tool_parser):
    """Simulate stream_interval > 1: the parameter + closing tags arrive in
    one big chunk, then the EOS token arrives with delta_text=''.

    The parser must emit the remaining arguments on the empty-delta call.
    """
    parser = minimax_m2_tool_parser
    parser._reset_streaming_state()

    full_text = (
        '<minimax:tool_call><invoke name="get_weather">'
        '<parameter name="city">Seattle</parameter>'
        '<parameter name="days">5</parameter>'
        "</invoke></minimax:tool_call>"
    )

    # Chunk 1: tool call start + function header only
    chunk1 = '<minimax:tool_call><invoke name="get_weather">'
    # Chunk 2: all parameters + closing tags arrive at once (big interval)
    chunk2 = (
        '<parameter name="city">Seattle</parameter>'
        '<parameter name="days">5</parameter>'
        "</invoke></minimax:tool_call>"
    )

    chunks = [
        chunk1,
        chunk2,
        # EOS arrives with empty delta; current_text is the full output
        (full_text, "", [99]),  # 99 = some non-special EOS token id
    ]

    args = _collect_args_fragments(parser, chunks)

    # All args must be emitted and form valid JSON
    assert 0 in args
    combined = args[0]
    parsed = json.loads(combined)
    # No request schema provided, so all values remain strings
    assert parsed == {"city": "Seattle", "days": "5"}

    # prev_tool_call_arr must be finalized
    assert len(parser.prev_tool_call_arr) == 1
    assert parser.prev_tool_call_arr[0]["arguments"] == {
        "city": "Seattle",
        "days": "5",
    }


def test_streaming_single_chunk_complete(minimax_m2_tool_parser):
    """Entire tool call arrives in a single delta (stream_interval very
    large or short output).  The parser must handle it in one pass.
    """
    parser = minimax_m2_tool_parser
    parser._reset_streaming_state()

    single = (
        '<minimax:tool_call><invoke name="get_weather">'
        '<parameter name="city">Seattle</parameter>'
        "</invoke></minimax:tool_call>"
    )

    args = _collect_args_fragments(parser, [single])

    assert 0 in args
    parsed = json.loads(args[0])
    assert parsed == {"city": "Seattle"}
    assert parser.prev_tool_call_arr[0]["arguments"] == {"city": "Seattle"}
