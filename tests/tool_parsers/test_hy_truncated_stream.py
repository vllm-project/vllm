# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Regression tests: truncated streams must still yield parseable arguments.

HYV3/HYV4 stream tool-call arguments incrementally and intentionally withhold
the closing JSON brace until the tool call is complete. When the stream is cut
short (max_tokens / stop) the parser never sees the closing tag, so the
framework's finalize step (``_append_unstreamed_tool_args``) must be able to
recover the withheld tail. These tests exercise that path via
``get_remaining_unstreamed_args`` with a mock tokenizer (no model download).
"""

import json
from types import SimpleNamespace

import pytest

from tests.parser.engine.replay_harness import MockTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.tool_parsers.hy_v3_tool_parser import HYV3ToolParser
from vllm.tool_parsers.hy_v4_tool_parser import HYV4ToolParser

_MARKERS = [
    "<tool_calls>",
    "</tool_calls>",
    "<tool_call>",
    "</tool_call>",
    "<tool_sep>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
]
_WORDS = ["get_weather", "get_current_date", "Beijing", "city", "date",
          "2026-03-30", "\n"]


def _make_tokenizer() -> MockTokenizer:
    vocab = {tok: 1000 + i for i, tok in enumerate(_MARKERS + _WORDS)}
    return MockTokenizer(vocab=vocab, tokens=[])


def _make_tools():
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(name="get_current_date", parameters={}),
        ),
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "date": {"type": "string"},
                    },
                },
            ),
        ),
    ]


@pytest.fixture
def mock_request():
    class Req:
        tools = _make_tools()
        tool_choice = "auto"
    return Req()


def _simulate_streaming(parser, deltas, request):
    """Feed deltas exactly like the serving loop, then finalize."""
    results = []
    previous_text, previous_token_ids = "", []
    vocab = parser.vocab
    for delta_text in deltas:
        current_text = previous_text + delta_text
        delta_token_ids = [tid for tok, tid in vocab.items() if tok in delta_text]
        current_token_ids = previous_token_ids + delta_token_ids
        result = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
            request=request,
        )
        results.append(result)
        previous_text = current_text
        previous_token_ids = current_token_ids
    # Finalize: replicate abstract_parser._append_unstreamed_tool_args, which
    # appends the remaining arguments to the last tool-call chunk streamed.
    remaining = parser.get_remaining_unstreamed_args()
    if remaining:
        for result in reversed(results):
            if (result is not None and result.tool_calls
                    and result.tool_calls[-1].function is not None):
                fn = result.tool_calls[-1].function
                fn.arguments = (fn.arguments or "") + remaining
                break
    return results


def _collect_all_args(results):
    """Cumulatively concatenate argument deltas (like the client does)."""
    tool_calls = {}
    for result in results:
        if result is None or not result.tool_calls:
            continue
        for tc in result.tool_calls:
            idx = tc.index
            if idx not in tool_calls:
                tool_calls[idx] = {
                    "name": tc.function.name or "",
                    "args": tc.function.arguments or "",
                }
            else:
                if tc.function.name:
                    tool_calls[idx]["name"] += tc.function.name
                if tc.function.arguments:
                    tool_calls[idx]["args"] += tc.function.arguments
    return [tool_calls[i] for i in sorted(tool_calls)]


class TestHYV3TruncatedStream:

    def test_truncated_stream_closes_json(self, mock_request):
        """Stream cut at max_tokens mid-args: output must stay parseable."""
        parser = HYV3ToolParser(_make_tokenizer(), tools=_make_tools())
        deltas = [
            "<tool_calls>",
            "\n<tool_call>",
            "get_weather",
            "<tool_sep>",
            "\n<arg_key>city</arg_key>",
            "\n<arg_value>Beijing</arg_value>",
        ]  # truncated: no </tool_call>
        results = _simulate_streaming(parser, deltas, mock_request)
        collected = _collect_all_args(results)
        assert len(collected) == 1
        assert collected[0]["name"] == "get_weather"
        assert json.loads(collected[0]["args"]) == {"city": "Beijing"}

    def test_complete_stream_unaffected(self, mock_request):
        """Full tool call: remaining is empty, no duplicated tail."""
        parser = HYV3ToolParser(_make_tokenizer(), tools=_make_tools())
        deltas = [
            "<tool_calls>",
            "\n<tool_call>",
            "get_weather",
            "<tool_sep>",
            "\n<arg_key>city</arg_key>",
            "\n<arg_value>Beijing</arg_value>",
            "\n<arg_key>date</arg_key>",
            "\n<arg_value>2026-03-30</arg_value>",
            "\n</tool_call>",
            "\n</tool_calls>",
        ]
        results = _simulate_streaming(parser, deltas, mock_request)
        collected = _collect_all_args(results)
        assert len(collected) == 1
        assert json.loads(collected[0]["args"]) == {
            "city": "Beijing", "date": "2026-03-30",
        }

    def test_truncated_after_name_only(self, mock_request):
        """Cut before any arg closes: no fabricated arguments."""
        parser = HYV3ToolParser(_make_tokenizer(), tools=_make_tools())
        deltas = [
            "<tool_calls>",
            "\n<tool_call>",
            "get_current_date",
            "<tool_sep>",
        ]  # truncated before any arg_key
        results = _simulate_streaming(parser, deltas, mock_request)
        collected = _collect_all_args(results)
        # The tool name was streamed; arguments were never started, so the
        # finalizer must not fabricate any.
        assert len(collected) == 1
        assert collected[0]["name"] == "get_current_date"
        assert collected[0]["args"] == ""


class TestHYV4TruncatedStream:

    def test_truncated_stream_closes_json(self, mock_request):
        parser = HYV4ToolParser(_make_tokenizer(), tools=_make_tools())
        deltas = [
            "<tool_calls>",
            "\n<tool_call>",
            "get_weather",
            "\n<arg_key>city</arg_key>",
            "\n<arg_value>Beijing</arg_value>",
        ]  # truncated: no </tool_call>
        results = _simulate_streaming(parser, deltas, mock_request)
        collected = _collect_all_args(results)
        assert len(collected) == 1
        assert collected[0]["name"] == "get_weather"
        assert json.loads(collected[0]["args"]) == {"city": "Beijing"}

    def test_complete_stream_unaffected(self, mock_request):
        parser = HYV4ToolParser(_make_tokenizer(), tools=_make_tools())
        deltas = [
            "<tool_calls>",
            "\n<tool_call>",
            "get_weather",
            "\n<arg_key>city</arg_key>",
            "\n<arg_value>Beijing</arg_value>",
            "\n<arg_key>date</arg_key>",
            "\n<arg_value>2026-03-30</arg_value>",
            "\n</tool_call>",
            "\n</tool_calls>",
        ]
        results = _simulate_streaming(parser, deltas, mock_request)
        collected = _collect_all_args(results)
        assert len(collected) == 1
        assert json.loads(collected[0]["args"]) == {
            "city": "Beijing", "date": "2026-03-30",
        }

    def test_truncated_guided_path_closes_json(self, mock_request):
        """Same truncation under structural-tag guided decoding (string-marker
        path): the withheld tail must still be emitted at stream end."""
        request = SimpleNamespace(
            tools=_make_tools(),
            tool_choice="required",
            structured_outputs=SimpleNamespace(structural_tag=object()),
        )
        parser = HYV4ToolParser(_make_tokenizer(), tools=_make_tools())
        deltas = [
            "<tool_calls>",
            "\n<tool_call>",
            "get_weather",
            "\n<arg_key>city</arg_key>",
            "\n<arg_value>Beijing</arg_value>",
        ]  # truncated: no </tool_call>
        results = _simulate_streaming(parser, deltas, request)
        collected = _collect_all_args(results)
        assert len(collected) == 1
        assert collected[0]["name"] == "get_weather"
        assert json.loads(collected[0]["args"]) == {"city": "Beijing"}
