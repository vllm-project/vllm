# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the engine-based Granite parser.

Granite emits tool calls as a JSON array following a marker
(``<|tool_call|>`` for 3.0, ``<tool_call>`` for 3.1) with no closing
terminal; the ``]`` ends the region and surrounding prose is content. The
engine's ``tool_call_body_array`` mode splits the array into one call per
element and ``_granite_arg_converter`` carves each element's ``arguments``.
"""

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.trace_builder import _GRANITE_SCENARIOS, _build_granite
from vllm.parser.granite import GraniteParser, _granite_arg_converter

TOOL_TOKEN = "<|tool_call|>"
TOOL_STRING = "<tool_call>"

_GRANITE_VOCAB = {TOOL_TOKEN: 49154}


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(_GRANITE_VOCAB)


@pytest.fixture
def parser(mock_tokenizer):
    return GraniteParser(mock_tokenizer)


def _tokenize(text: str) -> list[tuple[int, str]]:
    """Tokenize like the real stream: ``<|tool_call|>`` is one special token,
    plain text becomes one token per character."""
    tokens: list[tuple[int, str]] = []
    i = 0
    while i < len(text):
        if text.startswith(TOOL_TOKEN, i):
            tokens.append((_GRANITE_VOCAB[TOOL_TOKEN], TOOL_TOKEN))
            i += len(TOOL_TOKEN)
        else:
            tokens.append((ord(text[i]), text[i]))
            i += 1
    return tokens


def _stream(parser, request, text: str, chunk_size: int):
    tokens = _tokenize(text)
    results = []
    previous_text = ""
    previous_token_ids: list[int] = []
    for start in range(0, len(tokens), chunk_size):
        batch = tokens[start : start + chunk_size]
        delta_text = "".join(t for _, t in batch)
        delta_token_ids = [tid for tid, _ in batch]
        current_text = previous_text + delta_text
        current_token_ids = previous_token_ids + delta_token_ids
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=tuple(previous_token_ids),
            current_token_ids=tuple(current_token_ids),
            delta_token_ids=tuple(delta_token_ids),
            request=request,
        )
        results.append(delta)
        previous_text = current_text
        previous_token_ids = current_token_ids
    finish = parser.finish_streaming()
    if finish is not None:
        results.append(finish)
    return results


def _collect_content(results) -> str:
    return "".join(d.content for d in results if d and d.content)


def _collect_names(results) -> list[str]:
    names: dict[int, str] = {}
    for d in results:
        if not (d and d.tool_calls):
            continue
        for tc in d.tool_calls:
            if tc.function and tc.function.name:
                names[tc.index] = names.get(tc.index, "") + tc.function.name
    return [names[i] for i in sorted(names)]


def _collect_args(results) -> dict[int, str]:
    args: dict[int, str] = {}
    for d in results:
        if not (d and d.tool_calls):
            continue
        for tc in d.tool_calls:
            if tc.function and tc.function.arguments:
                args[tc.index] = args.get(tc.index, "") + tc.function.arguments
    return args


class TestArgConverter:
    def test_complete_wrapper(self):
        raw = '{"name": "get_weather", "arguments": {"city": "SF"}}'
        assert _granite_arg_converter(raw, False) == '{"city": "SF"}'

    def test_arguments_before_name(self):
        raw = '{"arguments": {"city": "SF"}, "name": "get_weather"}'
        assert _granite_arg_converter(raw, False) == '{"city": "SF"}'

    def test_partial_before_arguments(self):
        assert _granite_arg_converter('{"name": "get_w', True) == ""

    def test_partial_inside_arguments(self):
        raw = '{"name": "x", "arguments": {"a": 1'
        assert _granite_arg_converter(raw, True) == '{"a": 1'

    def test_prefix_stability(self):
        # Each growing prefix must extend the previous converter output.
        full = '{"name": "x", "arguments": {"city": "Tokyo"}}'
        prev = ""
        for i in range(1, len(full) + 1):
            out = _granite_arg_converter(full[:i], True)
            assert out.startswith(prev) or prev.startswith(out) or out == ""
            if out:
                prev = out

    def test_missing_arguments_defaults(self):
        assert _granite_arg_converter('{"name": "x"}', False) == "{}"
        assert _granite_arg_converter('{"name": "x"}', True) == ""

    def test_non_object_arguments_rejected(self):
        with pytest.raises(ValueError, match="JSON object"):
            _granite_arg_converter('{"name": "x", "arguments": [1]}', False)


class TestNonStreaming:
    def test_plain_text(self, parser, mock_request):
        reasoning, content, tools = parser.parse("Just a reply.", mock_request)
        assert reasoning is None
        assert content == "Just a reply."
        assert tools is None

    def test_single_tool_token_marker(self, parser, mock_request):
        text = (
            f'{TOOL_TOKEN} [{{"name": "get_weather", "arguments": {{"city": "SF"}}}}]'
        )
        _, content, tools = parser.parse(text, mock_request)
        assert content is None
        assert [t.name for t in tools] == ["get_weather"]
        assert json.loads(tools[0].arguments) == {"city": "SF"}

    def test_single_tool_string_marker(self, parser, mock_request):
        # Granite 3.1 uses the plain-text ``<tool_call>`` marker.
        text = f'{TOOL_STRING} [{{"name": "get_time", "arguments": {{}}}}]'
        _, content, tools = parser.parse(text, mock_request)
        assert [t.name for t in tools] == ["get_time"]
        assert json.loads(tools[0].arguments) == {}

    def test_parallel_calls_one_array(self, parser, mock_request):
        text = (
            f'{TOOL_TOKEN} [{{"name": "a", "arguments": {{"x": 1}}}}, '
            f'{{"name": "b", "arguments": {{"y": [1, 2]}}}}]'
        )
        _, _, tools = parser.parse(text, mock_request)
        assert [t.name for t in tools] == ["a", "b"]
        assert json.loads(tools[0].arguments) == {"x": 1}
        assert json.loads(tools[1].arguments) == {"y": [1, 2]}

    def test_surrounding_text_is_content(self, parser, mock_request):
        text = (
            "Let me check.\n"
            f'{TOOL_TOKEN} [{{"name": "get_weather", "arguments": {{"city": "SF"}}}}]'
        )
        _, content, tools = parser.parse(text, mock_request)
        assert content == "Let me check."
        assert [t.name for t in tools] == ["get_weather"]

    def test_arguments_before_name(self, parser, mock_request):
        text = (
            f'{TOOL_TOKEN} [{{"arguments": {{"city": "SF"}}, "name": "get_weather"}}]'
        )
        _, _, tools = parser.parse(text, mock_request)
        assert [t.name for t in tools] == ["get_weather"]
        assert json.loads(tools[0].arguments) == {"city": "SF"}

    def test_marker_without_array_is_not_a_tool_call(self, parser, mock_request):
        # A non-array body after the marker (malformed) yields no tool call.
        text = f'{TOOL_TOKEN} {{"name": "func", "arguments": {{}}}}'
        _, _, tools = parser.parse(text, mock_request)
        assert not tools


class TestStreaming:
    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 11])
    def test_parallel_calls_chunk_invariance(self, parser, mock_request, chunk_size):
        text = (
            f'{TOOL_TOKEN} [{{"name": "get_weather", '
            f'"arguments": {{"city": "Tokyo"}}}}, '
            f'{{"name": "get_time", "arguments": {{"timezone": "Asia/Tokyo"}}}}]'
        )
        results = _stream(parser, mock_request, text, chunk_size)
        assert _collect_names(results) == ["get_weather", "get_time"]
        args = _collect_args(results)
        assert json.loads(args[0]) == {"city": "Tokyo"}
        assert json.loads(args[1]) == {"timezone": "Asia/Tokyo"}

    @pytest.mark.parametrize("chunk_size", [1, 3, 7])
    def test_surrounding_text_streaming(self, parser, mock_request, chunk_size):
        text = (
            "Let me check.\n"
            f'{TOOL_TOKEN} [{{"name": "get_weather", "arguments": {{"city": "SF"}}}}]'
        )
        results = _stream(parser, mock_request, text, chunk_size)
        assert _collect_names(results) == ["get_weather"]
        assert "Let me check." in _collect_content(results)

    def test_incomplete_array_at_eos(self, parser, mock_request):
        # Truncated mid-element must not raise and must not leak the marker.
        text = f'{TOOL_TOKEN} [{{"name": "func", "arguments": {{"a": 1'
        results = _stream(parser, mock_request, text, 3)
        assert TOOL_TOKEN not in _collect_content(results)


class TestReplay:
    @pytest.mark.parametrize("scenario", _GRANITE_SCENARIOS, ids=lambda s: s.id)
    def test_replay_scenarios(self, scenario):
        # ``_build_granite`` replays the sample through GraniteParser at
        # chunk_size=1 and asserts reasoning/content/tool_calls (validate=True).
        _build_granite(scenario)
