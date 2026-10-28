# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the engine-based Granite parser.

Granite emits tool calls as a JSON array following a marker
(``<|tool_call|>`` for 3.0, ``<tool_call>`` for 3.1) with no closing
terminal; the ``]`` ends the region and surrounding prose is content. The
engine's ``tool_call_body_array`` mode splits the array into one call per
element and ``_granite_arg_converter`` carves each element's ``arguments``.
"""

import dataclasses
import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.trace_builder import _GRANITE_SCENARIOS, _build_granite
from vllm.parser.granite import (
    GraniteParser,
    _granite_arg_converter,
    _granite_arg_value_extender,
    granite_config,
)
from vllm.parser.utils import scan_json_value_extender

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


_EXTENDER_EQUIVALENCE_OUTPUTS = [
    (
        f'{TOOL_TOKEN} [{{"name": "get_weather", '
        f'"arguments": {{"city": "San Francisco", "unit": "celsius"}}}}]'
    ),
    (
        f'{TOOL_TOKEN} [{{"name": "write", '
        f'"arguments": {{"path": "a.py", '
        f'"content": "line 1\\nprint(\\"hello\\\\world\\")\\n", '
        f'"mode": "w"}}}}]'
    ),
    (
        f'{TOOL_TOKEN} [{{"name": "edit", '
        f'"arguments": {{"edits": [{{"path": "a.py", "text": "x = \\"1\\""}}], '
        f'"dry_run": false, "count": 2}}}}]'
    ),
    (
        f'{TOOL_TOKEN} [{{"name": "get_weather", "arguments": {{"city": "Tokyo"}}}}, '
        f'{{"name": "get_time", "arguments": {{"timezone": "Asia/Tokyo"}}}}]'
    ),
    (
        f"Let me check.\n{TOOL_STRING} "
        f'[{{"name": "get_weather", "arguments": {{"city": "SF"}}}}]'
    ),
    f'{TOOL_TOKEN} [{{"arguments": {{"query": "select * from t"}}, "name": "sql"}}]',
    (
        f'{TOOL_TOKEN} [{{"name": "write", '
        f'"arguments": {{"content": "partial with \\"escape\\" and trailing'
    ),
    (
        f'{TOOL_TOKEN} [{{"name": "edit", '
        f'"arguments": {{"edits": [{{"text": "nested partial'
    ),
]


class TestArgValueExtender:
    @staticmethod
    def _make_parser(mock_tokenizer, *, use_extender: bool) -> GraniteParser:
        config = granite_config()
        if not use_extender:
            config = dataclasses.replace(config, arg_value_extender=None)
        return GraniteParser(mock_tokenizer, parser_engine_config=config)

    @pytest.mark.parametrize(
        "raw_args",
        [
            "",
            '{"name": "fn',
            '{"name": "fn", "arguments": {',
            '{"name": "fn", "arguments": {"k',
            '{"name": "fn", "arguments": {"k": "v", "k2',
            '{"name": "fn", "arguments": {"k": {"a": "b"}, "k2',
            '{"name": "fn", "arguments": {"k": ',
            '{"name": "fn", "arguments": {"k": 12',
            '{"name": "fn", "arguments": {"k": "v"}}',
            '{"name": "fn", "arguments": [1',
        ],
        ids=[
            "empty",
            "open_wrapper_name",
            "open_arguments_object",
            "open_top_level_key",
            "open_second_key_after_string",
            "open_second_key_after_nested_object",
            "after_colon_before_value",
            "numeric_value",
            "closed_arguments_object",
            "non_object_arguments",
        ],
    )
    def test_returns_none_outside_open_value_string(self, raw_args):
        assert _granite_arg_value_extender(raw_args) is None

    @pytest.mark.parametrize(
        "raw_args",
        [
            '{"name": "fn", "arguments": {"k": {"nested": "v',
            '{"name": "fn", "arguments": {"k": ["v',
        ],
        ids=["nested_object", "nested_array"],
    )
    def test_extends_open_nested_string_value(self, raw_args):
        nested_ext = _granite_arg_value_extender(raw_args)
        assert nested_ext is not None
        assert nested_ext("more") == "more"
        assert nested_ext('end"') is None

    def test_extends_open_top_level_string_value_and_escapes(self):
        ext = _granite_arg_value_extender(
            '{"name": "fn", "arguments": {"code": "hello '
        )
        assert ext is not None
        assert ext("world") == "world"
        assert ext('\\"escaped\\"') == '\\"escaped\\"'
        assert ext("trail\\") == "trail\\"
        assert ext('"still_escaped') == '"still_escaped'
        assert ext('done"') is None

    def test_scan_json_value_extender_rejects_non_object_start(self):
        assert scan_json_value_extender("") is None
        assert scan_json_value_extender('["x', 0) is None

    @pytest.mark.parametrize("nested", [False, True], ids=["top_level", "nested"])
    def test_long_string_arg_calls_converter_constantly(
        self, mock_tokenizer, mock_request, nested
    ):
        base = granite_config()
        counts: dict[int, int] = {}
        for n_lines in (20, 200):
            calls = 0

            def counting_converter(raw: str, partial: bool) -> str:
                nonlocal calls
                calls += 1
                return _granite_arg_converter(raw, partial)

            body = "\\n".join(
                f"line {i}: x = {i} + \\\"quoted\\\" + 'single'" for i in range(n_lines)
            )
            expected_body = "\n".join(
                f"line {i}: x = {i} + \"quoted\" + 'single'" for i in range(n_lines)
            )
            if nested:
                text = (
                    f'{TOOL_TOKEN} [{{"name": "edit", '
                    f'"arguments": {{"edits": '
                    f'[{{"path": "a.py", "text": "{body}"}}]}}}}]'
                )
                expected = {"edits": [{"path": "a.py", "text": expected_body}]}
            else:
                text = (
                    f'{TOOL_TOKEN} [{{"name": "write", '
                    f'"arguments": {{"content": "{body}", "mode": "w"}}}}]'
                )
                expected = {"content": expected_body, "mode": "w"}
            p = GraniteParser(
                mock_tokenizer,
                parser_engine_config=dataclasses.replace(
                    base, arg_converter=counting_converter
                ),
            )
            results = _stream(p, mock_request, text, chunk_size=1)
            args = _collect_args(results)
            assert json.loads(args[0]) == expected
            counts[n_lines] = calls
        assert counts[200] == counts[20]

    @pytest.mark.parametrize("text", _EXTENDER_EQUIVALENCE_OUTPUTS)
    def test_parse_matches_without_extender(self, mock_tokenizer, mock_request, text):
        with_ext = self._make_parser(mock_tokenizer, use_extender=True)
        without_ext = self._make_parser(mock_tokenizer, use_extender=False)

        r_with, c_with, tools_with = with_ext.parse(text, mock_request)
        r_without, c_without, tools_without = without_ext.parse(text, mock_request)
        assert r_with == r_without
        assert c_with == c_without
        assert [(t.name, t.arguments) for t in (tools_with or [])] == [
            (t.name, t.arguments) for t in (tools_without or [])
        ]

        ext_with = with_ext.extract_tool_calls(text, mock_request)
        ext_without = without_ext.extract_tool_calls(text, mock_request)
        assert ext_with.tools_called == ext_without.tools_called
        assert ext_with.content == ext_without.content
        assert [
            (t.function.name, t.function.arguments) for t in ext_with.tool_calls
        ] == [(t.function.name, t.function.arguments) for t in ext_without.tool_calls]

    @pytest.mark.parametrize("chunk_size", [1, 2, 5, 13])
    @pytest.mark.parametrize("text", _EXTENDER_EQUIVALENCE_OUTPUTS)
    def test_streaming_matches_without_extender(
        self, mock_tokenizer, mock_request, text, chunk_size
    ):
        with_ext = _stream(
            self._make_parser(mock_tokenizer, use_extender=True),
            mock_request,
            text,
            chunk_size,
        )
        without_ext = _stream(
            self._make_parser(mock_tokenizer, use_extender=False),
            mock_request,
            text,
            chunk_size,
        )

        assert _collect_content(with_ext) == _collect_content(without_ext)
        assert _collect_names(with_ext) == _collect_names(without_ext)
        assert _collect_args(with_ext) == _collect_args(without_ext)

        def normalize_deltas(deltas):
            return [
                None
                if d is None
                else (
                    d.reasoning,
                    d.content,
                    [
                        (tc.index, tc.function.name, tc.function.arguments)
                        for tc in (d.tool_calls or [])
                        if tc.function
                    ],
                )
                for d in deltas
            ]

        assert normalize_deltas(with_ext) == normalize_deltas(without_ext)


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
