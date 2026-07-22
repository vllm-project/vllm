# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the engine-based Llama 3.x/4 JSON parser (llama3_json /
llama4_json).

The format is a bare JSON envelope — ``{"name": ..., "parameters": {...}}``
optionally prefixed with ``<|python_tag|>``, parallel calls separated by
``;``/newlines/nothing, and no end marker (calls close on JSON balance) —
so this file carries the replay coverage that the generic harness in
test_replay.py provides for marker-based formats (llama_json is skipped
there: it has no TOOL_END terminal).
"""

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.trace_builder import build_samples
from vllm.parser.engine.parser_engine_config import ParserState
from vllm.parser.llama_json import (
    LlamaJsonParser,
    _args_value_span,
    _envelope_name,
    _llama_arg_converter,
    _top_level_name,
)

PYTHON_TAG = "<|python_tag|>"

_LLAMA_VOCAB = {
    PYTHON_TAG: 128010,
    "<|eot_id|>": 128009,
    "<|eom_id|>": 128008,
}


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(_LLAMA_VOCAB)


@pytest.fixture
def parser(mock_tokenizer):
    return LlamaJsonParser(mock_tokenizer)


_MARKERS = sorted(_LLAMA_VOCAB, key=len, reverse=True)


def _tokenize(text: str) -> list[tuple[int, str]]:
    """Markers are atomic special tokens; plain text is one token per
    character (matching the mock tokenizer's ``chr``-based decode)."""
    tokens: list[tuple[int, str]] = []
    i = 0
    while i < len(text):
        for marker in _MARKERS:
            if text.startswith(marker, i):
                tokens.append((_LLAMA_VOCAB[marker], marker))
                i += len(marker)
                break
        else:
            tokens.append((ord(text[i]), text[i]))
            i += 1
    return tokens


def _stream_tokens(parser, request, tokens, chunk_size: int):
    results = []
    previous_text = ""
    previous_token_ids: list[int] = []
    for start in range(0, len(tokens), chunk_size):
        batch = tokens[start : start + chunk_size]
        delta_text = "".join(t for _, t in batch)
        delta_token_ids = [tid for tid, _ in batch]
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=previous_text + delta_text,
            delta_text=delta_text,
            previous_token_ids=tuple(previous_token_ids),
            current_token_ids=tuple(previous_token_ids) + tuple(delta_token_ids),
            delta_token_ids=tuple(delta_token_ids),
            request=request,
        )
        results.append(delta)
        previous_text += delta_text
        previous_token_ids += delta_token_ids
    results.append(parser.finish_streaming())
    return results


def _stream(parser, request, text: str, chunk_size: int):
    """Stream production-shaped deltas: ``chunk_size`` tokens per delta,
    with delta_token_ids covering every token (specials and text)."""
    return _stream_tokens(parser, request, _tokenize(text), chunk_size)


def _stream_text_only(parser, request, text: str, chunk_size: int):
    """Stream text-only deltas chunked at arbitrary character boundaries —
    exercises the text-lexing path, including markers split across chunks."""
    results = []
    previous_text = ""
    for start in range(0, len(text), chunk_size):
        delta_text = text[start : start + chunk_size]
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=previous_text + delta_text,
            delta_text=delta_text,
            previous_token_ids=(),
            current_token_ids=(),
            delta_token_ids=(),
            request=request,
        )
        results.append(delta)
        previous_text += delta_text
    results.append(parser.finish_streaming())
    return results


def _accumulate(results):
    """Fold streamed deltas into (content, [per-index tool call dicts])."""
    content_parts: list[str] = []
    calls: dict[int, dict] = {}
    for delta in results:
        if delta is None:
            continue
        if delta.content:
            content_parts.append(delta.content)
        for tc in delta.tool_calls or []:
            call = calls.setdefault(
                tc.index, {"id": None, "type": None, "name": None, "args": ""}
            )
            if tc.id:
                assert call["id"] is None, "id must be sent exactly once"
                call["id"] = tc.id
            if tc.type:
                call["type"] = tc.type
            if tc.function and tc.function.name:
                assert call["name"] is None, "name must arrive whole, once"
                call["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                call["args"] += tc.function.arguments
    return "".join(content_parts), [{"index": i, **calls[i]} for i in sorted(calls)]


class TestSpanHelpers:
    def test_complete_envelope_parameters(self):
        raw = '{"name": "f", "parameters": {"x": 1}}'
        assert _llama_arg_converter(raw, False) == '{"x": 1}'

    def test_complete_envelope_arguments(self):
        raw = '{"name": "f", "arguments": {"x": 1}}'
        assert _llama_arg_converter(raw, True) == '{"x": 1}'

    def test_partial_before_args_key(self):
        assert _llama_arg_converter('{"name": "f"', True) == ""
        assert _llama_arg_converter('{"name": "f"}', False) == "{}"

    def test_partial_inside_args_prefix_stable(self):
        raw = '{"name": "f", "parameters": {"x": "lo'
        partial = _llama_arg_converter(raw, True)
        assert partial == '{"x": "lo'
        full = _llama_arg_converter(raw + 'ng"}}', True)
        assert full.startswith(partial)

    def test_args_keys_inside_strings_ignored(self):
        raw = '{"name": "use \\"parameters\\": here", "parameters": {"a": 1}}'
        assert _llama_arg_converter(raw, False) == '{"a": 1}'

    @pytest.mark.parametrize(
        ("value", "span"),
        [("[1, 2]", "[1, 2]"), ('"text"', '"text"'), ("42", "42"), ("null", "null")],
    )
    def test_non_object_value_spans(self, value, span):
        assert _args_value_span(f'{{"name": "f", "parameters": {value}}}') == span

    def test_top_level_name_ignores_nested(self):
        raw = '{"parameters": {"name": "inner"}, "name": "outer"}'
        assert _top_level_name(raw) == "outer"
        assert _top_level_name('{"parameters": {"name": "inner"}}') is None
        # A completed name value is returned even mid-envelope; an
        # unterminated one is not.
        assert _top_level_name('{"name": "f"') == "f"
        assert _top_level_name('{"name": "f') is None

    def test_envelope_classification(self):
        # A call needs an args key next to "name" (legacy KeyError'd
        # otherwise); {"name": ...} alone counts only once closed.
        call = '{"name": "f", "parameters": {"x": 1}}'
        assert _envelope_name(call, closed=False) == "f"
        assert _envelope_name('{"name": "f"}', closed=True) == "f"
        assert _envelope_name('{"name": "f"}', closed=False) is None
        # Prose JSON with a "name" field is not a call (user-data shape).
        prose = '{"name": "John", "age": 30, "city": "New York"}'
        assert _envelope_name(prose, closed=True) is None
        assert _envelope_name('{"name": "f", "id": 1}', closed=True) is None
        # Extra keys are fine once an args key is present (legacy accepted).
        assert (
            _envelope_name('{"id": 1, "name": "f", "parameters": {}}', closed=True)
            == "f"
        )

    def test_top_level_name_escapes(self):
        # An escaped quote must not terminate the span, and completed
        # names are JSON-decoded like legacy json.loads did.
        assert _top_level_name('{"name": "a\\"') is None
        assert _top_level_name('{"name": "a\\"b"}') == 'a"b'
        assert _top_level_name('{"name": "tool\\u00e9"}') == "toolé"


class TestNonStreaming:
    def test_no_tool_calls_content_passthrough(self, parser, mock_request):
        text = "This is just some text without any tool calls"
        result = parser.extract_tool_calls_from_content(text, mock_request)
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == text

    def test_single_tool_call_bare_json(self, parser, mock_request):
        text = '{"name": "get_weather", "parameters": {"city": "SF"}}'
        result = parser.extract_tool_calls_from_content(text, mock_request)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.type == "function"
        assert tc.id.startswith("chatcmpl-tool-")
        assert tc.function.name == "get_weather"
        assert tc.function.arguments == '{"city": "SF"}'
        assert result.content is None

    def test_python_tag_prefixed_call_arguments_key(self, parser, mock_request):
        text = '<|python_tag|>{"name": "search", "arguments": {"q": "test"}}'
        result = parser.extract_tool_calls_from_content(text, mock_request)
        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "search"
        assert result.tool_calls[0].function.arguments == '{"q": "test"}'
        assert result.content is None

    def test_python_tag_non_json_is_content(self, parser, mock_request):
        # ipython-style output after the tag is not a tool call; legacy
        # streaming black-holed this entirely.
        text = '<|python_tag|>print("hello")'
        result = parser.extract_tool_calls_from_content(text, mock_request)
        assert result.tools_called is False
        assert result.content == 'print("hello")'

    @pytest.mark.parametrize(
        "sep",
        ["; ", " ; ", ";", ";\n", "\n", " ", ",", "", "<|python_tag|>"],
        ids=[
            "semi-space",
            "space-semi-space",
            "semi",
            "semi-newline",
            "newline",
            "space",
            "comma",
            "back-to-back",
            "python-tag",
        ],
    )
    def test_parallel_calls_separator_variants(self, parser, mock_request, sep):
        # Legacy scanned every "{"-rooted object regardless of separator;
        # "" (back-to-back) is what the xgrammar "llama" structural tag
        # forces for tool_choice="required".
        text = sep.join(
            [
                '{"name": "a", "parameters": {"q": "t1"}}',
                '{"name": "b", "parameters": {}}',
                '{"name": "c", "parameters": {"n": 3}}',
            ]
        )
        result = parser.extract_tool_calls_from_content(text, mock_request)
        assert result.tools_called is True
        assert [tc.function.name for tc in result.tool_calls] == ["a", "b", "c"]
        assert result.tool_calls[0].function.arguments == '{"q": "t1"}'
        assert result.tool_calls[1].function.arguments == "{}"
        assert result.tool_calls[2].function.arguments == '{"n": 3}'

    def test_prose_before_call_kept_trailing_dropped(self, parser, mock_request):
        # Deliberate change vs legacy (content was None): leading prose is
        # returned as content; text after the last call is dropped.
        text = 'Here is the result: {"name": "f", "parameters": {}} more?'
        result = parser.extract_tool_calls_from_content(text, mock_request)
        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "f"
        assert result.content == "Here is the result:"

    @pytest.mark.parametrize(
        "text",
        [
            'Here is JSON: {"a": 1} and more text after',
            "Sure: function f() { return 1; } and that is it",
            'The config is {"a": 1}; see the docs for details.',
            'Set {"a": "b"}; then restart the app to apply changes',
            '{"parameters": {"x": 1}}',
        ],
    )
    def test_prose_json_restored_as_content(self, parser, mock_request, text):
        # A balanced or unbalanced "{...}" with no top-level "name" is
        # ordinary content and must be restored byte-identically.
        result = parser.extract_tool_calls_from_content(text, mock_request)
        assert result.tools_called is False
        assert result.content == text

    def test_prose_json_with_name_field_not_a_call(self, parser, mock_request):
        # Observed E2E: a prose JSON example carrying a "name" field must
        # not become a fabricated call losing the other fields.
        text = (
            'Example: {"name": "John", "age": 30, "city": "New York"} — '
            'now the weather: {"name": "get_weather", "parameters": {"city": "SF"}}'
        )
        result = parser.extract_tool_calls_from_content(text, mock_request)
        assert [tc.function.name for tc in result.tool_calls] == ["get_weather"]
        assert result.tool_calls[0].function.arguments == '{"city": "SF"}'
        assert result.content == (
            'Example: {"name": "John", "age": 30, "city": "New York"} — '
            "now the weather:"
        )

    def test_name_with_extra_keys_no_args_is_prose(self, parser, mock_request):
        text = '{"name": "f", "id": 1}'
        result = parser.extract_tool_calls_from_content(text, mock_request)
        assert result.tools_called is False
        assert result.content == text

    def test_prose_json_then_real_call(self, parser, mock_request):
        # Prose JSON before a real call must not merge into it.
        text = (
            'Config {"parameters": {"a": 1}} used: '
            '{"name": "f", "parameters": {"b": 2}}'
        )
        result = parser.extract_tool_calls_from_content(text, mock_request)
        assert [tc.function.name for tc in result.tool_calls] == ["f"]
        assert result.tool_calls[0].function.arguments == '{"b": 2}'
        assert result.content == 'Config {"parameters": {"a": 1}} used:'

    def test_name_only_call_empty_args(self, parser, mock_request):
        # Deliberate change vs legacy (KeyError fallback to content).
        result = parser.extract_tool_calls_from_content('{"name": "f"}', mock_request)
        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "f"
        assert result.tool_calls[0].function.arguments == "{}"

    def test_nested_json_quotes_brackets_escapes(self, parser, mock_request):
        text = (
            '{"name": "parserTool", "parameters": {'
            '"query": "test {value} [complex]", '
            '"text": "He said \\"Hello {world}\\"", '
            '"config": {"database": {"pool": {"size": 10}}}}}'
        )
        result = parser.extract_tool_calls_from_content(text, mock_request)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["query"] == "test {value} [complex]"
        assert args["text"] == 'He said "Hello {world}"'
        assert args["config"]["database"]["pool"]["size"] == 10

    def test_semicolon_inside_string_arg_not_split(self, parser, mock_request):
        text = (
            '{"name": "run", "parameters": {"cmd": "echo a; echo b"}}; '
            '{"name": "g", "parameters": {}}'
        )
        result = parser.extract_tool_calls_from_content(text, mock_request)
        assert [tc.function.name for tc in result.tool_calls] == ["run", "g"]
        assert result.tool_calls[0].function.arguments == '{"cmd": "echo a; echo b"}'

    def test_nested_name_key_does_not_hijack(self, parser, mock_request):
        # The engine's regex name extraction would pick "inner"; the
        # parser must use the envelope's top-level "name".
        text = '{"parameters": {"name": "inner"}, "name": "outer"}'
        result = parser.extract_tool_calls_from_content(text, mock_request)
        assert [tc.function.name for tc in result.tool_calls] == ["outer"]
        assert result.tool_calls[0].function.arguments == '{"name": "inner"}'

    @pytest.mark.parametrize(
        ("value", "expected_args"),
        [
            ("[1, 2]", "[1, 2]"),
            ('"text"', '"text"'),
            ("42", "42"),
            ("null", "null"),
            ('"{\\"x\\": 1}"', '"{\\"x\\": 1}"'),
        ],
        ids=["array", "string", "number", "null", "string-encoded-json"],
    )
    def test_non_object_args_verbatim(self, parser, mock_request, value, expected_args):
        # Non-object values stream as their verbatim JSON span, matching
        # the legacy json.dumps output.
        text = f'{{"name": "f", "parameters": {value}}}'
        result = parser.extract_tool_calls_from_content(text, mock_request)
        assert result.tools_called is True
        assert result.tool_calls[0].function.arguments == expected_args

    def test_truncated_args_partial_span(self, parser, mock_request):
        text = '{"name": "f", "parameters": {"x": "lo'
        result = parser.extract_tool_calls_from_content(text, mock_request)
        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "f"
        assert result.tool_calls[0].function.arguments == '{"x": "lo'

    def test_via_registered_tool_parser_adapter(self, mock_tokenizer, mock_request):
        from vllm.tool_parsers import ToolParserManager

        for name in ("llama3_json", "llama4_json"):
            cls = ToolParserManager.get_tool_parser(name)
            assert cls.engine_based_streaming is True
            assert cls.structural_tag_model == "llama"
            adapter = cls(mock_tokenizer)
            result = adapter.extract_tool_calls(
                '{"name": "f", "parameters": {"x": 1}}', mock_request
            )
            assert result.tools_called is True
            assert result.tool_calls[0].function.name == "f"

    def test_llama4_tokenizer_without_python_tag(self, mock_request):
        # Legacy raised RuntimeError when <|python_tag|> was missing from
        # the vocab (real Llama-4 tokenizers); the text terminal covers it.
        parser = LlamaJsonParser(make_mock_tokenizer({"<|other|>": 7}))
        result = parser.extract_tool_calls_from_content(
            '<|python_tag|>{"name": "f", "parameters": {}}', mock_request
        )
        assert [tc.function.name for tc in result.tool_calls] == ["f"]


class TestStreaming:
    def test_basic_name_then_incremental_args(self, parser, mock_request):
        text = '{"name": "get_weather", "parameters": {"city": "San Francisco"}}'
        results = _stream(parser, mock_request, text, chunk_size=3)
        content, calls = _accumulate(results)
        assert content == ""
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["type"] == "function"
        assert calls[0]["id"].startswith("chatcmpl-tool-")
        assert calls[0]["args"] == '{"city": "San Francisco"}'

    def test_parallel_calls_index_accumulation(self, parser, mock_request):
        text = (
            '<|python_tag|>{"name": "a", "parameters": {"q": "t1"}}; '
            '{"name": "b", "parameters": {"n": 2}}'
        )
        results = _stream(parser, mock_request, text, chunk_size=4)
        content, calls = _accumulate(results)
        assert content == ""
        assert [c["index"] for c in calls] == [0, 1]
        assert [c["name"] for c in calls] == ["a", "b"]
        assert calls[0]["args"] == '{"q": "t1"}'
        assert calls[1]["args"] == '{"n": 2}'

    def test_prose_json_before_call_keeps_dense_indices(self, parser, mock_request):
        # The prose JSON opens (and retracts) an engine slot; the real
        # call must still stream at index 0.
        text = '{"x": 1}; {"name": "save", "parameters": {"y": 2}}'
        results = _stream(parser, mock_request, text, chunk_size=4)
        content, calls = _accumulate(results)
        assert content == '{"x": 1}; '
        assert [c["index"] for c in calls] == [0]
        assert calls[0]["name"] == "save"
        assert calls[0]["args"] == '{"y": 2}'

    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 64])
    @pytest.mark.parametrize(
        "text",
        [
            '{"name": "f", "parameters": {"x": 1}}',
            '<|python_tag|>{"name": "f", "arguments": {"s": "a; b"}}',
            '{"name": "a", "parameters": {}}; {"name": "b", "parameters": {}}',
            '{"name": "a", "parameters": {}}\n{"name": "b", "parameters": {}}',
            '{"name": "a", "parameters": {}}{"name": "b", "parameters": {}}',
            '{"name": "esc", "parameters": {"t": "say \\"hi\\""}}',
            "plain text with {braces: not json} inside",
            "Sure: function f() { return 1; } and that is it",
            '{"parameters": {"name": "inner"}, "name": "outer"}',
            '{"name": "f", "parameters": "text"}',
        ],
    )
    def test_chunk_invariance_token_mode(
        self, mock_tokenizer, mock_request, text, chunk_size
    ):
        reference = LlamaJsonParser(mock_tokenizer).extract_tool_calls_from_content(
            text, mock_request
        )
        parser = LlamaJsonParser(mock_tokenizer)
        results = _stream(parser, mock_request, text, chunk_size)
        content, calls = _accumulate(results)
        assert content == (reference.content or "")
        assert [c["name"] for c in calls] == [
            tc.function.name for tc in reference.tool_calls
        ]
        assert [c["args"] for c in calls] == [
            tc.function.arguments for tc in reference.tool_calls
        ]

    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 64])
    def test_chunk_invariance_text_only(self, mock_tokenizer, mock_request, chunk_size):
        # Marker split across text-only chunks must still be recognized.
        text = '<|python_tag|>{"name": "f", "parameters": {"x": 1}}'
        parser = LlamaJsonParser(mock_tokenizer)
        results = _stream_text_only(parser, mock_request, text, chunk_size)
        content, calls = _accumulate(results)
        assert content == ""
        assert [c["name"] for c in calls] == ["f"]
        assert calls[0]["args"] == '{"x": 1}'

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 100])
    @pytest.mark.parametrize(
        "text",
        ['\t{"a": 1}\t\n', ' {"a": 1} ok', "   "],
        ids=["tab-wrapped", "space-wrapped", "ws-only"],
    )
    def test_leading_ws_before_prose_json(
        self, mock_tokenizer, mock_request, text, chunk_size
    ):
        # Whitespace around prose JSON must survive streaming at every
        # chunking: a candidate slot must not trigger the engine's
        # whitespace-before-tools drop when it turns out to be prose.
        parser = LlamaJsonParser(mock_tokenizer)
        results = _stream_text_only(parser, mock_request, text, chunk_size)
        content, calls = _accumulate(results)
        assert calls == []
        assert content == text

    def test_ws_before_real_call_dropped(self, parser, mock_request):
        text = '  {"name": "f", "parameters": {}}'
        results = _stream(parser, mock_request, text, chunk_size=3)
        content, calls = _accumulate(results)
        assert content == ""
        assert [c["name"] for c in calls] == ["f"]

    def test_name_with_escaped_quote_streams_decoded(self, parser, mock_request):
        text = '{"name": "a\\"b", "parameters": {"x": 1}}'
        results = _stream(parser, mock_request, text, chunk_size=1)
        _, calls = _accumulate(results)
        assert [c["name"] for c in calls] == ['a"b']

    def test_token_id_tag_lookalike_is_content(self, parser, mock_request):
        # Prose "<|python_tag|>" made of ordinary text tokens (not the
        # special token id) must not trigger a tool call once real token
        # ids have been seen.
        text = 'the literal tag is "<|python_tag|>" ok'
        tokens = [(ord(ch), ch) for ch in text]
        results = _stream_tokens(parser, mock_request, tokens, chunk_size=2)
        content, calls = _accumulate(results)
        assert calls == []
        assert content == text

    def test_special_tokens_auto_dropped(self, parser, mock_request):
        text = '{"name": "f", "parameters": {}}<|eot_id|>'
        results = _stream(parser, mock_request, text, chunk_size=3)
        content, calls = _accumulate(results)
        assert content == ""
        assert [c["name"] for c in calls] == ["f"]
        assert calls[0]["args"] == "{}"

    def test_truncated_json_flushed_at_finish(self, parser, mock_request):
        # Legacy lost parsed-but-unstreamed args when generation stopped.
        text = '{"name": "f", "parameters": {"x": "lo'
        results = _stream(parser, mock_request, text, chunk_size=5)
        content, calls = _accumulate(results)
        assert content == ""
        assert calls[0]["name"] == "f"
        assert calls[0]["args"] == '{"x": "lo'

    def test_streaming_matches_non_streaming_case_table(
        self, mock_tokenizer, mock_request
    ):
        cases = [
            "plain text no tools",
            '{"name": "get_weather", "parameters": {"city": "SF"}}',
            '<|python_tag|>{"name": "f", "arguments": {"x": 1}}',
            '{"name": "a", "parameters": {"q": "t"}}; {"name": "b", "parameters": {}}',
            '{"name": "a", "parameters": {}}\n{"name": "b", "parameters": {}}',
            '{"name": "a", "parameters": {}}{"name": "b", "parameters": {}}',
            '{"name": "run", "parameters": {"cmd": "echo a; echo b"}}',
            '{"parameters": {"x": 1}}',
            '{"name": "f"}',
            '{"name": "f", "parameters": [1, 2]}',
            '{"name": "f", "parameters": "text"}',
            '{"name": "f", "parameters": 42}',
            '{"name": "f", "arguments": "{\\"x\\": 1}"}',
            '<|python_tag|>print("hello")',
            'Here is JSON: {"a": 1} and more text after',
            "Sure: function f() { return 1; } and that is it",
            'The config is {"a": 1}; see the docs for details.',
            'Config {"parameters": {"a": 1}} used: '
            '{"name": "f", "parameters": {"b": 2}}',
            '{"parameters": {"name": "inner"}, "name": "outer"}',
            'Example: {"name": "John", "age": 30} then {"name": "f", "parameters": {}}',
            '{"name": "f", "id": 1}',
        ]
        for text in cases:
            reference = LlamaJsonParser(mock_tokenizer).extract_tool_calls_from_content(
                text, mock_request
            )
            parser = LlamaJsonParser(mock_tokenizer)
            content, calls = _accumulate(_stream(parser, mock_request, text, 3))
            # Framework-wide: non-streaming strips content whitespace
            # around tool calls; streaming cannot retract emitted text.
            assert content.rstrip() == (reference.content or "").rstrip(), text
            assert [c["name"] for c in calls] == [
                tc.function.name for tc in reference.tool_calls
            ], text
            assert [c["args"] for c in calls] == [
                tc.function.arguments for tc in reference.tool_calls
            ], text


class TestToolChoiceNone:
    """tool_choice="none" (including tool-less requests, where "none" is
    the default) must return tool-call-shaped JSON as content.

    Deviation from the engine convention of dropping tool markup: a bare
    JSON envelope is indistinguishable from ordinary JSON content, so
    dropping it would eat legitimate output (legacy passed it through).
    """

    @pytest.fixture
    def none_request(self, mock_request):
        mock_request.tool_choice = "none"
        mock_request.tools = [{"type": "function"}]
        return mock_request

    @pytest.fixture
    def toolless_request(self, mock_request):
        mock_request.tool_choice = "none"
        mock_request.tools = None
        return mock_request

    @pytest.mark.parametrize(
        "text",
        [
            'Result: {"name": "f", "parameters": {"x": 1}}',
            '{"name": "f", "parameters": {"cmd": "a; b"}}; {"name": "g"}',
            'Here is the JSON you asked for: {"name": "John", "age": 30}',
        ],
    )
    def test_non_streaming(self, mock_tokenizer, none_request, text):
        parser = LlamaJsonParser(mock_tokenizer)
        result = parser.extract_tool_calls_from_content(text, none_request)
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == text

    def test_toolless_request_passthrough(self, mock_tokenizer, toolless_request):
        # A request with no tools defaults to tool_choice="none"; JSON in
        # ordinary content must survive.
        text = 'Here is the JSON you asked for: {"name": "John", "age": 30}'
        parser = LlamaJsonParser(mock_tokenizer)
        result = parser.extract_tool_calls_from_content(text, toolless_request)
        assert result.tools_called is False
        assert result.content == text

    @pytest.mark.parametrize("chunk_size", [1, 4])
    def test_streaming(self, mock_tokenizer, none_request, chunk_size):
        text = '{"name": "f", "parameters": {"cmd": "a; b"}}; {"name": "g"}'
        parser = LlamaJsonParser(mock_tokenizer)
        results = _stream(parser, none_request, text, chunk_size)
        content, calls = _accumulate(results)
        assert calls == []
        assert content == text


class TestReasoningDefaults:
    def test_no_reasoning_phase(self, parser, mock_request):
        assert parser.reasoning_ended is True
        assert parser.is_reasoning_end([1, 2, 3]) is True
        assert parser.count_reasoning_tokens([1, 2, 3]) == 0
        reasoning, content = parser.extract_reasoning("hello", mock_request)
        assert reasoning is None
        assert content == "hello"

    def test_initial_state_is_content(self, parser):
        assert parser.parser_engine_config.initial_state == ParserState.CONTENT


class TestTraceBuilderSamples:
    def test_samples_self_validate(self):
        samples = build_samples("llama_json")
        assert samples
