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
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.parser import llama_json
from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine_config import ParserState
from vllm.parser.llama_json import (
    PYTHON_END,
    PYTHON_START,
    LlamaJsonParser,
    _args_value_span,
    _closeable_prefix,
    _envelope_name,
    _llama_arg_converter,
    _scan_json_value,
    _splice_types,
    _top_level_name,
    _ValueScan,
)
from vllm.parser.parser_manager import ParserManager

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
        # A call needs an args key next to "name": legacy KeyError'd on
        # the bare {"name": ...} form and fell back to content.
        assert _envelope_name('{"name": "f", "parameters": {"x": 1}}') == "f"
        assert _envelope_name('{"name": "f"}') is None
        assert _envelope_name('{"name": "f"') is None
        # Prose JSON with a "name" field is not a call (user-data shape).
        assert _envelope_name('{"name": "John", "age": 30, "city": "NY"}') is None
        assert _envelope_name('{"name": "f", "id": 1}') is None
        # Extra keys are fine once an args key is present (legacy accepted).
        assert _envelope_name('{"id": 1, "name": "f", "parameters": {}}') == "f"

    def test_top_level_name_escapes(self):
        # An escaped quote must not terminate the span, and completed
        # names are JSON-decoded like legacy json.loads did.
        assert _top_level_name('{"name": "a\\"') is None
        assert _top_level_name('{"name": "a\\"b"}') == 'a"b'
        assert _top_level_name('{"name": "tool\\u00e9"}') == "toolé"

    def test_escaped_key_names_recognized(self):
        # Keys with JSON escapes must match after decoding, as legacy's
        # json.loads did (llama_json.py _decode_key).
        assert _envelope_name('{"na\\u006de": "f", "parameters": {}}') == "f"
        esc_args = '{"name": "f", "arg\\u0075ments": {"x": 1}}'
        assert _args_value_span(esc_args) == '{"x": 1}'

    @pytest.mark.parametrize(
        ("span", "completed"),
        [
            ('{"x": 1}', '{"x": 1}'),
            ('{"x": "lo', '{"x": "lo"}'),
            ('{"x": "lo\\', '{"x": "lo"}'),
            ('{"a": 1, "b": ', '{"a": 1}'),
            ('{"a": 1, "b": tr', '{"a": 1}'),
            ('{"a": [1, {"b": "c', '{"a": [1, {"b": "c"}]}'),
            ('["a", ', '["a"]'),
            ("{invalid json}", "{}"),
            ("{", "{}"),
            ('"str', '"str"'),
            ("42", "42"),
            ("null", "null"),
        ],
    )
    def test_incomplete_spans_completed_to_valid_json(self, span, completed):
        # Truncated/malformed argument spans are cut back to their last
        # completable point and closed by appending only, so the final
        # arguments always parse and never retract streamed text.
        end, closers = _closeable_prefix(span)
        assert span[:end] + closers == completed
        json.loads(completed)
        raw = f'{{"name": "f", "parameters": {span}'
        assert _llama_arg_converter(raw, False) == completed
        assert completed.startswith(_llama_arg_converter(raw, True))

    @pytest.mark.parametrize(
        ("span", "completed"),
        [
            ('{"x": 0}', '{"x": 0}'),
            ('{"x": -0}', '{"x": -0}'),
            ('{"x": 1.5}', '{"x": 1.5}'),
            ('{"x": 1e5}', '{"x": 1e5}'),
            ('{"x": 1E+5}', '{"x": 1E+5}'),
            ('{"x": 1.5e-3}', '{"x": 1.5e-3}'),
            ('{"x": 0.}', '{"x": 0}'),
            ('{"x": 0.e1}', '{"x": 0}'),
            ('{"x": 1.e+2}', '{"x": 1}'),
            ('{"x": 1.e-2}', '{"x": 1}'),
            ('{"x": 1e}', '{"x": 1}'),
            ('{"x": 1e+}', '{"x": 1}'),
            ('{"x": 01}', '{"x": 0}'),
            ('{"x": -}', "{}"),
            ('{"x": .5}', "{}"),
            ('{"x": +1}', "{}"),
            ('{"x": "a\nb"}', '{"x": "a"}'),
            ('{"x": "a\rb"}', '{"x": "a"}'),
            ('{"x": "a\tb"}', '{"x": "a"}'),
            ('{"x": "a\x00b"}', '{"x": "a"}'),
            ('{"x": "a\\nb"}', '{"x": "a\\nb"}'),
            ('{"o": {"x": "a\nb"}, "y": 2}', '{"o": {"x": "a"}}'),
            ('{"l": ["a\nb", "c"]}', '{"l": ["a"]}'),
            ('{"a": 1, "k\ny": 2}', '{"a": 1}'),
        ],
    )
    def test_invalid_tokens_cut_back_to_valid_json(self, span, completed):
        # A number that never becomes valid (no digits after the point,
        # a leading zero) and a raw control character inside a string are
        # lexical errors *inside* a token; they are cut back like a
        # truncation so the reported arguments still parse.
        end, closers = _closeable_prefix(span)
        assert span[:end] + closers == completed
        json.loads(completed)
        raw = f'{{"name": "f", "parameters": {span}'
        assert _llama_arg_converter(raw, False) == completed

    @pytest.mark.parametrize(
        ("span", "completed"),
        [
            ('{"a": "\\u0041"}', '{"a": "\\u0041"}'),
            ('{"a": "x\\u00e9y"}', '{"a": "x\\u00e9y"}'),
            ('{"k\\u00e9y": 1}', '{"k\\u00e9y": 1}'),
            ('{"a": "x\\u00e9', '{"a": "x\\u00e9"}'),
            ('{"a": "x\\', '{"a": "x"}'),
            ('{"a": "x\\u', '{"a": "x"}'),
            ('{"a": "x\\u0', '{"a": "x"}'),
            ('{"a": "x\\u00', '{"a": "x"}'),
            ('{"a": "x\\u00e', '{"a": "x"}'),
            ('{"a": "x\\u00e"}', '{"a": "x"}'),
            ('{"a": "x\\uZZZZ"}', '{"a": "x"}'),
            ('{"a": "x\\u00g9"}', '{"a": "x"}'),
            ('{"a": "x\\qy"}', '{"a": "x"}'),
            ('{"k\\u00e": 1}', "{}"),
        ],
    )
    def test_unicode_escapes_cut_back_to_valid_json(self, span, completed):
        # A "\\uXXXX" escape is six characters, so every prefix of it must
        # cut back to before the backslash: appending a quote inside the
        # escape emits '{"a": "x\\u"}', which does not parse.  Invalid hex
        # and unknown escapes are the same lexical error, one token later.
        end, closers = _closeable_prefix(span)
        assert span[:end] + closers == completed
        json.loads(completed)
        raw = f'{{"name": "f", "parameters": {span}'
        assert _llama_arg_converter(raw, False) == completed
        assert completed.startswith(_llama_arg_converter(raw, True))

    @pytest.mark.parametrize(
        ("span", "completed"),
        [
            ('{"l": [], "a": "z"}', '{"l": [], "a": "z"}'),
            ('{"o": {}, "a": "z"}', '{"o": {}, "a": "z"}'),
            ('{"l": [[], {}], "a": "z"}', '{"l": [[], {}], "a": "z"}'),
            ('{"l": [], "a": "z', '{"l": [], "a": "z"}'),
            ('{"o": {}, "a": "z', '{"o": {}, "a": "z"}'),
            ('{"l": [', '{"l": []}'),
            ('{"l": []', '{"l": []}'),
            ('{"l": [],', '{"l": []}'),
            ('{"o": {', '{"o": {}}'),
            ('{"o": {}', '{"o": {}}'),
        ],
    )
    def test_empty_containers_do_not_end_the_span(self, span, completed):
        # An empty "[]"/"{}" closes from a state no other input reaches
        # (nothing has been read since the opener), and a scanner that
        # stops there still returns valid JSON -- '{"l": []}' -- while
        # silently dropping every later member.
        end, closers = _closeable_prefix(span)
        assert span[:end] + closers == completed
        json.loads(completed)
        raw = f'{{"name": "f", "parameters": {span}'
        assert _llama_arg_converter(raw, False) == completed

    @pytest.mark.parametrize(
        "document",
        [
            "{}",
            "[]",
            '{"a": 1}',
            '{"l": [], "a": "z"}',
            '{"o": {}, "a": "z"}',
            '{"l": [[], {}], "n": null}',
            '[{}, [], "s", 1.5e-3, true, false, null]',
            '{"a": "x\\u00e9y", "b": [1, {"c": {}}]}',
        ],
    )
    def test_complete_document_needs_no_repair(self, document):
        # The repair must be a no-op on well-formed arguments: consuming
        # the whole document with no closers is what says nothing was cut.
        json.loads(document)
        assert _closeable_prefix(document) == (len(document), "")

    def test_closeable_prefix_is_monotonic(self):
        # Prefix-stability of the streamed text depends on the cut point
        # never moving backwards as more of the value arrives.
        span = '{"a": [1, 2.5e3, {"b": "c\\"d"}], "e": true, "f": null}'
        prev = 0
        for i in range(len(span) + 1):
            end, closers = _closeable_prefix(span[:i])
            assert end >= prev
            if end:
                json.loads(span[:end] + closers)
            prev = end


class TestArgCoalescing:
    """Guards the O(n^2) fix: the engine emits ~one arg event per char, so
    the arg-rescan must run once per feed, not once per char."""

    def _arg(self, value, idx=0):
        return SemanticEvent(EventType.ARG_VALUE_CHUNK, value, idx)

    def test_consecutive_same_index_merged(self):
        events = [self._arg(c) for c in '{"x": 1}']
        out = LlamaJsonParser._coalesce_arg_events(events)
        assert len(out) == 1
        assert out[0].type == EventType.ARG_VALUE_CHUNK
        assert out[0].value == '{"x": 1}'
        assert out[0].tool_index == 0

    def test_boundary_events_break_runs(self):
        events = [
            SemanticEvent(EventType.TOOL_CALL_START, "{", 0),
            self._arg("a", 0),
            self._arg("b", 0),
            SemanticEvent(EventType.TOOL_CALL_END, "", 0),
            SemanticEvent(EventType.TOOL_CALL_START, "{", 1),
            self._arg("c", 1),
            self._arg("d", 1),
            SemanticEvent(EventType.TOOL_CALL_END, "", 1),
        ]
        out = LlamaJsonParser._coalesce_arg_events(events)
        assert [(e.type, e.value, e.tool_index) for e in out] == [
            (EventType.TOOL_CALL_START, "{", 0),
            (EventType.ARG_VALUE_CHUNK, "ab", 0),
            (EventType.TOOL_CALL_END, "", 0),
            (EventType.TOOL_CALL_START, "{", 1),
            (EventType.ARG_VALUE_CHUNK, "cd", 1),
            (EventType.TOOL_CALL_END, "", 1),
        ]

    def test_different_index_not_merged(self):
        events = [self._arg("a", 0), self._arg("b", 1)]
        out = LlamaJsonParser._coalesce_arg_events(events)
        assert [(e.value, e.tool_index) for e in out] == [("a", 0), ("b", 1)]


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

    @pytest.mark.parametrize(
        ("trailing", "names"),
        [
            (" I used {searchTool} for this", ["f"]),
            (' the args were {"query": "x"}', ["f"]),
            ("{", ["f"]),
            # A trailing object that *does* carry a name is a second call
            # with repaired arguments, not junk -- the deliberate deviation
            # documented as behavior change 3.
            ('{"name": "g", "parameters": {invalid}}', ["f", "g"]),
        ],
        ids=["brace-prose", "quoted-args", "lone-brace", "malformed-call"],
    )
    def test_completed_call_survives_a_bad_object_after_it(
        self, parser, mock_request, trailing, names
    ):
        # Legacy was all-or-nothing: any unparsable brace region *after* a
        # good call made extract_tool_calls return the whole output as
        # content, discarding the call the client was waiting on
        # (vllm-project/vllm#49923, bug 2).  Checked against the legacy
        # parser: it returns tools_called=False for brace-prose,
        # quoted-args and malformed-call.  "lone-brace" is the one shape
        # legacy already got right, kept here as a parity case.
        text = '{"name": "f", "parameters": {"query": "a"}}' + trailing
        result = parser.extract_tool_calls_from_content(text, mock_request)
        assert result.tools_called is True
        assert [tc.function.name for tc in result.tool_calls] == names
        assert result.tool_calls[0].function.arguments == '{"query": "a"}'
        for call in result.tool_calls:
            json.loads(call.function.arguments)

    @pytest.mark.parametrize(
        ("span", "arguments"),
        [
            ('{"a": "x\\u', '{"a": "x"}'),
            ('{"a": "x\\u00e9', '{"a": "x\\u00e9"}'),
            ('{"a": "x\\uZZZZ"}', '{"a": "x"}'),
            ('{"l": [], "a": "z"}', '{"l": [], "a": "z"}'),
            ('{"o": {}, "a": "z"}', '{"o": {}, "a": "z"}'),
        ],
    )
    def test_truncated_arguments_reach_the_client_parseable(
        self, parser, mock_request, span, arguments
    ):
        # What the span helpers repair is what the client is handed: a
        # generation cut mid-escape must not emit unparsable arguments,
        # and one holding an empty container must not lose its later keys.
        text = f'{{"name": "f", "parameters": {span}'
        result = parser.extract_tool_calls_from_content(text, mock_request)
        emitted = result.tool_calls[0].function.arguments
        assert emitted == arguments
        json.loads(emitted)

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

    @pytest.mark.parametrize(
        "text",
        ['{"name": "f"}', '{"name": "f"', '{"name": "Alice"}'],
        ids=["closed", "truncated", "person-shaped"],
    )
    def test_name_only_envelope_is_content(self, parser, mock_request, text):
        # An envelope with no parameters/arguments key is not a call:
        # legacy KeyError'd and fell back to content, and a name-only
        # object is far more often prose than a call.  Nothing has been
        # streamed for it at that point, so the fallback is safe.
        result = parser.extract_tool_calls_from_content(text, mock_request)
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == text

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

    @pytest.mark.parametrize(
        ("text", "expected_args"),
        [
            ('{"name": "f", "parameters": {"x": "lo', '{"x": "lo"}'),
            ('{"name": "f", "parameters": {"a": 1, "b": ', '{"a": 1}'),
            ('{"name": "f", "parameters": ["a", ', '["a"]'),
            ('{"name": "f", "parameters":', "{}"),
            ('{"name": "f", "parameters": {invalid json}}', "{}"),
        ],
        ids=["open-string", "dangling-key", "open-array", "no-value", "malformed"],
    )
    def test_incomplete_args_completed_to_valid_json(
        self, parser, mock_request, text, expected_args
    ):
        # The name is streamed as soon as the args key is seen and cannot
        # be retracted, so the call stands — but its arguments must still
        # be parseable JSON.  This is a deliberate deviation from legacy
        # non-streaming, which failed closed to content when raw_decode
        # rejected the envelope; it matches the other engine parsers and
        # is pending maintainer sign-off (see the re-review response).
        result = parser.extract_tool_calls_from_content(text, mock_request)
        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "f"
        assert result.tool_calls[0].function.arguments == expected_args
        json.loads(result.tool_calls[0].function.arguments)

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
        # The already-streamed '{"x": "lo' is completed by appending.
        text = '{"name": "f", "parameters": {"x": "lo'
        results = _stream(parser, mock_request, text, chunk_size=5)
        content, calls = _accumulate(results)
        assert content == ""
        assert calls[0]["name"] == "f"
        assert calls[0]["args"] == '{"x": "lo"}'

    @pytest.mark.parametrize("chunk_size", [1, 4])
    def test_every_truncation_offset_parity_and_valid_args(
        self, mock_tokenizer, mock_request, chunk_size
    ):
        # Generation can stop at any byte.  At every cut, streaming must
        # still agree with non-streaming and any reported call must carry
        # parseable arguments.
        full = '{"name": "get_weather", "parameters": {"city": "SF", "n": 3}}'
        for i in range(len(full) + 1):
            text = full[:i]
            reference = LlamaJsonParser(mock_tokenizer).extract_tool_calls_from_content(
                text, mock_request
            )
            parser = LlamaJsonParser(mock_tokenizer)
            content, calls = _accumulate(
                _stream(parser, mock_request, text, chunk_size)
            )
            assert content.rstrip() == (reference.content or "").rstrip(), text
            assert [(c["name"], c["args"]) for c in calls] == [
                (tc.function.name, tc.function.arguments) for tc in reference.tool_calls
            ], text
            for call in calls:
                json.loads(call["args"])

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 100])
    def test_invalid_token_repair_parity_and_valid_args(
        self, mock_tokenizer, mock_request, chunk_size
    ):
        # Malformed numbers and raw control characters are repaired by
        # cutting the value back, which must happen identically in both
        # modes and at every feed size.
        bodies = [
            '{"x": 1.5e-3}',
            '{"x": 0.e1}',
            '{"x": 1.e+2}',
            '{"x": 1.e-2}',
            '{"x": 0.}',
            '{"x": 1e}',
            '{"x": 01}',
            '{"x": .5}',
            '{"x": +1}',
            '{"x": -}',
            '{"x": "a\nb"}',
            '{"x": "a\rb"}',
            '{"x": "a\tb"}',
            '{"x": "a\x00b"}',
            '{"o": {"x": "a\nb"}, "y": 2}',
            '{"l": ["a\nb", "c"]}',
        ]
        for body in bodies:
            text = '{"name": "f", "parameters": ' + body + "}"
            reference = LlamaJsonParser(mock_tokenizer).extract_tool_calls_from_content(
                text, mock_request
            )
            parser = LlamaJsonParser(mock_tokenizer)
            _, calls = _accumulate(_stream(parser, mock_request, text, chunk_size))
            assert [(c["name"], c["args"]) for c in calls] == [
                (tc.function.name, tc.function.arguments) for tc in reference.tool_calls
            ], body
            for call in calls:
                json.loads(call["args"])

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
            '{"name": "f", "parameters": {invalid json}}',
            '{"name": "f", "parameters": {"x": "lo',
            '{"name": "f", "parameters": {"a": 1, "b": ',
            '{"name": "f"}{"name": "g", "parameters": {"a": 1}}',
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


class TestLegacyParityContracts:
    """Deliberate, documented behavior changes vs. the legacy parser.

    Both are forced by the streaming contract — output is append-only and
    must match the non-streaming result exactly; see the module docstring
    of vllm/parser/llama_json.py.
    """

    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 64])
    def test_leading_prose_is_content_alongside_call_in_both_modes(
        self, mock_tokenizer, mock_request, chunk_size
    ):
        """Prose before the envelope is content in BOTH modes.

        Legacy dropped it non-streaming (content=None) and, streaming,
        returned the whole output as content with no tool call at all.
        Streaming cannot retract prose already emitted before the "{"
        arrives, so reporting it is the only parity-preserving option.
        """
        text = 'Let me check. {"name": "get_weather", "parameters": {"city": "SF"}}'
        reference = LlamaJsonParser(mock_tokenizer).extract_tool_calls_from_content(
            text, mock_request
        )
        assert reference.tools_called is True
        assert reference.content == "Let me check."
        assert reference.tool_calls[0].function.name == "get_weather"

        parser = LlamaJsonParser(mock_tokenizer)
        content, calls = _accumulate(_stream(parser, mock_request, text, chunk_size))
        # Non-streaming strips content whitespace around tool calls;
        # streaming cannot retract the space it already emitted.
        assert content.rstrip() == reference.content
        assert [c["name"] for c in calls] == ["get_weather"]
        assert [c["args"] for c in calls] == [
            tc.function.arguments for tc in reference.tool_calls
        ]

    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 64])
    @pytest.mark.parametrize(
        "text,expected",
        [
            (
                '{"name": "f", "parameters": {"a": 1}, "arguments": {"b": 2}}',
                '{"a": 1}',
            ),
            (
                '{"name": "f", "arguments": {"b": 2}, "parameters": {"a": 1}}',
                '{"b": 2}',
            ),
        ],
        ids=["parameters-first", "arguments-first"],
    )
    def test_duplicate_alias_first_in_text_wins(
        self, mock_tokenizer, mock_request, text, expected, chunk_size
    ):
        """With both aliases present, the first one in the text wins.

        Legacy preferred "arguments" non-streaming, but its streaming path
        asserted on the duplicate and emitted no arguments at all.  The
        value streams as soon as its key is seen, so preferring a later
        "arguments" would have to retract already-streamed "parameters"
        text; first-in-text-wins is identical at every chunk size and in
        both modes.
        """
        reference = LlamaJsonParser(mock_tokenizer).extract_tool_calls_from_content(
            text, mock_request
        )
        assert [tc.function.arguments for tc in reference.tool_calls] == [expected]

        parser = LlamaJsonParser(mock_tokenizer)
        content, calls = _accumulate(_stream(parser, mock_request, text, chunk_size))
        assert content == ""
        assert [c["args"] for c in calls] == [expected]


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


_STR = {"type": "string"}
_INT = {"type": "integer"}


class TestSchemaCoercionStreamingParity:
    """Schema coercion must not truncate streamed arguments.

    The engine streams the model's verbatim argument text and only coerces
    at flush, re-serialising the whole object; the corrected value then
    stopped being an extension of what had already been streamed, the
    append-only guard dropped it, and the client was left with invalid JSON
    such as ``{"a":"foo","x":``.
    """

    @staticmethod
    def _tools(properties):
        return [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "f",
                    "parameters": {"type": "object", "properties": properties},
                },
            )
        ]

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 7, 17, 100])
    @pytest.mark.parametrize(
        ("properties", "body"),
        [
            ({"a": _STR, "x": _INT}, '{"a":"foo","x":"1"}'),
            ({"x": _INT, "a": _STR}, '{"x":"1","a":"foo"}'),
            ({"x": _INT, "y": _INT}, '{"x": "1", "y": "2"}'),
            ({"b": {"type": "boolean"}, "a": _STR}, '{"b":"true","a":"q"}'),
            ({"z": _STR, "a": _STR}, '{"z":123,"a":"q"}'),
            (
                {"o": {"type": "object", "properties": {"n": _INT}}, "a": _STR},
                '{"o":{"n":"5"},"a":"q"}',
            ),
            (
                {"l": {"type": "array", "items": _INT}, "a": _STR},
                '{"l":["1","2"],"a":"q"}',
            ),
            ({"a": _STR, "x": _INT}, '{"a":"foo","x":2}'),
            ({"a": _STR}, '{"a":"foo","zz":"1"}'),
            ({"a": _STR}, "{}"),
            # An empty schema is still a schema: it types every value as
            # a string, so streaming must splice it like any other one
            # (a truthiness test made it a no-op mid-stream only).
            ({"x": {}}, '{"x":1.5}'),
            ({"x": {}}, '{"x":true}'),
            ({"x": {}}, '{"x":null}'),
            ({"x": {}, "y": {}}, '{"x":"s","y":2}'),
            ({"o": {"type": "object", "properties": {"n": {}}}}, '{"o":{"n":5}}'),
            ({"l": {"type": "array", "items": {}}}, '{"l":[1,2]}'),
            ({"o": {}}, '{"o":{"n":5}}'),
        ],
    )
    def test_streamed_arguments_match_non_streaming(
        self, mock_tokenizer, mock_request, properties, body, chunk_size
    ):
        tools = self._tools(properties)
        mock_request.tools = tools
        text = '{"name": "f", "parameters": ' + body + "}"

        non_streaming = LlamaJsonParser(mock_tokenizer, tools).extract_tool_calls(
            text, mock_request
        )
        expected = non_streaming.tool_calls[0].function.arguments
        json.loads(expected)

        parser = LlamaJsonParser(mock_tokenizer, tools)
        _, calls = _accumulate(_stream(parser, mock_request, text, chunk_size))

        assert calls[0]["args"] == expected
        assert json.loads(calls[0]["args"]) == json.loads(expected)

    @pytest.mark.parametrize("chunk_size", [1, 2, 4, 7, 100])
    def test_coercion_applied_and_json_valid_at_every_chunk_size(
        self, mock_tokenizer, mock_request, chunk_size
    ):
        tools = self._tools({"a": _STR, "x": _INT})
        mock_request.tools = tools
        text = '{"name":"f","parameters":{"a":"foo","x":"1"}}'

        parser = LlamaJsonParser(mock_tokenizer, tools)
        _, calls = _accumulate(_stream(parser, mock_request, text, chunk_size))

        assert json.loads(calls[0]["args"]) == {"a": "foo", "x": 1}

    def test_coercion_keeps_the_model_separators(self, mock_tokenizer):
        tools = self._tools({"a": _STR, "x": _INT})
        parser = LlamaJsonParser(mock_tokenizer, tools)
        assert parser._fix_arg_types('{"a":"foo","x":"1"}', "f") == '{"a":"foo","x":1}'
        assert (
            parser._fix_arg_types('{"a": "foo", "x": "1"}', "f")
            == '{"a": "foo", "x": 1}'
        )


class TestIncrementalArgScanning:
    """Argument scanning carries state across feeds instead of restarting.

    Re-reading the whole accumulated envelope on every streamed chunk made
    a single tool call quadratic in its argument length.  These are shape
    assertions rather than timings: the resumed scan must answer exactly
    what a full rescan answers, and the schema lookup must not repeat per
    chunk.
    """

    @pytest.mark.parametrize(
        "raw",
        [
            '{"a": 1, "b": [2, {"c": "d"}]}',
            '"plain string"',
            '"br{a}ce[s] and \\" escapes \\\\"',
            '{"k": "unbalanced { [ \\" inside"}',
            "12345",
            "true",
            '["a", "b"]',
            '{"a": "\\u0041\\\\", "b": {}}',
            "{}",
        ],
    )
    def test_value_scan_matches_full_rescan_at_every_length(self, raw):
        scan = _ValueScan(0)
        for cut in range(1, len(raw) + 1):
            assert scan.end(raw[:cut]) == _scan_json_value(raw[:cut], 0)

    def test_schema_lookup_is_not_repeated_per_chunk(
        self, mock_tokenizer, mock_request, monkeypatch
    ):
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    },
                },
            )
        ]
        mock_request.tools = tools
        looked_up = []
        real = llama_json.find_tool_properties

        def counting(tools, name):
            looked_up.append(name)
            return real(tools, name)

        monkeypatch.setattr(llama_json, "find_tool_properties", counting)

        text = '{"name": "f", "parameters": {"x": "' + "a" * 200 + '"}}'
        parser = LlamaJsonParser(mock_tokenizer, tools)
        _, calls = _accumulate(_stream_text_only(parser, mock_request, text, 1))

        assert json.loads(calls[0]["args"]) == {"x": "a" * 200}
        # One lookup for the call, not one per streamed character.
        assert looked_up == ["f"]

    def test_splice_keeps_every_unchanged_byte_with_many_edits(self):
        raw = '{"a": 1, "b": 2, "c": "keep", "d": 3}'
        schema = {
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
                "c": {"type": "string"},
                "d": {"type": "string"},
            }
        }
        assert (
            _splice_types(raw, 0, len(raw), schema)
            == '{"a": "1", "b": "2", "c": "keep", "d": "3"}'
        )


class TestRequiredAndNamedToolChoice:
    """Required/named tool choice must be parsed here, not by the shared helpers.

    ``extract_required_tool_call_streaming`` and its named counterpart
    rebuild their state from the cumulative document, but engine-based
    parsers are fed one delta at a time -- ``previous_text`` is always
    empty, so those helpers never see a complete call.  A llama parser that
    advertised ``supports_required_and_named`` therefore streamed nothing at
    all for required choice, and streamed the whole ``{"name": ...,
    "parameters": ...}`` envelope as the arguments for named choice.

    Declaring the flag False sends both choices to this parser instead, as
    the other engine-based parsers do.  Guided decoding is unaffected: the
    tool schema is applied from the request's ``tool_choice``, independent
    of the flag.
    """

    ARRAY = '[{"name": "get_weather", "parameters": {"city": "SF"}}]'
    NATIVE = '{"name": "get_weather", "parameters": {"city": "SF"}}'
    NAMED = {"type": "function", "function": {"name": "get_weather"}}

    def test_flag_holds_without_strict_enforcement(self, monkeypatch):
        """The regression guard for this whole class.

        ``ToolParser.__init_subclass__`` forces the flag False only while
        ``VLLM_ENFORCE_STRICT_TOOL_CALLING`` is set, and that is read once,
        at class-creation time.  With strict enforcement off the class must
        still declare it in its own body -- otherwise it inherits True and
        required/named tool choice routes to the generic helpers, which is
        the broken configuration.  Re-import the module to see what a
        strict-disabled server actually gets.
        """
        import importlib

        import vllm.tool_parsers.llama_tool_parser as module

        monkeypatch.setenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", "0")
        try:
            reloaded = importlib.reload(module)
            assert reloaded.Llama3JsonToolParser.supports_required_and_named is False
        finally:
            monkeypatch.undo()
            importlib.reload(module)

    @staticmethod
    def _request(tool_choice):
        return ChatCompletionRequest(
            model="llama",
            messages=[{"role": "user", "content": "weather in SF?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            tool_choice=tool_choice,
        )

    @staticmethod
    def _stream(parser, request, text, chunk_size):
        names: list[str] = []
        args: dict[int, str] = {}
        for start in range(0, len(text), chunk_size):
            delta_text = text[start : start + chunk_size]
            delta = parser.parse_delta(
                delta_text,
                [ord(c) for c in delta_text],
                request,
                prompt_token_ids=[1] if start == 0 else None,
                finished=start + chunk_size >= len(text),
            )
            for tool_call in (delta.tool_calls if delta else None) or []:
                if tool_call.function and tool_call.function.name:
                    names.append(tool_call.function.name)
                if tool_call.function and tool_call.function.arguments:
                    args[tool_call.index] = (
                        args.get(tool_call.index, "") + tool_call.function.arguments
                    )
        return names, args

    @pytest.fixture
    def parser(self, mock_tokenizer):
        parser_cls = ParserManager.get_parser(
            tool_parser_name="llama3_json",
            reasoning_parser_name=None,
            enable_auto_tools=True,
        )
        assert parser_cls is not None
        return parser_cls(mock_tokenizer)

    @pytest.mark.parametrize("chunk_size", [1, 7, 10_000])
    @pytest.mark.parametrize("body", ["ARRAY", "NATIVE"])
    def test_required_streams_the_call(self, parser, chunk_size, body):
        """Required choice streams a call for both wire shapes.

        Without a llama structural tag the model is guided to the
        ``[{...}]`` array schema; with one it emits the bare envelope.
        """
        text = getattr(self, body)
        names, args = self._stream(parser, self._request("required"), text, chunk_size)

        assert names == ["get_weather"]
        assert [json.loads(a) for a in args.values()] == [{"city": "SF"}]

    @pytest.mark.parametrize("chunk_size", [1, 7, 10_000])
    def test_named_streams_only_the_arguments(self, parser, chunk_size):
        """Named choice must stream the parameters, not the whole envelope."""
        names, args = self._stream(
            parser, self._request(self.NAMED), self.NATIVE, chunk_size
        )

        assert names == ["get_weather"]
        assert [json.loads(a) for a in args.values()] == [{"city": "SF"}]

    @pytest.mark.parametrize("tool_choice", ["required", "named"])
    @pytest.mark.parametrize("body", ["ARRAY", "NATIVE"])
    def test_non_streaming_matches_streaming(self, parser, tool_choice, body):
        choice = self.NAMED if tool_choice == "named" else tool_choice
        info = parser.extract_tool_calls(getattr(self, body), self._request(choice))

        assert info.tools_called
        assert [tc.function.name for tc in info.tool_calls] == ["get_weather"]
        assert [json.loads(tc.function.arguments) for tc in info.tool_calls] == [
            {"city": "SF"}
        ]

    @pytest.mark.parametrize("chunk_size", [1, 7, 10_000])
    @pytest.mark.parametrize("tool_choice", ["required", "named"])
    def test_forced_choice_emits_no_content(self, parser, chunk_size, tool_choice):
        """The array schema's brackets must not leak as content.

        Required/named apply a ``[{...}, {...}]`` JSON schema, whose opening
        bracket reaches the parser before any call completes -- so the
        "drop content once a call completed" rule does not cover it, and it
        surfaced as ``content="["``.  Gemma4 hit the same class of leak
        (vllm-project/vllm#45795), where the whole forced JSON leaked.
        """
        choice = self.NAMED if tool_choice == "named" else tool_choice
        content_parts: list[str] = []
        request = self._request(choice)
        for start in range(0, len(self.ARRAY), chunk_size):
            delta_text = self.ARRAY[start : start + chunk_size]
            delta = parser.parse_delta(
                delta_text,
                [ord(c) for c in delta_text],
                request,
                prompt_token_ids=[1] if start == 0 else None,
                finished=start + chunk_size >= len(self.ARRAY),
            )
            if delta and delta.content:
                content_parts.append(delta.content)

        assert "".join(content_parts) == ""

    @pytest.mark.parametrize("tool_choice", ["required", "named"])
    def test_forced_choice_emits_no_content_non_streaming(self, parser, tool_choice):
        choice = self.NAMED if tool_choice == "named" else tool_choice
        _, content = parser._extract_tool_calls(
            self.ARRAY, self._request(choice), enable_auto_tools=True
        )

        assert not content

    def test_auto_choice_still_keeps_leading_prose(self, parser):
        """Dropping content is scoped to forced choice, not to every request."""
        text = (
            'Sure, here you go. {"name": "get_weather", "parameters": {"city": "SF"}}'
        )
        calls, content = parser._extract_tool_calls(
            text, self._request("auto"), enable_auto_tools=True
        )

        assert [c.name for c in calls or []] == ["get_weather"]
        assert content and "Sure, here you go." in content


class TestNamedChoiceWithoutStructuralTag:
    """sc-01: a named choice must produce a call with strict calling off.

    Without ``VLLM_ENFORCE_STRICT_TOOL_CALLING`` there is no llama structural
    tag, so ``adjust_request()`` constrains the model to the selected
    function's ``parameters`` alone.  The model then emits ``{"city": "SF"}``
    and never writes the ``{"name": ..., "parameters": ...}`` envelope this
    parser keys on, so the parameters come back as content and the call is
    lost.  Legacy synthesized the name from ``tool_choice``.

    Chat completions carry that constraint in ``structured_outputs.json``
    and the Responses API in ``text.format``, so both are driven through
    ``adjust_request()`` and assert the field it actually wrote first.
    """

    PARAMETERS = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
        "required": ["city", "unit"],
    }
    BARE_PARAMETERS = '{"city":"SF","unit":"C"}'

    @pytest.fixture
    def parser_cls(self, monkeypatch):
        monkeypatch.setenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", "0")
        cls = ParserManager.get_parser(
            tool_parser_name="llama3_json",
            reasoning_parser_name=None,
            enable_auto_tools=True,
        )
        assert cls is not None
        return cls

    def _request(self, request_kind):
        if request_kind == "chat":
            return ChatCompletionRequest(
                model="llama",
                messages=[{"role": "user", "content": "weather in SF?"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": self.PARAMETERS,
                        },
                    }
                ],
                tool_choice={
                    "type": "function",
                    "function": {"name": "get_weather"},
                },
            )

        from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

        return ResponsesRequest(
            model="llama",
            input="weather in SF?",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "look up the weather",
                    "parameters": self.PARAMETERS,
                    "strict": True,
                }
            ],
            tool_choice={"type": "function", "name": "get_weather"},
        )

    def _adjusted(self, parser, request_kind):
        request = parser.adjust_request(self._request(request_kind))
        # Precondition: this is the constraint that makes the model emit
        # bare parameters at all.  Assert it so a change in the request
        # plumbing fails here and says so, instead of leaving the rest of
        # the test quietly exercising nothing.
        if request_kind == "chat":
            assert request.structured_outputs is not None
            assert request.structured_outputs.json == self.PARAMETERS
        else:
            assert request.text is not None
            assert request.text.format.schema_ == self.PARAMETERS
        return request

    @pytest.mark.parametrize("request_kind", ["chat", "responses"])
    def test_named_choice_returns_a_call_not_bare_parameters(
        self, parser_cls, mock_tokenizer, request_kind
    ):
        parser = parser_cls(mock_tokenizer)
        request = self._adjusted(parser, request_kind)

        _, content, tool_calls = parser.parse(
            self.BARE_PARAMETERS, request, enable_auto_tools=True
        )

        assert [
            (call.name, json.loads(call.arguments)) for call in tool_calls or []
        ] == [("get_weather", {"city": "SF", "unit": "C"})]
        assert content is None

    @pytest.mark.parametrize("chunk_size", [1, 4])
    @pytest.mark.parametrize("request_kind", ["chat", "responses"])
    def test_named_choice_streams_before_the_last_chunk(
        self, parser_cls, mock_tokenizer, request_kind, chunk_size
    ):
        """The name and arguments must not be buffered until EOF.

        A parser that holds every tool-call event until the envelope
        balances still accumulates to the right answer, so an assertion on
        the folded result cannot tell the two apart.  Only deltas emitted
        while ``finished`` is False count here.
        """
        parser = parser_cls(mock_tokenizer)
        request = self._adjusted(parser, request_kind)
        early = []

        for start in range(0, len(self.BARE_PARAMETERS), chunk_size):
            delta_text = self.BARE_PARAMETERS[start : start + chunk_size]
            finished = start + chunk_size >= len(self.BARE_PARAMETERS)
            delta = parser.parse_delta(
                delta_text,
                [ord(c) for c in delta_text],
                request,
                prompt_token_ids=[1] if start == 0 else None,
                finished=finished,
            )
            if not finished and delta and delta.tool_calls:
                early.extend(delta.tool_calls)

        assert any(
            call.function and call.function.name == "get_weather" for call in early
        )
        assert any(call.function and call.function.arguments for call in early)


class TestSchemaCoercionPreservesValidValues:
    """sc-02: coercion repairs a value written in the wrong JSON type.

    A value already valid for its schema must come back exactly as the
    model wrote it.  ``extract_types_from_schema`` answers ``["string"]``
    when it can determine nothing, which is right for reading a value out
    of raw text and wrong as a constraint: it made an empty schema, a
    ``const`` and a ``$ref`` all retype an integer as a string.  Where
    types *were* known, an already-valid value still went through the
    coercion machinery and came back altered.
    """

    @staticmethod
    def _tools(property_schema, definitions=None):
        parameters = {"type": "object", "properties": {"x": property_schema}}
        if definitions is not None:
            parameters["$defs"] = definitions
        return [
            ChatCompletionToolsParam(
                type="function",
                function={"name": "f", "parameters": parameters},
            )
        ]

    @classmethod
    def _arguments(
        cls, mock_tokenizer, mock_request, property_schema, literal, definitions=None
    ):
        tools = cls._tools(property_schema, definitions)
        mock_request.tools = tools
        arguments = f'{{"x":{literal}}}'
        text = f'{{"name":"f","parameters":{arguments}}}'
        result = LlamaJsonParser(mock_tokenizer, tools).extract_tool_calls(
            text, mock_request
        )
        return result.tool_calls[0].function.arguments

    @pytest.mark.parametrize(
        ("property_schema", "literal", "definitions"),
        [
            ({"type": "number"}, "9007199254740993", None),
            ({"type": ["string", "integer"]}, '"007"', None),
            ({"anyOf": [{"type": "string"}, {"type": "boolean"}]}, '"false"', None),
            ({"oneOf": [{"type": "string"}, {"type": "integer"}]}, '"007"', None),
            ({"type": ["string", "null"]}, '"null"', None),
            ({}, "7", None),
            ({"const": 7}, "7", None),
            (
                {"$ref": "#/$defs/integer_value"},
                "7",
                {"integer_value": {"type": "integer"}},
            ),
        ],
        ids=[
            "large-integer",
            "type-union-string",
            "anyof-string",
            "oneof-string",
            "nullable-union-string",
            "empty-schema",
            "const",
            "ref",
        ],
    )
    def test_already_valid_values_are_returned_unchanged(
        self, mock_tokenizer, mock_request, property_schema, literal, definitions
    ):
        assert (
            self._arguments(
                mock_tokenizer, mock_request, property_schema, literal, definitions
            )
            == f'{{"x":{literal}}}'
        )

    def test_allof_is_a_conjunction_not_a_union(self, mock_tokenizer, mock_request):
        """``allOf`` narrows the permitted types; it does not widen them.

        Both halves live in one test on purpose.  The first fails against a
        parser with no validity check at all, the second against one that
        unions the ``allOf`` branches -- a value permitted by some branch
        but not by every branch is not already valid, and skipping coercion
        for it reports a type the schema forbids.
        """
        keep = {
            "allOf": [
                {"type": ["string", "integer"]},
                {"type": ["string", "boolean"]},
            ]
        }
        assert (
            self._arguments(mock_tokenizer, mock_request, keep, '"007"')
            == '{"x":"007"}'
        )

        coerce = {
            "allOf": [
                {"type": ["string", "boolean"]},
                {"type": ["boolean", "integer"]},
            ]
        }
        assert (
            self._arguments(mock_tokenizer, mock_request, coerce, '"true"')
            == '{"x":true}'
        )

    @pytest.mark.parametrize(
        ("property_schema", "literal"),
        [
            (
                {"type": "object", "properties": {"n": {"type": "number"}}},
                '{"n":9007199254740993}',
            ),
            ({"type": "array", "items": {"type": "number"}}, "[9007199254740993]"),
        ],
        ids=["nested-object", "array-item"],
    )
    def test_valid_values_survive_inside_containers(
        self, mock_tokenizer, mock_request, property_schema, literal
    ):
        """Coercion recurses through ``properties`` and ``items``.

        The scalar cases above only prove the top level is left alone; a
        validity check placed at the wrong depth still rewrites a large
        integer nested one level down.
        """
        assert (
            self._arguments(mock_tokenizer, mock_request, property_schema, literal)
            == f'{{"x":{literal}}}'
        )

    @pytest.mark.parametrize("chunk_size", [1, 4])
    def test_streaming_preserves_already_valid_values(
        self, mock_tokenizer, mock_request, chunk_size
    ):
        """The streaming path must preserve them too, and agree exactly.

        Coercion is applied twice by different machinery -- spliced into
        the settled prefix while streaming, and over the whole object at
        flush -- so a fix applied to only one path yields a client that
        sees a different number depending on whether it asked for a stream.
        """
        literal = "9007199254740993"
        property_schema = {"type": "number"}
        tools = self._tools(property_schema)
        mock_request.tools = tools
        arguments = f'{{"x":{literal}}}'
        text = f'{{"name":"f","parameters":{arguments}}}'

        reference = LlamaJsonParser(mock_tokenizer, tools).extract_tool_calls(
            text, mock_request
        )
        parser = LlamaJsonParser(mock_tokenizer, tools)
        _, calls = _accumulate(_stream(parser, mock_request, text, chunk_size))

        assert [c["args"] for c in calls] == [
            tc.function.arguments for tc in reference.tool_calls
        ]
        assert [c["args"] for c in calls] == [arguments]


class TestPhantomRetractionIsLinear:
    """Prose JSON must not cost time quadratic in how much of it there is.

    A ``{...}`` with no top-level ``"name"`` is not a tool call, so its
    events are retracted and the text restored as content.  Rebuilding the
    whole output list on each retraction made a document of N such objects
    cost O(N^2): a model asked for JSON lines while ``tools`` was set could
    burn seconds of CPU in one parse, with no tool call anywhere in it.
    """

    @staticmethod
    def _document(count: int) -> str:
        return "\n".join(f'{{"id": {i}, "value": "row{i}"}}' for i in range(count))

    def _request(self):
        return ChatCompletionRequest(
            model="llama",
            messages=[{"role": "user", "content": "one json object per line"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            tool_choice="auto",
        )

    def test_retraction_examines_only_its_own_call(self, mock_tokenizer, monkeypatch):
        """Retraction work must grow with the document, not with its square.

        The offset handed to _retract_call is not evidence on its own: a body
        that ignores it and rebuilds the whole list receives the same offset
        and restores the same events, so measuring the argument proves
        nothing.  Count what the body actually reads instead, by handing it a
        list that records every element each read touches.
        """

        class _CountingList(list):
            def __init__(self, items):
                super().__init__(items)
                self.touched = 0

            def __iter__(self):
                self.touched += len(self)
                return super().__iter__()

            def __getitem__(self, index):
                if isinstance(index, slice):
                    self.touched += len(range(*index.indices(len(self))))
                else:
                    self.touched += 1
                return super().__getitem__(index)

        real = LlamaJsonParser._retract_call
        touched = 0
        retractions = 0

        def measuring(out, start, dense_idx):
            nonlocal touched, retractions
            probe = _CountingList(out)
            real(probe, start, dense_idx)
            # Read the count before the write-back: assigning out[:] = probe
            # iterates the probe and would charge the harness's own copy to
            # the callee.
            touched += probe.touched
            out[:] = list(list.__iter__(probe))
            retractions += 1

        monkeypatch.setattr(LlamaJsonParser, "_retract_call", staticmethod(measuring))

        work = []
        for count in (200, 400):
            touched = retractions = 0
            parser = LlamaJsonParser(mock_tokenizer)
            parser.extract_tool_calls(self._document(count), self._request())
            # Liveness: an implementation that never retracts would make the
            # growth assertion vacuous.
            assert retractions == count
            work.append(touched)

        # Linear doubles, quadratic quadruples.  The slack absorbs the couple
        # of events each call contributes without admitting a rate change.
        assert work[1] <= 3 * work[0]

    @pytest.mark.parametrize("count", [1, 5, 50])
    def test_prose_json_is_returned_as_content(self, mock_tokenizer, count):
        """The retraction must still restore every object, in order."""
        document = self._document(count)
        result = LlamaJsonParser(mock_tokenizer).extract_tool_calls(
            document, self._request()
        )

        assert not result.tools_called
        assert (result.content or "").count('"id"') == count


class TestNamedChoiceOnParameterlessTool:
    """A tool declaring no ``parameters`` must still produce a call.

    ``get_json_schema_from_tools`` returns the selected function's
    ``parameters``, which is ``None`` for such a tool, so no guided-decoding
    schema was applied.  With no llama structural tag either
    (``VLLM_ENFORCE_STRICT_TOOL_CALLING=0``) nothing constrained the model
    and nothing told the parser the output was bare parameters, so it
    demanded a full envelope and returned no tool call at all -- for a
    request that explicitly named the tool it wanted.
    """

    @pytest.fixture
    def parser_cls(self, monkeypatch):
        monkeypatch.setenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", "0")
        cls = ParserManager.get_parser(
            tool_parser_name="llama3_json",
            reasoning_parser_name=None,
            enable_auto_tools=True,
        )
        assert cls is not None
        return cls

    @staticmethod
    def _request():
        return ChatCompletionRequest(
            model="llama",
            messages=[{"role": "user", "content": "ping the server"}],
            # No "parameters" key at all -- the shape that got no schema.
            tools=[{"type": "function", "function": {"name": "ping"}}],
            tool_choice={"type": "function", "function": {"name": "ping"}},
        )

    def test_a_schema_is_applied_so_the_output_shape_is_known(
        self, parser_cls, mock_tokenizer
    ):
        parser = parser_cls(mock_tokenizer)
        request = parser.adjust_request(self._request())

        structured = getattr(request, "structured_outputs", None)
        assert getattr(structured, "json", None) is not None

    def test_bare_empty_object_is_a_call(self, parser_cls, mock_tokenizer):
        parser = parser_cls(mock_tokenizer)
        request = parser.adjust_request(self._request())

        calls, content = parser._extract_tool_calls(
            "{}", request, enable_auto_tools=True
        )

        assert [(c.name, json.loads(c.arguments)) for c in calls or []] == [
            ("ping", {})
        ]
        assert not content

    @pytest.mark.parametrize("chunk_size", [1, 2, 10_000])
    def test_bare_empty_object_streams_a_call(
        self, parser_cls, mock_tokenizer, chunk_size
    ):
        parser = parser_cls(mock_tokenizer)
        request = parser.adjust_request(self._request())
        body = "{}"

        names: list[str] = []
        args: dict[int, str] = {}
        for start in range(0, len(body), chunk_size):
            delta_text = body[start : start + chunk_size]
            delta = parser.parse_delta(
                delta_text,
                [ord(c) for c in delta_text],
                request,
                prompt_token_ids=[1] if start == 0 else None,
                finished=start + chunk_size >= len(body),
            )
            for tool_call in (delta.tool_calls if delta else None) or []:
                if tool_call.function and tool_call.function.name:
                    names.append(tool_call.function.name)
                if tool_call.function and tool_call.function.arguments:
                    args[tool_call.index] = (
                        args.get(tool_call.index, "") + tool_call.function.arguments
                    )

        assert names == ["ping"]
        assert [json.loads(a) for a in args.values()] == [{}]


class TestLlama4PythonMarkers:
    """Llama 4 wraps tool calls in ``<|python_start|>``/``<|python_end|>``.

    The engine sets ``skip_special_tokens=False`` so the detokenizer no
    longer strips special tokens, and the engine's own drop machinery only
    covers ``tokenizer.all_special_tokens`` -- on a real Llama tokenizer
    that is just begin_of_text and eot_id.  Without explicit terminals the
    wrappers reached the client as content, next to a correctly parsed call.
    The sibling llama4_pythonic parser strips them for the same reason.
    """

    VOCAB = {
        PYTHON_START: 200000,
        PYTHON_END: 200001,
        "<|eot_id|>": 128009,
    }
    CALL = '{"name": "get_weather", "parameters": {"city": "SF"}}'

    @pytest.fixture
    def tokenizer(self):
        # Deliberately no <|python_tag|>: Llama 4 tokenizers lack it, which
        # is the case the parser must not depend on.  The markers are in the
        # vocab but NOT in all_special_tokens, which is how a real Llama
        # tokenizer reports them -- on Llama-3.1-8B-Instruct
        # all_special_tokens is exactly [begin_of_text, eot_id] while
        # <|python_tag|> is in the vocab.  Marking them special instead would
        # let the engine drop them for free and make these tests vacuous.
        return make_mock_tokenizer(self.VOCAB, special_tokens=["<|eot_id|>"])

    @classmethod
    def _tokenize(cls, text: str) -> list[tuple[int, str]]:
        markers = sorted(cls.VOCAB, key=len, reverse=True)
        tokens: list[tuple[int, str]] = []
        i = 0
        while i < len(text):
            for marker in markers:
                if text.startswith(marker, i):
                    tokens.append((cls.VOCAB[marker], marker))
                    i += len(marker)
                    break
            else:
                tokens.append((ord(text[i]), text[i]))
                i += 1
        return tokens

    @pytest.mark.parametrize("chunk_size", [1, 3, 10_000])
    @pytest.mark.parametrize(
        "body",
        ["<|python_start|>{call}<|python_end|>", "<|python_start|>{call}"],
        ids=["wrapped", "unterminated-wrapper"],
    )
    def test_markers_never_reach_content_as_tokens(
        self, tokenizer, mock_request, chunk_size, body
    ):
        text = body.format(call=self.CALL)
        parser = LlamaJsonParser(tokenizer)

        content, calls = _accumulate(
            _stream_tokens(parser, mock_request, self._tokenize(text), chunk_size)
        )

        assert [c["name"] for c in calls] == ["get_weather"]
        assert json.loads(calls[0]["args"]) == {"city": "SF"}
        assert "<|python_" not in content

    @pytest.mark.parametrize("chunk_size", [1, 3, 10_000])
    def test_markers_never_reach_content_as_text(
        self, tokenizer, mock_request, chunk_size
    ):
        """Markers split across text chunks must still be consumed."""
        text = f"<|python_start|>{self.CALL}<|python_end|>"
        parser = LlamaJsonParser(tokenizer)

        content, calls = _accumulate(
            _stream_text_only(parser, mock_request, text, chunk_size)
        )

        assert [c["name"] for c in calls] == ["get_weather"]
        assert "<|python_" not in content

    def test_wrapped_prose_keeps_the_prose_and_drops_the_markers(
        self, tokenizer, mock_request
    ):
        """No call here -- the text survives, the wrappers do not."""
        text = "<|python_start|>no call here<|python_end|>"
        parser = LlamaJsonParser(tokenizer)

        content, calls = _accumulate(
            _stream_tokens(parser, mock_request, self._tokenize(text), 3)
        )

        assert calls == []
        assert content == "no call here"

    def test_non_streaming_agrees(self, tokenizer, mock_request):
        text = f"<|python_start|>{self.CALL}<|python_end|>"
        result = LlamaJsonParser(tokenizer).extract_tool_calls(text, mock_request)

        assert [tc.function.name for tc in result.tool_calls] == ["get_weather"]
        assert "<|python_" not in (result.content or "")

    @pytest.mark.parametrize(
        "text",
        [
            "<|python_start|>{}<|python_end|>".format('{"items": [1, 2]}'),
            "<|python_tag|>print('hello')",
        ],
        ids=["wrapped-prose-json", "ipython-content"],
    )
    def test_the_parser_cleans_markers_but_the_serving_path_still_leaks(
        self, tokenizer, mock_request, text
    ):
        """Where the cleaning stops, and why.

        ``LlamaJsonParser.extract_tool_calls`` removes the markers, but it
        sits *below* the delegating layer, and that layer returns the raw
        text when no call was promoted.  So the parser is clean and the
        client still receives the markers.

        Fixing that means changing ``DelegatingParser._extract_tool_calls``
        for every engine-based parser, which is
        vllm-project/vllm#47562's job, not this migration's -- carrying it
        here silently deletes reasoning markup for nine other parsers when
        no ``--reasoning-parser`` is configured.

        Asserting both halves rather than only the parser's: a bare
        ``"<|python_" not in content`` passes when the layer drops the
        content entirely, i.e. it is satisfied by the regression it is
        supposed to catch.
        """
        parser_cls = ParserManager.get_parser(
            tool_parser_name="llama3_json",
            reasoning_parser_name=None,
            enable_auto_tools=True,
        )
        parser = parser_cls(tokenizer)

        cleaned = parser.extract_tool_calls(text, mock_request).content
        _, served = parser._extract_tool_calls(
            text, mock_request, enable_auto_tools=True
        )

        assert cleaned and "<|python_" not in cleaned
        assert served == text


class TestServingPathMarkerParity:
    """``<|python_tag|>`` survives on the non-streaming route. Known gap.

    ``ParserEngine.adjust_request`` sets ``skip_special_tokens=False``, so the
    tag reaches the parser verbatim.  Streaming consumes it via the PYTHON_TAG
    terminal, but when no call is promoted the serving layer's non-streaming
    route returns the *raw* text, leaking the 14-character tag to the client
    -- a divergence this migration introduces and released vLLM does not have.

    The fix belongs in ``DelegatingParser._extract_tool_calls`` and is
    vllm-project/vllm#47562's, not this migration's: it changes non-streaming
    content for all twelve engine-based parsers, and as written it also deletes
    reasoning markup for nine of them when no ``--reasoning-parser`` is
    configured.  Carrying it here would make a Llama migration silently rewrite
    other parsers' output.

    So this class pins the divergence rather than the fix.  When #47562 (or a
    narrowed version of it) lands, these assertions fail and say so.

    Both routes are driven through ``DelegatingParser`` (``parse`` /
    ``parse_delta``), not through ``LlamaJsonParser.extract_tool_calls``: the
    leak lives in the delegating layer, so the parser's own extraction never
    sees it.
    """

    BODIES = {
        # max_tokens cut the envelope mid-key: no call is promoted, so the
        # whole raw string used to come back as content.
        "truncated-call": (
            PYTHON_TAG + '{"name": "get_weather", "para',
            '{"name": "get_weather", "para',
        ),
        # JSON-shaped output that is not a call at all.
        "prose-json": (
            PYTHON_TAG + '{"items": [{"id": 1, "name": "x"}]}',
            '{"items": [{"id": 1, "name": "x"}]}',
        ),
    }

    @pytest.fixture
    def make_parser(self):
        cls = ParserManager.get_parser(
            tool_parser_name="llama3_json",
            reasoning_parser_name=None,
            enable_auto_tools=True,
        )
        assert cls is not None
        # <|python_tag|> is in the vocab but NOT in all_special_tokens, which
        # is how Llama-3.1 reports it.  Marking it special would let the
        # engine's drop machinery consume it for free, making these vacuous.
        return lambda: cls(
            make_mock_tokenizer(_LLAMA_VOCAB, special_tokens=["<|eot_id|>"])
        )

    @staticmethod
    def _request():
        return ChatCompletionRequest(
            model="llama",
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                ChatCompletionToolsParam(
                    type="function",
                    function={
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                )
            ],
            tool_choice="auto",
        )

    @staticmethod
    def _stream(parser, request, text: str, chunk_size: int) -> str:
        tokens = _tokenize(text)
        parts: list[str] = []
        for start in range(0, len(tokens), chunk_size):
            batch = tokens[start : start + chunk_size]
            delta = parser.parse_delta(
                "".join(t for _, t in batch),
                [tid for tid, _ in batch],
                request,
                prompt_token_ids=[1] if start == 0 else None,
                finished=start + chunk_size >= len(tokens),
            )
            assert not (delta.tool_calls if delta else None)
            if delta and delta.content:
                parts.append(delta.content)
        return "".join(parts)

    @pytest.mark.parametrize("chunk_size", [1, 10_000])
    @pytest.mark.parametrize("case", list(BODIES))
    def test_streaming_drops_the_tag_and_non_streaming_does_not(
        self, make_parser, case, chunk_size
    ):
        text, expected = self.BODIES[case]

        _, content, tool_calls = make_parser().parse(
            text, self._request(), enable_auto_tools=True
        )
        streamed = self._stream(make_parser(), self._request(), text, chunk_size)

        assert not tool_calls
        # Streaming consumes the tag via the PYTHON_TAG terminal, at every
        # chunk size.  This half must not regress.
        assert streamed == expected
        # Non-streaming does not, because the fix lives in the delegating
        # layer rather than in this parser, and that hunk is deliberately
        # left to vllm-project/vllm#47562 rather than carried here (it
        # changes non-streaming content for every engine-based parser).
        # Pinned so the divergence is a stated limitation rather than a
        # surprise; when #47562 lands this assertion is what says so.
        assert content == text
        assert content != streamed


class TestArgumentsStreamIncrementally:
    """Arguments must reach the client as they are produced, not at the end.

    The whole point of the splice machinery (_compute_arg_delta,
    _stable_arg_prefix, _ArgScan) is that a client rendering a tool call
    sees its arguments grow.  Every other streaming test folds the deltas
    together before asserting, so a change that buffered the whole call
    until it closed -- turning streaming into non-streaming -- would leave
    the suite green.

    The request carries a tool schema deliberately.  Without one,
    _compute_arg_delta takes the schema-less ``_safe_arg_prefix`` branch,
    which production tool calling never reaches, and the splice machinery
    this class names is not exercised at all.
    """

    TOOLS = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"body": {"type": "string"}},
                },
            },
        )
    ]

    @staticmethod
    def _text(length: int) -> str:
        return '{"name": "f", "parameters": {"body": "' + "x" * length + '"}}'

    def _parser(self, mock_tokenizer):
        return LlamaJsonParser(mock_tokenizer, self.TOOLS)

    def _request(self, mock_request):
        mock_request.tools = self.TOOLS
        return mock_request

    def _arg_deltas(self, parser, request, text, chunk_size):
        """Argument fragments, and how many arrived before the last feed."""
        deltas: list[str] = []
        before_final = 0
        results = _stream_text_only(parser, request, text, chunk_size)
        for position, delta in enumerate(results):
            if delta is None:
                continue
            for tool_call in delta.tool_calls or []:
                if tool_call.function and tool_call.function.arguments:
                    deltas.append(tool_call.function.arguments)
                    if position < len(results) - 1:
                        before_final += 1
        return deltas, before_final

    def test_arguments_arrive_in_bounded_pieces(self, mock_tokenizer, mock_request):
        """No delta may carry much more than one feed's worth of text.

        Counting deltas is not enough on its own: an implementation that
        withholds a string value until it closes still emits a couple of
        deltas, so only a bound on their size separates streaming from a
        blob at the end.
        """
        chunk_size = 8
        value_length = 200
        deltas, before_final = self._arg_deltas(
            self._parser(mock_tokenizer),
            self._request(mock_request),
            self._text(value_length),
            chunk_size,
        )

        assert json.loads("".join(deltas)) == {"body": "x" * value_length}
        assert before_final > 1
        assert max(len(d) for d in deltas) <= 4 * chunk_size

    def test_more_feeds_produce_more_argument_deltas(
        self, mock_tokenizer, mock_request
    ):
        """Halving the chunk size must not collapse to a single delta."""
        text = self._text(240)

        coarse, _ = self._arg_deltas(
            self._parser(mock_tokenizer), self._request(mock_request), text, 64
        )
        fine, _ = self._arg_deltas(
            self._parser(mock_tokenizer), self._request(mock_request), text, 8
        )

        assert json.loads("".join(coarse)) == json.loads("".join(fine))
        assert len(fine) > len(coarse)

    def test_the_schema_path_is_the_one_under_test(self, mock_tokenizer, mock_request):
        """Guard the guard: this class must not drift back to the
        schema-less branch, where none of the above proves anything."""
        seen = 0
        real = LlamaJsonParser._stable_arg_prefix

        def counting(self_, *args, **kwargs):
            nonlocal seen
            seen += 1
            return real(self_, *args, **kwargs)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(LlamaJsonParser, "_stable_arg_prefix", counting)
        try:
            self._arg_deltas(
                self._parser(mock_tokenizer),
                self._request(mock_request),
                self._text(64),
                8,
            )
        finally:
            monkeypatch.undo()

        assert seen > 0


class TestBareArgumentsRepair:
    """Truncated bare parameters must still be valid JSON.

    A named choice constrained to bare parameters carries no envelope, so
    _llama_bare_arg_converter is what closes an unfinished object. If
    generation stops mid-string the client must still receive parseable
    arguments rather than '{"city":"S'.
    """

    @pytest.fixture
    def parser_cls(self, monkeypatch):
        monkeypatch.setenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", "0")
        cls = ParserManager.get_parser(
            tool_parser_name="llama3_json",
            reasoning_parser_name=None,
            enable_auto_tools=True,
        )
        assert cls is not None
        return cls

    @staticmethod
    def _request():
        return ChatCompletionRequest(
            model="llama",
            messages=[{"role": "user", "content": "weather in SF?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "unit": {"type": "string"},
                            },
                        },
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )

    TRUNCATED = [
        '{"city":"S',
        '{"city":"SF","unit":',
        '{"city":"SF", "unit": tr',
        '{"city":"SF"',
        "{",
    ]

    @pytest.mark.parametrize("body", TRUNCATED)
    def test_non_streaming_arguments_are_parseable(
        self, parser_cls, mock_tokenizer, body
    ):
        parser = parser_cls(mock_tokenizer)
        request = parser.adjust_request(self._request())

        calls, _ = parser._extract_tool_calls(body, request, enable_auto_tools=True)

        assert [c.name for c in calls or []] == ["get_weather"]
        json.loads(calls[0].arguments)

    @pytest.mark.parametrize("body", TRUNCATED)
    @pytest.mark.parametrize("chunk_size", [1, 4])
    def test_streamed_arguments_are_parseable(
        self, parser_cls, mock_tokenizer, body, chunk_size
    ):
        parser = parser_cls(mock_tokenizer)
        request = parser.adjust_request(self._request())

        args: dict[int, str] = {}
        for start in range(0, len(body), chunk_size):
            delta_text = body[start : start + chunk_size]
            delta = parser.parse_delta(
                delta_text,
                [ord(c) for c in delta_text],
                request,
                prompt_token_ids=[1] if start == 0 else None,
                finished=start + chunk_size >= len(body),
            )
            for tool_call in (delta.tool_calls if delta else None) or []:
                if tool_call.function and tool_call.function.arguments:
                    args[tool_call.index] = (
                        args.get(tool_call.index, "") + tool_call.function.arguments
                    )

        for streamed in args.values():
            json.loads(streamed)


class TestBareArgsArmingRespectsTheSchema:
    """Bare-arguments mode may only arm when the schema forbids an envelope.

    A named tool choice installs the selected function's ``parameters`` as
    the guided-decoding schema.  When that schema declares no properties --
    ``{"type": "object"}`` or ``{}`` -- it does not forbid
    ``{"name": ..., "parameters": ...}``, so an effectively unconstrained
    model writes the envelope the chat template asks for.  Reading the whole
    object as the arguments then hands the tool executor the envelope, which
    is what upstream does today for these schemas.
    """

    NAMED = {"type": "function", "function": {"name": "run_python"}}
    ENVELOPE = '{"name": "run_python", "parameters": {"code": "print(1)"}}'
    BARE = '{"code": "print(1)"}'

    @pytest.fixture
    def parser_cls(self, monkeypatch):
        monkeypatch.setenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", "0")
        cls = ParserManager.get_parser(
            tool_parser_name="llama3_json",
            reasoning_parser_name=None,
            enable_auto_tools=True,
        )
        assert cls is not None
        return cls

    def _request(self, parameters):
        function = {"name": "run_python", "description": "run python"}
        if parameters is not None:
            function["parameters"] = parameters
        return ChatCompletionRequest(
            model="llama",
            messages=[{"role": "user", "content": "run python that prints 1"}],
            tools=[{"type": "function", "function": function}],
            tool_choice=self.NAMED,
        )

    # Schemas that do NOT constrain the object's keys. Guided decoding
    # permits the envelope, so the model writes one.
    UNCONSTRAINED = [
        pytest.param({"type": "object"}, id="free-form-object"),
        pytest.param({}, id="empty-schema"),
    ]

    @pytest.mark.parametrize("parameters", UNCONSTRAINED)
    def test_envelope_is_carved_when_the_schema_permits_one(
        self, parser_cls, mock_tokenizer, parameters
    ):
        parser = parser_cls(mock_tokenizer)
        request = parser.adjust_request(self._request(parameters))

        calls, _ = parser._extract_tool_calls(
            self.ENVELOPE, request, enable_auto_tools=True
        )

        assert [c.name for c in calls or []] == ["run_python"]
        # The load-bearing assertion: the arguments are the parameters,
        # not the whole envelope.
        assert json.loads(calls[0].arguments) == {"code": "print(1)"}

    @pytest.mark.parametrize("parameters", UNCONSTRAINED)
    @pytest.mark.parametrize("chunk_size", [1, 4, 10_000])
    def test_envelope_streams_the_parameters(
        self, parser_cls, mock_tokenizer, parameters, chunk_size
    ):
        parser = parser_cls(mock_tokenizer)
        request = parser.adjust_request(self._request(parameters))

        names: list[str] = []
        args: dict[int, str] = {}
        for start in range(0, len(self.ENVELOPE), chunk_size):
            delta_text = self.ENVELOPE[start : start + chunk_size]
            delta = parser.parse_delta(
                delta_text,
                [ord(c) for c in delta_text],
                request,
                prompt_token_ids=[1] if start == 0 else None,
                finished=start + chunk_size >= len(self.ENVELOPE),
            )
            for tool_call in (delta.tool_calls if delta else None) or []:
                if tool_call.function and tool_call.function.name:
                    names.append(tool_call.function.name)
                if tool_call.function and tool_call.function.arguments:
                    args[tool_call.index] = (
                        args.get(tool_call.index, "") + tool_call.function.arguments
                    )

        assert names == ["run_python"]
        assert [json.loads(a) for a in args.values()] == [{"code": "print(1)"}]

    # A schema that names its keys DOES forbid the envelope, so the model
    # writes bare parameters and the whole object is the arguments.
    @pytest.mark.parametrize(
        "parameters",
        [
            pytest.param(
                {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
                id="typed-properties",
            ),
            pytest.param(None, id="no-parameters-key"),
        ],
    )
    def test_bare_parameters_still_work_when_the_schema_constrains_keys(
        self, parser_cls, mock_tokenizer, parameters
    ):
        parser = parser_cls(mock_tokenizer)
        request = parser.adjust_request(self._request(parameters))

        calls, _ = parser._extract_tool_calls(
            self.BARE, request, enable_auto_tools=True
        )

        assert [c.name for c in calls or []] == ["run_python"]
        assert json.loads(calls[0].arguments) == {"code": "print(1)"}

    def test_arming_is_decided_by_the_schema_not_its_mere_presence(
        self, parser_cls, mock_tokenizer
    ):
        """Guard the guard: the two schema families must arm differently."""
        wrapper = parser_cls(mock_tokenizer)

        def armed_name(parameters):
            engine = LlamaJsonParser(mock_tokenizer)
            request = wrapper.adjust_request(self._request(parameters))
            engine._check_skip_tool_parsing(request)
            return engine._named_bare_args_name

        assert armed_name({"type": "object"}) is None
        assert (
            armed_name({"type": "object", "properties": {"code": {"type": "string"}}})
            == "run_python"
        )


class TestForcedChoiceKeepsUnpromotedText:
    """A forced tool choice must not turn a truncated call into an empty reply.

    Required and named choice suppress content, because the model is
    constrained to emit only tool calls and the wire format's scaffolding --
    the ``[`` of the required-mode array -- must not reach the client.  That
    reasoning fails when generation stops before any call is promoted: the
    text is then the whole response, and suppressing it hands the caller a
    message with neither content nor tool calls.

    ``auto`` returns the text in exactly this case, so forced choice losing
    it is also a divergence between two paths that should agree.
    """

    # Plain dicts: vLLM's named-tool-choice validator indexes tools as
    # mappings (protocol.py), so model objects raise there.
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "run_cmd",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            },
        }
    ]
    NAMED = {"type": "function", "function": {"name": "run_cmd"}}
    TRUNCATED = '{"name": "run_cmd",'
    COMPLETE = '{"name": "run_cmd", "parameters": {"cmd": "ls"}}'

    @pytest.fixture
    def parser_cls(self, monkeypatch):
        # Pin the configuration rather than inherit it: with strict calling
        # off, a named choice installs the parameters schema and arms
        # bare-argument mode, which is a different code path with its own
        # tests.  Leaving this implicit makes the class order-dependent.
        monkeypatch.setenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", "1")
        cls = ParserManager.get_parser(
            tool_parser_name="llama3_json",
            reasoning_parser_name=None,
            enable_auto_tools=True,
        )
        assert cls is not None
        return cls

    def _request(self, tool_choice):
        return ChatCompletionRequest(
            model="llama",
            messages=[{"role": "user", "content": "list files"}],
            tools=self.TOOLS,
            tool_choice=tool_choice,
        )

    def _adjusted(self, parser, tool_choice):
        request = parser.adjust_request(self._request(tool_choice))
        if tool_choice != "auto":
            # Precondition: the structural tag is what keeps this on the
            # envelope path.  Assert it, so a plumbing change fails here and
            # says so rather than silently testing bare-argument mode.
            structured = getattr(request, "structured_outputs", None)
            assert getattr(structured, "structural_tag", None) is not None
        return request

    def _choices(self):
        return [
            pytest.param("required", id="required"),
            pytest.param(self.NAMED, id="named"),
            pytest.param("auto", id="auto"),
        ]

    @pytest.mark.parametrize(
        "tool_choice",
        [
            pytest.param("required", id="required"),
            pytest.param(
                {"type": "function", "function": {"name": "run_cmd"}}, id="named"
            ),
            pytest.param("auto", id="auto"),
        ],
    )
    def test_truncated_call_comes_back_as_content(
        self, parser_cls, mock_tokenizer, tool_choice
    ):
        parser = parser_cls(mock_tokenizer)
        request = self._adjusted(parser, tool_choice)

        calls, content = parser._extract_tool_calls(
            self.TRUNCATED, request, enable_auto_tools=True
        )

        assert not calls
        # Every tool_choice must agree, and none may swallow the response.
        assert content == self.TRUNCATED

    @pytest.mark.parametrize(
        "tool_choice",
        [
            pytest.param("required", id="required"),
            pytest.param(
                {"type": "function", "function": {"name": "run_cmd"}}, id="named"
            ),
            pytest.param("auto", id="auto"),
        ],
    )
    def test_a_promoted_call_still_suppresses_content(
        self, parser_cls, mock_tokenizer, tool_choice
    ):
        """The guard for the fix: suppression must survive where it belongs."""
        parser = parser_cls(mock_tokenizer)
        request = self._adjusted(parser, tool_choice)

        calls, content = parser._extract_tool_calls(
            self.COMPLETE, request, enable_auto_tools=True
        )

        assert [c.name for c in calls or []] == ["run_cmd"]
        assert not content
