# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for :class:`ParserEngine` — the glue layer between
:class:`StreamingParserEngine` events and the serving layer's
DeltaMessage / ExtractedToolCallInformation protocol.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import regex as re

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaToolCall,
    FunctionDefinition,
)
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.engine.adapters import make_adapters
from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

# ── Shared test configs ──────────────────────────────────────────────

_VOCAB: dict[str, int] = {
    "<think>": 200,
    "</think>": 201,
    "<tool_call>": 202,
    "</tool_call>": 203,
}


def _combined_config() -> ParserEngineConfig:
    """Config with reasoning tags and tool-call tags."""
    return ParserEngineConfig(
        name="combined_test",
        terminals={
            "THINK_START": "<think>",
            "THINK_END": "</think>",
            "TOOL_START": "<tool_call>",
            "TOOL_END": "</tool_call>",
        },
        token_id_terminals={
            "THINK_START": "<think>",
            "THINK_END": "</think>",
            "TOOL_START": "<tool_call>",
            "TOOL_END": "</tool_call>",
        },
        transitions={
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
        },
        initial_state=ParserState.REASONING,
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
    )


def _hermes_config() -> ParserEngineConfig:
    """Tool-call-only config (no reasoning)."""
    return ParserEngineConfig(
        name="hermes_test",
        terminals={
            "TOOL_START": "<tool_call>",
            "TOOL_END": "</tool_call>",
        },
        token_id_terminals={
            "TOOL_START": "<tool_call>",
            "TOOL_END": "</tool_call>",
        },
        transitions={
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
    )


def _make_engine(
    config: ParserEngineConfig | None = None,
    tools: list | None = None,
) -> ParserEngine:
    tokenizer = make_mock_tokenizer(_VOCAB)
    cfg = config or _combined_config()
    return ParserEngine(
        tokenizer,
        tools=tools,
        parser_engine_config=cfg,
    )


# ── TestEventsToDelta ────────────────────────────────────────────────


class TestEventsToDelta:
    """Unit tests for ParserEngine._events_to_delta()."""

    def test_text_chunk_produces_content(self):
        engine = _make_engine()
        delta = engine._events_to_delta(
            [
                SemanticEvent(EventType.TEXT_CHUNK, "Hello world"),
            ]
        )
        assert delta is not None
        assert delta.content == "Hello world"
        assert not delta.tool_calls

    def test_reasoning_chunk_produces_reasoning(self):
        engine = _make_engine()
        delta = engine._events_to_delta(
            [
                SemanticEvent(EventType.REASONING_CHUNK, "Let me think"),
            ]
        )
        assert delta is not None
        assert delta.reasoning == "Let me think"
        assert delta.content is None

    def test_empty_events_returns_none(self):
        engine = _make_engine()
        delta = engine._events_to_delta([])
        assert delta is None

    def test_tool_call_produces_tool_call_delta(self):
        engine = _make_engine()
        events = [
            SemanticEvent(EventType.TOOL_CALL_START, tool_index=0),
            SemanticEvent(EventType.TOOL_NAME, "get_weather", tool_index=0),
            SemanticEvent(
                EventType.ARG_VALUE_CHUNK,
                '{"location": "NYC"}',
                tool_index=0,
            ),
            SemanticEvent(EventType.TOOL_CALL_END, tool_index=0),
        ]
        delta = engine._events_to_delta(events)
        assert delta is not None
        assert len(delta.tool_calls) > 0
        names = [
            tc.function.name
            for tc in delta.tool_calls
            if tc.function and tc.function.name
        ]
        assert "get_weather" in names

    def test_reasoning_end_sets_flag(self):
        engine = _make_engine()
        assert engine._reasoning_ended is False
        engine._events_to_delta([SemanticEvent(EventType.REASONING_END)])
        assert engine._reasoning_ended is True

    def test_mixed_content_and_reasoning(self):
        engine = _make_engine()
        events = [
            SemanticEvent(EventType.REASONING_CHUNK, "thinking..."),
            SemanticEvent(EventType.REASONING_END),
            SemanticEvent(EventType.TEXT_CHUNK, "answer"),
        ]
        delta = engine._events_to_delta(events)
        assert delta is not None
        assert delta.reasoning == "thinking..."
        assert delta.content == "answer"

    @pytest.mark.parametrize(
        "events,expected,excluded",
        [
            (
                [SemanticEvent(EventType.TEXT_CHUNK, "Hello world")],
                "content",
                ["tool_calls", "reasoning"],
            ),
            (
                [SemanticEvent(EventType.REASONING_CHUNK, "Let me think")],
                "reasoning",
                ["tool_calls", "content"],
            ),
            (
                [
                    SemanticEvent(EventType.TOOL_CALL_START, tool_index=0),
                    SemanticEvent(EventType.TOOL_NAME, "fn", tool_index=0),
                    SemanticEvent(EventType.ARG_VALUE_CHUNK, '{"k":1}', tool_index=0),
                    SemanticEvent(EventType.TOOL_CALL_END, tool_index=0),
                ],
                "tool_calls",
                ["content", "reasoning"],
            ),
        ],
        ids=["content_only", "reasoning_only", "tool_call_only"],
    )
    def test_delta_excludes_unset_fields(self, events, expected, excluded):
        engine = _make_engine()
        delta = engine._events_to_delta(events)
        assert delta is not None
        dumped = delta.model_dump(exclude_unset=True)
        assert expected in dumped
        for field in excluded:
            assert field not in dumped

    def test_kimi_k2_tool_call_id_includes_func_name(self):
        engine = _make_engine()
        engine._stream_state.tool_call_id_type = "kimi_k2"
        events = [
            SemanticEvent(EventType.TOOL_CALL_START, tool_index=0),
            SemanticEvent(EventType.TOOL_NAME, "get_weather", tool_index=0),
            SemanticEvent(
                EventType.ARG_VALUE_CHUNK,
                '{"city": "NYC"}',
                tool_index=0,
            ),
            SemanticEvent(EventType.TOOL_CALL_END, tool_index=0),
        ]
        delta = engine._events_to_delta(events)
        assert delta is not None
        assert len(delta.tool_calls) == 1
        assert delta.tool_calls[0].id == "functions.get_weather:0"

    def test_multiple_arg_chunks_same_batch_coalesced(self):
        """Multiple events for the same tool in one batch must produce
        at most one DeltaToolCall per index."""
        engine = _make_engine()
        events = [
            SemanticEvent(EventType.TOOL_CALL_START, tool_index=0),
            SemanticEvent(EventType.TOOL_NAME, "get_weather", tool_index=0),
            SemanticEvent(
                EventType.ARG_VALUE_CHUNK,
                '{"city": ',
                tool_index=0,
            ),
            SemanticEvent(
                EventType.ARG_VALUE_CHUNK,
                '"Tokyo"}',
                tool_index=0,
            ),
            SemanticEvent(EventType.TOOL_CALL_END, tool_index=0),
        ]
        delta = engine._events_to_delta(events)
        assert delta is not None
        indices = [tc.index for tc in delta.tool_calls]
        assert len(indices) == len(set(indices)), (
            f"Duplicate indices in tool_calls: {delta.tool_calls}"
        )
        assert delta.tool_calls[0].function.name == "get_weather"
        assert delta.tool_calls[0].id is not None


# ── TestCoalesceToolCallDeltas ──────────────────────────────────────


class TestCoalesceToolCallDeltas:
    """Unit tests for ParserEngine._coalesce_tool_call_deltas()."""

    def test_no_duplicates_unchanged(self):
        deltas = [
            DeltaToolCall(
                index=0,
                id="a",
                type="function",
                function=DeltaFunctionCall(name="f"),
            ),
            DeltaToolCall(
                index=1,
                function=DeltaFunctionCall(arguments="{}"),
            ),
        ]
        result = ParserEngine._coalesce_tool_call_deltas(deltas)
        assert len(result) == 2
        assert result[0].index == 0
        assert result[1].index == 1

    def test_name_and_args_same_index_merged(self):
        deltas = [
            DeltaToolCall(
                index=0,
                id="call_1",
                type="function",
                function=DeltaFunctionCall(name="get_weather"),
            ),
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(arguments='{"city":'),
            ),
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(arguments='"Tokyo"}'),
            ),
        ]
        result = ParserEngine._coalesce_tool_call_deltas(deltas)
        assert len(result) == 1
        assert result[0].index == 0
        assert result[0].id == "call_1"
        assert result[0].type == "function"
        assert result[0].function.name == "get_weather"
        assert result[0].function.arguments == '{"city":"Tokyo"}'

    def test_empty_list(self):
        assert ParserEngine._coalesce_tool_call_deltas([]) == []

    def test_single_element(self):
        tc = DeltaToolCall(
            index=0,
            function=DeltaFunctionCall(name="f"),
        )
        result = ParserEngine._coalesce_tool_call_deltas([tc])
        assert result == [tc]

    def test_partial_duplicates(self):
        deltas = [
            DeltaToolCall(
                index=0,
                id="a",
                type="function",
                function=DeltaFunctionCall(name="f1"),
            ),
            DeltaToolCall(
                index=1,
                id="b",
                type="function",
                function=DeltaFunctionCall(name="f2"),
            ),
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(arguments='{"x":1}'),
            ),
        ]
        result = ParserEngine._coalesce_tool_call_deltas(deltas)
        assert len(result) == 2
        assert result[0].index == 0
        assert result[0].function.name == "f1"
        assert result[0].function.arguments == '{"x":1}'
        assert result[1].index == 1

    def test_id_type_from_later_entry(self):
        deltas = [
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(arguments='{"a":1}'),
            ),
            DeltaToolCall(
                index=0,
                id="call_1",
                type="function",
                function=DeltaFunctionCall(name="f"),
            ),
        ]
        result = ParserEngine._coalesce_tool_call_deltas(deltas)
        assert len(result) == 1
        assert result[0].id == "call_1"
        assert result[0].type == "function"
        assert result[0].function.name == "f"
        assert result[0].function.arguments == '{"a":1}'


# ── TestContentWhitespaceHandling ────────────────────────────────────


class TestContentWhitespaceHandling:
    """Unit tests for whitespace deferral / dropping in _events_to_delta."""

    def test_whitespace_only_deferred_until_next_tick(self):
        engine = _make_engine()
        d1 = engine._events_to_delta(
            [SemanticEvent(EventType.TEXT_CHUNK, "  \n")],
        )
        assert d1 is None
        d2 = engine._events_to_delta(
            [SemanticEvent(EventType.TEXT_CHUNK, "hello")],
        )
        assert d2 is not None
        assert d2.content == "  \nhello"

    def test_whitespace_only_emitted_on_finished(self):
        engine = _make_engine()
        d = engine._events_to_delta(
            [SemanticEvent(EventType.TEXT_CHUNK, "  \n")],
            finished=True,
        )
        assert d is not None
        assert d.content == "  \n"

    def test_whitespace_dropped_before_tool_call(self):
        engine = _make_engine()
        engine._events_to_delta(
            [
                SemanticEvent(EventType.TEXT_CHUNK, "  \n"),
                SemanticEvent(EventType.TOOL_CALL_START, tool_index=0),
            ]
        )
        d = engine._events_to_delta(
            [
                SemanticEvent(EventType.TOOL_NAME, "f", tool_index=0),
                SemanticEvent(
                    EventType.ARG_VALUE_CHUNK,
                    '{"a":1}',
                    tool_index=0,
                ),
                SemanticEvent(EventType.TOOL_CALL_END, tool_index=0),
            ]
        )
        assert d is not None
        assert d.content is None
        assert d.tool_calls

    def test_real_content_before_tool_preserved(self):
        engine = _make_engine()
        d = engine._events_to_delta(
            [
                SemanticEvent(EventType.TEXT_CHUNK, "prefix"),
                SemanticEvent(EventType.TOOL_CALL_START, tool_index=0),
            ]
        )
        assert d is not None
        assert d.content == "prefix"

    def test_whitespace_after_nonws_content_preserved(self):
        engine = _make_engine()
        engine._events_to_delta(
            [SemanticEvent(EventType.TEXT_CHUNK, "hello")],
        )
        d = engine._events_to_delta(
            [SemanticEvent(EventType.TEXT_CHUNK, "  \n")],
        )
        assert d is not None
        assert d.content == "  \n"

    def test_whitespace_after_nonws_not_dropped_with_tools(self):
        engine = _make_engine()
        engine._events_to_delta(
            [SemanticEvent(EventType.TEXT_CHUNK, "hello")],
        )
        d = engine._events_to_delta(
            [
                SemanticEvent(EventType.TEXT_CHUNK, "  \n"),
                SemanticEvent(EventType.TOOL_CALL_START, tool_index=0),
            ]
        )
        assert d is not None
        assert d.content == "  \n"


# ── TestPostToolContentDeferral ──────────────────────────────────────


class TestPostToolContentDeferral:
    """Regression: content after TOOL_CALL_END in the same batch must not
    produce a mixed DeltaMessage(content=..., tool_calls=...) — that causes
    split_delta to reorder content before tool_calls, breaking the Responses
    API state machine."""

    def test_text_after_tool_end_deferred(self):
        engine = _make_engine()
        events = [
            SemanticEvent(EventType.TOOL_CALL_START, tool_index=0),
            SemanticEvent(EventType.TOOL_NAME, "get_weather", tool_index=0),
            SemanticEvent(EventType.ARG_VALUE_CHUNK, '{"city":"NYC"}', tool_index=0),
            SemanticEvent(EventType.TOOL_CALL_END, tool_index=0),
            SemanticEvent(EventType.TEXT_CHUNK, "\nHere is the result"),
        ]
        delta = engine._events_to_delta(events)
        assert delta is not None
        assert delta.tool_calls
        assert delta.content is None

        deferred = engine._events_to_delta([])
        assert deferred is not None
        assert deferred.content == "\nHere is the result"
        assert not deferred.tool_calls

    def test_text_after_tool_deferred_even_when_finished(self):
        engine = _make_engine()
        events = [
            SemanticEvent(EventType.TOOL_CALL_START, tool_index=0),
            SemanticEvent(EventType.TOOL_NAME, "f", tool_index=0),
            SemanticEvent(EventType.ARG_VALUE_CHUNK, "{}", tool_index=0),
            SemanticEvent(EventType.TOOL_CALL_END, tool_index=0),
            SemanticEvent(EventType.TEXT_CHUNK, "done"),
        ]
        delta = engine._events_to_delta(events, finished=True)
        assert delta is not None
        assert delta.tool_calls
        assert delta.content is None

    def test_text_before_tool_not_deferred(self):
        engine = _make_engine()
        engine._content_has_nonws = True
        events = [
            SemanticEvent(EventType.TEXT_CHUNK, "hello"),
            SemanticEvent(EventType.TOOL_CALL_START, tool_index=0),
            SemanticEvent(EventType.TOOL_NAME, "f", tool_index=0),
            SemanticEvent(EventType.ARG_VALUE_CHUNK, "{}", tool_index=0),
            SemanticEvent(EventType.TOOL_CALL_END, tool_index=0),
        ]
        delta = engine._events_to_delta(events)
        assert delta is not None
        assert delta.content == "hello"
        assert delta.tool_calls

    def test_deferred_content_not_flushed_during_arg_continuation(self):
        """Deferred content from batch N must not mix with arg-continuation
        tool events in batch N+1 — that creates a DeltaMessage with both
        content and nameless tool_calls, which crashes the Responses API
        state machine (name=None → Pydantic ValidationError)."""
        engine = _make_engine()
        engine._content_has_nonws = True

        batch1 = [
            SemanticEvent(EventType.TOOL_CALL_START, tool_index=0),
            SemanticEvent(EventType.TOOL_NAME, "get_weather", tool_index=0),
            SemanticEvent(EventType.ARG_VALUE_CHUNK, '{"city":', tool_index=0),
            SemanticEvent(EventType.TEXT_CHUNK, "\n"),
        ]
        delta1 = engine._events_to_delta(batch1)
        assert delta1 is not None
        assert delta1.tool_calls
        assert delta1.content is None

        batch2 = [
            SemanticEvent(EventType.ARG_VALUE_CHUNK, '"NYC"}', tool_index=0),
        ]
        delta2 = engine._events_to_delta(batch2)
        assert delta2 is not None
        assert delta2.tool_calls
        assert delta2.content is None

        flush = engine._events_to_delta([])
        assert flush is not None
        assert flush.content == "\n"
        assert not flush.tool_calls


# ── TestFixArgTypes ──────────────────────────────────────────────────


def _make_tool(name: str, properties: dict) -> ChatCompletionToolsParam:
    return ChatCompletionToolsParam(
        type="function",
        function=FunctionDefinition(
            name=name,
            parameters={"type": "object", "properties": properties},
        ),
    )


class TestFixArgTypes:
    """Tests for ParserEngine._fix_arg_types()."""

    def test_string_param_reverted_from_int(self):
        tool = _make_tool("f", {"zipcode": {"type": "string"}})
        engine = _make_engine(tools=[tool])
        result = engine._fix_arg_types('{"zipcode": 12345}', "f")
        assert '"zipcode": "12345"' in result

    def test_string_param_reverted_from_bool(self):
        tool = _make_tool("f", {"flag": {"type": "string"}})
        engine = _make_engine(tools=[tool])
        result = engine._fix_arg_types('{"flag": true}', "f")
        assert '"flag": "true"' in result

    def test_string_param_reverted_from_null(self):
        tool = _make_tool("f", {"val": {"type": "string"}})
        engine = _make_engine(tools=[tool])
        result = engine._fix_arg_types('{"val": null}', "f")
        assert '"val": "null"' in result

    def test_int_param_not_changed(self):
        tool = _make_tool("f", {"count": {"type": "integer"}})
        engine = _make_engine(tools=[tool])
        result = engine._fix_arg_types('{"count": 42}', "f")
        assert '"count": 42' in result

    def test_no_tools_returns_unchanged(self):
        engine = _make_engine(tools=None)
        original = '{"a": 1}'
        assert engine._fix_arg_types(original, "f") == original

    def test_unknown_function_returns_unchanged(self):
        tool = _make_tool("known", {"x": {"type": "string"}})
        engine = _make_engine(tools=[tool])
        original = '{"x": 1}'
        assert engine._fix_arg_types(original, "unknown") == original

    def test_invalid_json_returns_unchanged(self):
        tool = _make_tool("f", {"x": {"type": "string"}})
        engine = _make_engine(tools=[tool])
        original = "not json"
        assert engine._fix_arg_types(original, "f") == original

    def test_string_value_not_touched(self):
        tool = _make_tool("f", {"name": {"type": "string"}})
        engine = _make_engine(tools=[tool])
        original = '{"name": "Alice"}'
        assert engine._fix_arg_types(original, "f") == original

    @pytest.mark.parametrize(
        "properties, input_json, expected_substr",
        [
            ({"count": {"type": "integer"}}, '{"count": "42"}', '"count": 42'),
            ({"score": {"type": "number"}}, '{"score": "3.14"}', '"score": 3.14'),
            ({"flag": {"type": "boolean"}}, '{"flag": "true"}', '"flag": true'),
            ({"flag": {"type": "boolean"}}, '{"flag": "false"}', '"flag": false'),
            ({"val": {"type": "null"}}, '{"val": "null"}', '"val": null'),
            ({"val": {"type": ["string", "null"]}}, '{"val": "null"}', '"val": null'),
            ({"score": {"type": "number"}}, '{"score": "108."}', '"score": 108'),
        ],
        ids=[
            "string_to_int",
            "string_to_float",
            "string_to_bool_true",
            "string_to_bool_false",
            "string_to_null",
            "string_to_null_union",
            "trailing_dot_float",
        ],
    )
    def test_string_coerced_to_schema_type(
        self,
        properties,
        input_json,
        expected_substr,
    ):
        tool = _make_tool("f", properties)
        engine = _make_engine(tools=[tool])
        result = engine._fix_arg_types(input_json, "f")
        assert expected_substr in result

    def test_mixed_types_coerced(self):
        tool = _make_tool(
            "f",
            {
                "count": {"type": "integer"},
                "active": {"type": "boolean"},
                "score": {"type": "number"},
            },
        )
        engine = _make_engine(tools=[tool])
        result = engine._fix_arg_types(
            '{"count": "42", "active": "true", "score": "3.14"}', "f"
        )
        parsed = json.loads(result)
        assert parsed["count"] == 42
        assert parsed["active"] is True
        assert parsed["score"] == 3.14

    def test_nested_object_coercion(self):
        tool = _make_tool(
            "f",
            {
                "inner": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                    },
                },
            },
        )
        engine = _make_engine(tools=[tool])
        result = engine._fix_arg_types('{"inner": {"count": "42"}}', "f")
        parsed = json.loads(result)
        assert parsed["inner"]["count"] == 42

    def test_array_item_coercion(self):
        tool = _make_tool(
            "f",
            {
                "nums": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
        )
        engine = _make_engine(tools=[tool])
        result = engine._fix_arg_types('{"nums": ["42", "5"]}', "f")
        parsed = json.loads(result)
        assert parsed["nums"] == [42, 5]

    def test_array_mixed_item_types(self):
        tool = _make_tool(
            "f",
            {
                "vals": {
                    "type": "array",
                    "items": {"type": "number"},
                },
            },
        )
        engine = _make_engine(tools=[tool])
        result = engine._fix_arg_types('{"vals": ["42", "3.14"]}', "f")
        parsed = json.loads(result)
        assert parsed["vals"] == [42, 3.14]


# ── TestBuildExtractedResult ─────────────────────────────────────────


class TestBuildExtractedResult:
    """Tests for ParserEngine._build_extracted_result()."""

    def test_no_tool_calls(self):
        engine = _make_engine()
        result = engine._build_extracted_result()
        assert result.tools_called is False
        assert result.tool_calls == []

    def test_single_tool_call(self):
        engine = _make_engine(_hermes_config())
        text = '<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>'
        events = engine._engine.feed(text, [])
        events.extend(engine._engine.finish())
        delta = engine._events_to_delta(events)
        result = engine._build_extracted_result(delta)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "f"

    def test_content_passthrough(self):
        engine = _make_engine(_hermes_config())
        text = "Hello world"
        events = engine._engine.feed(text, [])
        events.extend(engine._engine.finish())
        delta = engine._events_to_delta(events, finished=True)
        result = engine._build_extracted_result(delta)
        assert result.tools_called is False
        assert result.content == "Hello world"


# ── TestEngineBasedPath ──────────────────────────────────────────────


class TestEngineBasedPath:
    """Tests for the _engine_based accumulation behavior in
    DelegatingParser.parse_delta."""

    def test_engine_based_true_when_both_parsers_engine(self):
        r = SimpleNamespace(engine_based_streaming=True)
        t = SimpleNamespace(engine_based_streaming=True)
        engine_based = r.engine_based_streaming and t.engine_based_streaming
        assert engine_based is True

    def test_engine_based_false_when_reasoning_parser_not_engine(self):
        r = SimpleNamespace(engine_based_streaming=False)
        t = SimpleNamespace(engine_based_streaming=True)
        engine_based = r.engine_based_streaming and t.engine_based_streaming
        assert engine_based is False

    def test_parse_delta_streaming(self, mock_request):
        """Engine's parse_delta returns content from streaming events."""
        engine = _make_engine(_hermes_config())
        engine._streaming_initialized = True
        result = engine.parse_delta(
            "Hello",
            [],
            mock_request,
            finished=False,
        )
        assert result is not None
        assert result.content == "Hello"

    def test_parse_delta_tool_call(self, mock_request):
        """Engine's parse_delta handles tool calls in streaming."""
        engine = _make_engine(_hermes_config())
        engine._streaming_initialized = True
        result = engine.parse_delta(
            '<tool_call>{"name": "f", "arguments": {}}</tool_call>',
            [],
            mock_request,
            finished=True,
        )
        assert result is not None
        assert len(result.tool_calls) > 0


# ── TestParseTokenIdPassthrough ────────────────────────────────────


class TestParseTokenIdPassthrough:
    """parse() must forward model_output_token_ids to _single_pass_parse
    so that token-ID-based strict terminal matching is active."""

    def test_literal_tool_tag_in_content_preserved_with_token_ids(self, mock_request):
        engine = _make_engine(_hermes_config())
        text = (
            "Use <tool_call> to call tools."
            '<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>'
        )
        token_ids = [
            65,
            66,
            67,
            68,
            69,
            70,
            71,  # "Use <tool_call> to call tools."
            202,  # real <tool_call>
            72,
            73,
            74,  # '{"name": "f", ...}'
            203,  # real </tool_call>
        ]

        _, content, tool_calls = engine.parse(
            text, mock_request, model_output_token_ids=token_ids
        )

        assert content is not None
        assert "<tool_call>" in content
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "f"

    def test_parse_with_token_ids_basic(self, mock_request):
        engine = _make_engine(_hermes_config())
        text = '<tool_call>{"name": "h", "arguments": {"x": 1}}</tool_call>'
        token_ids = [202, 65, 66, 67, 203]

        _, content, tool_calls = engine.parse(
            text, mock_request, model_output_token_ids=token_ids
        )

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "h"

    def test_parse_without_token_ids_backward_compat(self, mock_request):
        engine = _make_engine(_hermes_config())
        text = '<tool_call>{"name": "g", "arguments": {}}</tool_call>'

        _, content, tool_calls = engine.parse(text, mock_request)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "g"


# ── TestAdapterFinishOnStreamEnd ────────────────────────────────────


class _CombinedTestEngine(ParserEngine):
    def __init__(self, tokenizer, tools=None, **kwargs):
        super().__init__(
            tokenizer, tools, parser_engine_config=_combined_config(), **kwargs
        )


_CombinedReasoningAdapter, _CombinedToolAdapter = make_adapters(_CombinedTestEngine)


class _CombinedDelegating(DelegatingParser):
    reasoning_parser_cls = _CombinedReasoningAdapter
    tool_parser_cls = _CombinedToolAdapter


def _make_delegating_request():
    req = MagicMock(spec=ChatCompletionRequest)
    req.tools = []
    req.tool_choice = "auto"
    return req


class TestAdapterFinishOnStreamEnd:
    """Engine adapters must flush buffered text when streaming ends.

    When a DelegatingParser wraps engine adapters, the underlying
    StreamingParserEngine.finish() must be called on the last
    parse_delta(finished=True) so that lexer-buffered text (terminal
    prefixes) and scanner-deferred terminals are not silently lost.
    """

    def test_lexer_buffer_flushed_on_finished(self):
        """Text buffered as a potential terminal prefix must be emitted
        as content when the stream ends."""
        tokenizer = make_mock_tokenizer(_VOCAB)
        parser = _CombinedDelegating(tokenizer)
        request = _make_delegating_request()

        # Feed reasoning then content with a trailing '<' that looks like
        # the start of a terminal ('<think>' or '<tool_call>').
        parser.parse_delta("</think>", [201], request, finished=False)
        delta = parser.parse_delta("Hello world<", [], request, finished=True)
        # The '<' must NOT be silently dropped.
        assert delta is not None
        assert delta.content is not None
        assert "<" in delta.content, (
            "Trailing '<' lost: lexer buffer was not flushed on finish"
        )

    def test_args_buffer_flushed_on_finished(self):
        """Pending arg buffer text must be emitted when stream ends
        mid-tool-call (closing brace held back in buffer)."""
        tokenizer = make_mock_tokenizer(_VOCAB)
        parser = _CombinedDelegating(tokenizer)
        request = _make_delegating_request()

        parser.parse_delta("</think>", [201], request, finished=False)
        parser.parse_delta("<tool_call>", [202], request, finished=False)
        parser.parse_delta('{"name": "f"}', [], request, finished=False)
        # The closing } is held back in args buffer, waiting for
        # a TOOL_END terminal. Stream ends without one — finish()
        # must flush the buffer.
        delta = parser.parse_delta("", [], request, finished=True)
        assert delta is not None, (
            "Engine finish should produce a delta with flushed args/end"
        )


# ── TestReasoningOnlyDelegatingParser ─────────────────────────────


class _ReasoningOnlyDelegating(DelegatingParser):
    """DelegatingParser with reasoning adapter but NO tool adapter."""

    reasoning_parser_cls = _CombinedReasoningAdapter
    tool_parser_cls = None


class TestReasoningOnlyEndTokenLeak:
    """When there is no tool parser, the content passthrough must not
    re-emit the end-of-reasoning marker (e.g. ``</think>``) as content.

    Regression test for the scenario where ``</think>`` arrives as a
    single-token delta: the engine correctly consumes it (emitting
    REASONING_END with no content), but the content passthrough
    fired because ``delta_message is None`` and reasoning had just ended.
    """

    def test_think_end_not_leaked_as_content(self):
        tokenizer = make_mock_tokenizer(_VOCAB)
        parser = _ReasoningOnlyDelegating(tokenizer)
        request = _make_delegating_request()

        # Feed reasoning text.
        d1 = parser.parse_delta(
            "I am thinking",
            [],
            request,
            finished=False,
        )
        assert d1 is not None
        assert d1.reasoning is not None
        assert d1.content is None

        # Feed </think> as a single-token delta.
        d2 = parser.parse_delta(
            "</think>",
            [201],
            request,
            finished=False,
        )
        # The end-of-reasoning marker must NOT appear as content.
        if d2 is not None:
            assert d2.content is None, f"</think> leaked as content: {d2.content!r}"

        # Feed content after reasoning.
        d3 = parser.parse_delta(
            "\n\nHello!",
            [],
            request,
            finished=False,
        )
        assert d3 is not None
        assert d3.content is not None
        assert "</think>" not in d3.content

    def test_streaming_content_matches_non_streaming(self):
        """Concatenated streaming content must match extract_reasoning."""
        tokenizer = make_mock_tokenizer(_VOCAB)
        # No <think> in input: the combined config starts in REASONING
        # state, so all text before </think> is reasoning.
        full_text = "reasoning</think>\n\nHello!"

        # Non-streaming extraction.
        parser_ns = _ReasoningOnlyDelegating(tokenizer)
        request = _make_delegating_request()
        reasoning, content = parser_ns.extract_reasoning(full_text, request)
        assert reasoning == "reasoning"
        assert content == "\n\nHello!"

        # Streaming extraction — simulate per-token deltas.
        parser_s = _ReasoningOnlyDelegating(tokenizer)
        deltas = [
            ("reasoning", []),
            ("</think>", [201]),
            ("\n\n", []),
            ("Hello!", []),
        ]
        content_parts: list[str] = []
        for text, ids in deltas:
            dm = parser_s.parse_delta(text, ids, request, finished=False)
            if dm is not None and dm.content:
                content_parts.append(dm.content)
        dm = parser_s.parse_delta("", [], request, finished=True)
        if dm is not None and dm.content:
            content_parts.append(dm.content)

        streaming_content = "".join(content_parts)
        assert streaming_content == content, (
            f"Streaming content {streaming_content!r} "
            f"does not match non-streaming {content!r}"
        )

    def test_multi_token_delta_preserves_content_after_think_end(self):
        """Content after </think> in the same delta must not be lost."""
        tokenizer = make_mock_tokenizer(_VOCAB)
        parser = _ReasoningOnlyDelegating(tokenizer)
        request = _make_delegating_request()

        # Feed reasoning text.
        d1 = parser.parse_delta(
            "thinking",
            [],
            request,
            finished=False,
        )
        assert d1 is not None
        assert d1.reasoning is not None

        # Feed </think> and content in the same delta (e.g. speculative
        # decoding accepting multiple tokens at once).  Token IDs must
        # cover all text so the scanner can split correctly.
        # chr(10)='\n', chr(72)='H', chr(105)='i', chr(33)='!'
        d2 = parser.parse_delta(
            "</think>\n\nHi!",
            [201, 10, 10, 72, 105, 33],
            request,
            finished=False,
        )
        assert d2 is not None, "Content after </think> in multi-token delta was lost"
        assert d2.content is not None, (
            "Content after </think> in multi-token delta was nullified"
        )
        assert "</think>" not in d2.content
        assert "Hi!" in d2.content


# ── TestToolAdapterForwardsKwargs ──────────────────────────────────


class TestToolAdapterForwardsKwargs:
    """ParserEngineToolAdapter.__init__ must forward **kwargs to the
    parser engine class so chat_template_kwargs reach model parsers."""

    @pytest.mark.parametrize(
        "enable_thinking,expected_state",
        [
            (False, ParserState.CONTENT),
            (True, ParserState.REASONING),
        ],
    )
    def test_kwargs_forwarded_to_parser_engine(self, enable_thinking, expected_state):
        from vllm.parser.qwen3 import Qwen3Parser

        vocab = {"<think>": 100, "</think>": 101}
        tokenizer = make_mock_tokenizer(vocab)

        _, ToolAdapter = make_adapters(Qwen3Parser)
        adapter = ToolAdapter(
            tokenizer,
            tools=None,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
        engine = adapter._parser_engine
        assert engine.parser_engine_config.initial_state == expected_state


# ── TestExtractContentIdsNoEmptyReturn ─────────────────────────────


class TestExtractContentIdsNoEmptyReturn:
    """extract_content_ids must return input_ids (not []) when there is
    no THINK_END token ID and _reasoning_ended is True."""

    _NO_THINK_CONFIG = ParserEngineConfig(name="no_think_end", token_id_terminals={})

    @pytest.mark.parametrize("input_ids", [[1, 2, 3], []])
    def test_returns_input_ids_without_think_end(self, input_ids):
        engine = _make_engine(self._NO_THINK_CONFIG)
        assert engine._reasoning_end_token_id is None
        engine._reasoning_ended = True
        assert engine.extract_content_ids(input_ids) == input_ids


# ── TestValuePostprocessorRemoved ──────────────────────────────────


class TestValuePostprocessorRemoved:
    """ParserEngineConfig no longer has a value_postprocessor field."""

    def test_no_value_postprocessor_field(self):
        config = ParserEngineConfig(name="test")
        assert not hasattr(config, "value_postprocessor")

    def test_constructor_rejects_value_postprocessor(self):
        with pytest.raises(TypeError):
            ParserEngineConfig(
                name="test",
                value_postprocessor=lambda x: x,  # type: ignore[call-arg]
            )


# ── TestArgDeltaWithConverter ─────────────────────────────────────


_KV_RE = re.compile(r"(\w+)=(\S+)")


def _kv_converter(raw_args: str, partial: bool) -> str:
    params: dict[str, str] = {}
    for m in _KV_RE.finditer(raw_args):
        params[m.group(1)] = m.group(2)
    return json.dumps(params, ensure_ascii=False)


def _converter_config(
    converter=_kv_converter,
    name: str = "converter_test",
) -> ParserEngineConfig:
    return ParserEngineConfig(
        name=name,
        terminals={
            "TOOL_START": "<tool_call>",
            "TOOL_END": "</tool_call>",
        },
        token_id_terminals={
            "TOOL_START": "<tool_call>",
            "TOOL_END": "</tool_call>",
        },
        transitions={
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
        arg_converter=converter,
        stream_arg_deltas=True,
    )


def _collect_arg_deltas(deltas: list) -> str:
    parts: list[str] = []
    for d in deltas:
        if d is None:
            continue
        for tc in d.tool_calls or []:
            if tc.function and tc.function.arguments:
                parts.append(tc.function.arguments)
    return "".join(parts)


def _run_streaming_tool(engine, name: str, chunks: list[str]) -> dict:
    deltas = []
    deltas.append(
        engine._events_to_delta(
            [SemanticEvent(EventType.TOOL_CALL_START, tool_index=0)]
        )
    )
    deltas.append(
        engine._events_to_delta(
            [SemanticEvent(EventType.TOOL_NAME, name, tool_index=0)]
        )
    )
    for chunk in chunks:
        deltas.append(
            engine._events_to_delta(
                [SemanticEvent(EventType.ARG_VALUE_CHUNK, chunk, tool_index=0)]
            )
        )
    deltas.append(
        engine._events_to_delta([SemanticEvent(EventType.TOOL_CALL_END, tool_index=0)])
    )
    return json.loads(_collect_arg_deltas(deltas))


class TestArgDeltaWithConverter:
    """Exercise _compute_arg_delta with arg_converter + stream_arg_deltas.

    The startswith guard on line 814 of parser_engine.py validates that
    converted JSON grows prefix-monotonically across streaming ticks.
    These tests exercise that path with a synthetic config.
    """

    def test_streaming_arg_deltas_prefix_monotonic(self):
        engine = _make_engine(_converter_config())
        deltas = []

        deltas.append(
            engine._events_to_delta(
                [SemanticEvent(EventType.TOOL_CALL_START, tool_index=0)],
            )
        )
        deltas.append(
            engine._events_to_delta(
                [SemanticEvent(EventType.TOOL_NAME, "f", tool_index=0)],
            )
        )
        deltas.append(
            engine._events_to_delta(
                [
                    SemanticEvent(
                        EventType.ARG_VALUE_CHUNK,
                        "a=hello ",
                        tool_index=0,
                    )
                ],
            )
        )
        deltas.append(
            engine._events_to_delta(
                [
                    SemanticEvent(
                        EventType.ARG_VALUE_CHUNK,
                        "b=world ",
                        tool_index=0,
                    )
                ],
            )
        )
        deltas.append(
            engine._events_to_delta(
                [
                    SemanticEvent(
                        EventType.ARG_VALUE_CHUNK,
                        "c=ok",
                        tool_index=0,
                    )
                ],
            )
        )
        deltas.append(
            engine._events_to_delta(
                [SemanticEvent(EventType.TOOL_CALL_END, tool_index=0)],
            )
        )

        all_args = _collect_arg_deltas(deltas)
        assert json.loads(all_args) == {
            "a": "hello",
            "b": "world",
            "c": "ok",
        }

    def test_streaming_arg_deltas_with_type_coercion(self):
        tool = _make_tool(
            "f",
            {
                "count": {"type": "integer"},
                "name": {"type": "string"},
            },
        )
        engine = _make_engine(_converter_config(), tools=[tool])
        deltas = []

        deltas.append(
            engine._events_to_delta(
                [SemanticEvent(EventType.TOOL_CALL_START, tool_index=0)],
            )
        )
        deltas.append(
            engine._events_to_delta(
                [SemanticEvent(EventType.TOOL_NAME, "f", tool_index=0)],
            )
        )
        deltas.append(
            engine._events_to_delta(
                [
                    SemanticEvent(
                        EventType.ARG_VALUE_CHUNK,
                        "count=5 ",
                        tool_index=0,
                    )
                ],
            )
        )
        deltas.append(
            engine._events_to_delta(
                [
                    SemanticEvent(
                        EventType.ARG_VALUE_CHUNK,
                        "name=test",
                        tool_index=0,
                    )
                ],
            )
        )
        deltas.append(
            engine._events_to_delta(
                [SemanticEvent(EventType.TOOL_CALL_END, tool_index=0)],
            )
        )

        all_args = _collect_arg_deltas(deltas)
        parsed = json.loads(all_args)
        assert parsed == {"count": 5, "name": "test"}
        assert isinstance(parsed["count"], int)


# ── TestSafeArgPrefix ────────────────────────────────────────────


class TestSafeArgPrefix:
    """Unit tests for ParserEngine._safe_arg_prefix."""

    @pytest.mark.parametrize(
        "json_str, expected",
        [
            ('{"a": 1}', '{"a": '),
            ('{"a": 1, "b": 2}', '{"a": 1, "b": '),
            ('{"a": "hello", "b": "world"}', '{"a": "hello", "b": '),
            ('{"obj": {"x": 1}, "b": 2}', '{"obj": {"x": 1}, "b": '),
            ('{"url": "http://x:80", "b": 1}', '{"url": "http://x:80", "b": '),
            ('{"a": 1', '{"a": '),
            ("{}", ""),
            ("{", ""),
            ("", ""),
            ('{"k":1}', '{"k":'),
            ('{"k": 1, "v":2}', '{"k": 1, "v":'),
        ],
    )
    def test_safe_arg_prefix(self, json_str, expected):
        assert ParserEngine._safe_arg_prefix(json_str) == expected


# ── Coercion instability regression tests ────────────────────────


def _growing_kv_converter(raw_args: str, partial: bool) -> str:
    """Converter that produces growing bare values (no delimiter)."""
    params: dict[str, str] = {}
    for part in raw_args.split(" "):
        if "=" in part:
            k, v = part.split("=", 1)
            params[k] = v
    return json.dumps(params, ensure_ascii=False)


class TestCoercionInstabilityRegression:
    """Regression tests for _fix_arg_types coercion instability.

    These tests exercise scenarios where a trailing value's coercion
    status changes between ticks (e.g. "4" coerces to int but "4e"
    does not).  Before the _safe_arg_prefix fix, these would violate
    the startswith prefix invariant and permanently drop deltas.
    """

    def test_coercion_flip_does_not_corrupt_stream(self):
        tool = _make_tool(
            "f",
            {
                "count": {"type": "integer"},
                "flag": {"type": "string"},
            },
        )
        engine = _make_engine(_converter_config(), tools=[tool])
        parsed = _run_streaming_tool(
            engine,
            "f",
            ["count=42 ", "flag=ok"],
        )
        assert parsed == {"count": 42, "flag": "ok"}
        assert isinstance(parsed["count"], int)

    def test_bool_partial_value_coercion_is_safe(self):
        """Boolean value building char by char must not break prefix."""
        tool = _make_tool(
            "f",
            {
                "name": {"type": "string"},
                "flag": {"type": "boolean"},
            },
        )
        cfg = _converter_config(_growing_kv_converter)
        engine = _make_engine(cfg, tools=[tool])
        parsed = _run_streaming_tool(
            engine,
            "f",
            ["name=hello ", "flag=t", "r", "u", "e"],
        )
        assert parsed == {"name": "hello", "flag": True}
        assert isinstance(parsed["flag"], bool)

    def test_int_partial_value_flip_is_safe(self):
        """Integer that becomes non-coercible must not break prefix.

        A dummy first arg is needed so the name emission consumes the
        first ARG_VALUE_CHUNK, ensuring _compute_arg_delta runs for the
        chunk where val="4" coerces to int 4.  On the next chunk val
        grows to "4e" which is NOT a valid int, flipping the coercion.
        """
        tool = _make_tool(
            "f",
            {
                "dummy": {"type": "string"},
                "val": {"type": "integer"},
                "extra": {"type": "string"},
            },
        )
        cfg = _converter_config(_growing_kv_converter)
        engine = _make_engine(cfg, tools=[tool])
        parsed = _run_streaming_tool(
            engine,
            "f",
            ["dummy=x ", "val=4", "e ", "extra=ok"],
        )
        assert parsed["dummy"] == "x"
        assert parsed["val"] == "4e"
        assert isinstance(parsed["val"], str)
        assert parsed["extra"] == "ok"
