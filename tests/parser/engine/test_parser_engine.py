# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for :class:`ParserEngine` — the glue layer between
:class:`StreamingParserEngine` events and the serving layer's
DeltaMessage / ExtractedToolCallInformation protocol.
"""

from __future__ import annotations

from types import SimpleNamespace

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import FunctionDefinition
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
                [EventType.REASONING_END],
            ),
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                [EventType.REASONING_START],
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_ARGS,
                [EventType.TOOL_CALL_START],
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                [EventType.TOOL_CALL_END],
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
                [EventType.TOOL_CALL_START],
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                [EventType.TOOL_CALL_END],
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


# ── TestExtractResponseOutputs ───────────────────────────────────────


class TestExtractResponseOutputs:
    """Tests for ParserEngine.extract_response_outputs()."""

    def test_plain_content(self, mock_request):
        engine = _make_engine(_hermes_config())
        outputs = engine.extract_response_outputs(
            model_output="Hello world",
            model_output_token_ids=[],
            request=mock_request,
        )
        assert len(outputs) == 1
        assert outputs[0].type == "message"
        assert outputs[0].content[0].text == "Hello world"

    def test_reasoning_and_content(self, mock_request):
        engine = _make_engine()
        outputs = engine.extract_response_outputs(
            model_output="Let me think</think>The answer is 42",
            model_output_token_ids=[],
            request=mock_request,
        )
        types = [o.type for o in outputs]
        assert "reasoning" in types
        assert "message" in types

        reasoning_item = next(o for o in outputs if o.type == "reasoning")
        assert reasoning_item.content[0].text == "Let me think"

        message_item = next(o for o in outputs if o.type == "message")
        assert message_item.content[0].text == "The answer is 42"

    def test_tool_call(self, mock_request):
        engine = _make_engine(_hermes_config())
        text = '<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>'
        outputs = engine.extract_response_outputs(
            model_output=text,
            model_output_token_ids=[],
            request=mock_request,
        )
        types = [o.type for o in outputs]
        assert "function_call" in types

        tc = next(o for o in outputs if o.type == "function_call")
        assert tc.name == "f"

    def test_reasoning_plus_tool_call(self, mock_request):
        engine = _make_engine()
        text = 'hmm</think><tool_call>{"name": "g", "arguments": {}}</tool_call>'
        outputs = engine.extract_response_outputs(
            model_output=text,
            model_output_token_ids=[],
            request=mock_request,
        )
        types = [o.type for o in outputs]
        assert "reasoning" in types
        assert "function_call" in types
