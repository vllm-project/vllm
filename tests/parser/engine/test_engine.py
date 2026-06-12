# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the streaming parser engine core pipeline."""

from unittest.mock import MagicMock

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.incremental_lexer import (
    LexerShape,
    TerminalDef,
    terminals_from_literals,
)
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine


def _hermes_config() -> ParserEngineConfig:
    """Simple Hermes-style config: <tool_call>JSON</tool_call>."""
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


def _think_config() -> ParserEngineConfig:
    """Simple think-tag reasoning config: <think>...</think>."""
    return ParserEngineConfig(
        name="think_test",
        terminals={
            "THINK_START": "<think>",
            "THINK_END": "</think>",
        },
        transitions={
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
        },
    )


class TestNonStreaming:
    def test_plain_text(self):
        engine = StreamingParserEngine(_hermes_config(), tokenizer=None)
        events = engine.parse_complete("Hello, world!")
        assert len(events) == 1
        assert events[0].type == EventType.TEXT_CHUNK
        assert events[0].value == "Hello, world!"

    def test_single_tool_call(self):
        engine = StreamingParserEngine(_hermes_config(), tokenizer=None)
        text = (
            '<tool_call>{"name": "get_weather",'
            ' "arguments": {"city": "SF"}}'
            "</tool_call>"
        )
        events = engine.parse_complete(text)

        types = [e.type for e in events]
        assert EventType.TOOL_CALL_START in types
        assert EventType.TOOL_CALL_END in types
        assert EventType.ARG_VALUE_CHUNK in types

        arg_text = "".join(
            e.value for e in events if e.type == EventType.ARG_VALUE_CHUNK
        )
        assert '"name": "get_weather"' in arg_text
        assert '"city": "SF"' in arg_text

    def test_text_then_tool_call(self):
        engine = StreamingParserEngine(_hermes_config(), tokenizer=None)
        text = 'Sure!<tool_call>{"name": "add"}</tool_call>'
        events = engine.parse_complete(text)

        types = [e.type for e in events]
        assert types[0] == EventType.TEXT_CHUNK
        assert events[0].value == "Sure!"
        assert EventType.TOOL_CALL_START in types
        assert EventType.TOOL_CALL_END in types

    def test_multiple_tool_calls(self):
        engine = StreamingParserEngine(_hermes_config(), tokenizer=None)
        text = (
            '<tool_call>{"name": "a"}</tool_call><tool_call>{"name": "b"}</tool_call>'
        )
        events = engine.parse_complete(text)

        starts = [e for e in events if e.type == EventType.TOOL_CALL_START]
        ends = [e for e in events if e.type == EventType.TOOL_CALL_END]
        assert len(starts) == 2
        assert len(ends) == 2
        assert starts[0].tool_index == 0
        assert starts[1].tool_index == 1

    def test_reasoning(self):
        engine = StreamingParserEngine(_think_config(), tokenizer=None)
        text = "<think>Let me think...</think>The answer is 42."
        events = engine.parse_complete(text)

        types = [e.type for e in events]
        assert types[0] == EventType.REASONING_START
        assert EventType.REASONING_CHUNK in types
        assert EventType.REASONING_END in types
        assert EventType.TEXT_CHUNK in types

        reasoning = "".join(
            e.value for e in events if e.type == EventType.REASONING_CHUNK
        )
        assert "Let me think..." in reasoning

        content = "".join(e.value for e in events if e.type == EventType.TEXT_CHUNK)
        assert "The answer is 42." in content


class TestStreaming:
    @staticmethod
    def _feed_chars(
        engine: StreamingParserEngine,
        text: str,
    ) -> list[SemanticEvent]:
        """Feed text one character at a time."""
        all_events = []
        for ch in text:
            all_events.extend(engine.feed(ch, []))
        all_events.extend(engine.finish())
        return all_events

    @staticmethod
    def _feed_chunks(
        engine: StreamingParserEngine,
        text: str,
        chunk_size: int,
    ) -> list[SemanticEvent]:
        """Feed text in fixed-size chunks."""
        all_events = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            all_events.extend(engine.feed(chunk, []))
        all_events.extend(engine.finish())
        return all_events

    def test_char_by_char_tool_call(self):
        engine = StreamingParserEngine(_hermes_config(), tokenizer=None)
        text = '<tool_call>{"name": "add", "arguments": {"a": 1}}</tool_call>'
        events = self._feed_chars(engine, text)

        types = [e.type for e in events]
        assert EventType.TOOL_CALL_START in types
        assert EventType.TOOL_CALL_END in types
        assert EventType.ARG_VALUE_CHUNK in types

        arg_text = "".join(
            e.value for e in events if e.type == EventType.ARG_VALUE_CHUNK
        )
        assert '"name": "add"' in arg_text

    @pytest.mark.parametrize(
        "text",
        [
            '<tool_call>{"name": "get", "arguments": {"x": "hello"}}</tool_call>',
            '<tool_call>{"name": "f", "arguments": '
            '{"items": [1, [2, 3]], "obj": {"k": "v"}}}'
            "</tool_call>",
        ],
        ids=["flat_args", "nested_arrays"],
    )
    def test_chunk_sizes_produce_same_content(self, text):
        """Different chunk sizes must produce identical concatenated content."""
        results = {}
        for chunk_size in [1, 2, 3, 5, 7, len(text)]:
            engine = StreamingParserEngine(_hermes_config(), tokenizer=None)
            events = self._feed_chunks(engine, text, chunk_size)
            arg_text = "".join(
                e.value for e in events if e.type == EventType.ARG_VALUE_CHUNK
            )
            results[chunk_size] = arg_text

        values = list(results.values())
        for v in values[1:]:
            assert v == values[0], f"Mismatch: {results}"

    def test_prefix_buffering_prevents_premature_emit(self):
        """Text like '<tool_' should be buffered, not emitted as content."""
        engine = StreamingParserEngine(_hermes_config(), tokenizer=None)

        events1 = engine.feed("<tool_", [])
        content_events = [e for e in events1 if e.type == EventType.TEXT_CHUNK]
        assert len(content_events) == 0, "Should buffer partial tag"

        events2 = engine.feed("call>", [])
        starts = [e for e in events2 if e.type == EventType.TOOL_CALL_START]
        assert len(starts) == 1

    def test_prefix_buffering_flush_on_mismatch(self):
        """Text like '<tool_box' should eventually flush as content."""
        engine = StreamingParserEngine(_hermes_config(), tokenizer=None)

        events1 = engine.feed("<tool_", [])
        assert len([e for e in events1 if e.type == EventType.TEXT_CHUNK]) == 0

        events2 = engine.feed("box>rest", [])
        events2.extend(engine.finish())
        content = "".join(e.value for e in events2 if e.type == EventType.TEXT_CHUNK)
        assert content == "<tool_box>rest"

    def test_reasoning_streaming(self):
        engine = StreamingParserEngine(_think_config(), tokenizer=None)
        events = self._feed_chars(engine, "<think>hmm</think>answer")

        reasoning = "".join(
            e.value for e in events if e.type == EventType.REASONING_CHUNK
        )
        content = "".join(e.value for e in events if e.type == EventType.TEXT_CHUNK)
        assert "hmm" in reasoning
        assert "answer" in content

    def test_text_between_tool_calls(self):
        engine = StreamingParserEngine(_hermes_config(), tokenizer=None)
        text = (
            'Hi<tool_call>{"name":"a"}</tool_call>'
            'mid<tool_call>{"name":"b"}</tool_call>end'
        )
        events = self._feed_chunks(engine, text, 3)

        texts = "".join(e.value for e in events if e.type == EventType.TEXT_CHUNK)
        assert "Hi" in texts
        assert "mid" in texts
        assert "end" in texts

        starts = [e for e in events if e.type == EventType.TOOL_CALL_START]
        assert len(starts) == 2

    def test_unmatched_close_brace_does_not_poison_depth(self):
        """A stray } in malformed JSON must not kill streaming for
        all subsequent content."""
        engine = StreamingParserEngine(_hermes_config(), tokenizer=None)
        engine.feed("<tool_call>", [])

        malformed = '}{{"a": 1}}'
        events = self._feed_chars(engine, malformed + "</tool_call>")

        arg_chunks = [e.value for e in events if e.type == EventType.ARG_VALUE_CHUNK]
        assert len(arg_chunks) > 1, (
            "Content after stray } should still stream incrementally"
        )
        arg_text = "".join(arg_chunks)
        assert '"a": 1' in arg_text

    def test_json_args_no_premature_close_brace(self):
        """Closing braces of the top-level JSON shouldn't be streamed
        until confirmed by the end tag."""
        engine = StreamingParserEngine(_hermes_config(), tokenizer=None)

        engine.feed("<tool_call>", [])
        events = engine.feed('{"name": "f"}', [])

        arg_text = "".join(
            e.value for e in events if e.type == EventType.ARG_VALUE_CHUNK
        )
        assert "}" not in arg_text, "Top-level } should be held back"

        events2 = engine.feed("</tool_call>", [])
        arg_text2 = "".join(
            e.value for e in events2 if e.type == EventType.ARG_VALUE_CHUNK
        )
        assert "}" in arg_text2, "} should flush on end tag"


_START_ID = 50
_END_ID = 51
_TOOL_START_ID = 60
_TOOL_END_ID = 61


def _make_think_tokenizer():
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.get_vocab.return_value = {"<think>": _START_ID, "</think>": _END_ID}
    tok.decode.side_effect = lambda ids: {
        _START_ID: "<think>",
        _END_ID: "</think>",
    }.get(ids[0], f"tok{ids[0]}")
    return tok


def _make_hermes_tokenizer():
    """Tokenizer that resolves tool_call tags to special IDs."""
    _special = {_TOOL_START_ID: "<tool_call>", _TOOL_END_ID: "</tool_call>"}
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.get_vocab.return_value = {
        "<tool_call>": _TOOL_START_ID,
        "</tool_call>": _TOOL_END_ID,
    }
    tok.decode.side_effect = lambda ids: "".join(
        _special.get(i, chr(i) if i < 128 else f"<{i}>") for i in ids
    )
    return tok


class TestLexerBufferFlush:
    """Lexer buffer must be flushed before PreLexedTerminal transitions."""

    def test_buffered_prefix_emitted_in_current_state(self):
        """Text buffered by the lexer (e.g. '<') must be emitted as
        REASONING_CHUNK before THINK_END transitions to CONTENT."""
        engine = StreamingParserEngine(_think_config(), _make_think_tokenizer())

        events = engine.feed("<think>", [_START_ID])
        assert any(e.type == EventType.REASONING_START for e in events)

        events = engine.feed("reasoning text<", [])
        reasoning_text = "".join(
            e.value for e in events if e.type == EventType.REASONING_CHUNK
        )
        assert "reasoning text" in reasoning_text

        events = engine.feed("</think>", [_END_ID])
        event_types = [e.type for e in events]
        if EventType.REASONING_CHUNK in event_types:
            rc_idx = event_types.index(EventType.REASONING_CHUNK)
            re_idx = event_types.index(EventType.REASONING_END)
            assert rc_idx < re_idx, (
                "'<' must be emitted as REASONING_CHUNK before REASONING_END"
            )
            flushed = events[rc_idx].value
            assert "<" in flushed

    def test_empty_buffer_no_extra_events(self):
        """When the lexer buffer is empty, flushing is a no-op."""
        engine = StreamingParserEngine(_think_config(), _make_think_tokenizer())

        engine.feed("<think>", [_START_ID])
        engine.feed("clean text", [])

        events = engine.feed("</think>", [_END_ID])
        assert any(e.type == EventType.REASONING_END for e in events)
        chunk_events = [e for e in events if e.type == EventType.REASONING_CHUNK]
        assert all(e.value for e in chunk_events)


class TestTokenIdFiltering:
    """When token IDs are available, lex-matched terminals that also
    have token_id_terminal entries should be demoted to content."""

    def test_lex_matched_terminal_demoted_after_token_ids_seen(self):
        """After receiving token IDs, text that matches a token-ID
        terminal should be treated as content, not trigger a transition."""
        engine = StreamingParserEngine(_hermes_config(), _make_hermes_tokenizer())

        # First feed with a non-special token ID to set _ever_had_token_ids
        engine.feed("prefix ", [1])

        # Now feed text containing <tool_call> as literal text
        events = engine.feed(
            "Use <tool_call> to invoke tools.</tool_call>", [2, 3, 4, 5]
        )
        events.extend(engine.finish())

        types = [e.type for e in events]
        assert EventType.TOOL_CALL_START not in types
        assert EventType.TEXT_CHUNK in types

        text = "".join(e.value for e in events if e.type == EventType.TEXT_CHUNK)
        assert "<tool_call>" in text

    def test_scanner_matched_terminal_bypasses_filter(self):
        """PreLexedTerminals from the scanner bypass the filter and
        still trigger state transitions."""
        engine = StreamingParserEngine(_hermes_config(), _make_hermes_tokenizer())

        events = engine.feed("<tool_call>", [_TOOL_START_ID])
        assert any(e.type == EventType.TOOL_CALL_START for e in events)

        events = engine.feed('{"name": "f"}', [2, 3])
        events.extend(engine.feed("</tool_call>", [_TOOL_END_ID]))
        events.extend(engine.finish())
        assert any(e.type == EventType.TOOL_CALL_END for e in events)

    def test_no_filtering_without_token_ids(self):
        """When no token IDs are ever provided (non-streaming),
        text matching still triggers transitions."""
        engine = StreamingParserEngine(_hermes_config(), _make_hermes_tokenizer())

        events = engine.feed('<tool_call>{"name": "f"}</tool_call>', [])
        events.extend(engine.finish())

        types = [e.type for e in events]
        assert EventType.TOOL_CALL_START in types
        assert EventType.TOOL_CALL_END in types

    def test_mixed_text_then_real_tool_call(self):
        """Text mentioning tool syntax followed by a real special-token
        tool call."""
        engine = StreamingParserEngine(_hermes_config(), _make_hermes_tokenizer())

        events1 = engine.feed("Mention <tool_call> in text. ", [1, 2, 3, 4])
        events2 = engine.feed("<tool_call>", [_TOOL_START_ID])
        events3 = engine.feed('{"name": "a"}', [5, 6])
        events4 = engine.feed("</tool_call>", [_TOOL_END_ID])
        events4.extend(engine.finish())

        all_events = events1 + events2 + events3 + events4

        content = "".join(e.value for e in all_events if e.type == EventType.TEXT_CHUNK)
        assert "<tool_call>" in content

        assert sum(1 for e in all_events if e.type == EventType.TOOL_CALL_START) == 1
        assert sum(1 for e in all_events if e.type == EventType.TOOL_CALL_END) == 1


def _func_prefix_config() -> ParserEngineConfig:
    """Config mixing token-ID terminals (TOOL_START/END) with
    text-only terminals (FUNC_PREFIX) and fallback transitions."""
    return ParserEngineConfig(
        name="func_prefix_test",
        terminals={
            "TOOL_START": "<tool_call>",
            "TOOL_END": "</tool_call>",
            "FUNC_PREFIX": "<function=",
            "FUNC_END": "</function>",
            "CLOSE_ANGLE": ">",
        },
        token_id_terminals={
            "TOOL_START": "<tool_call>",
            "TOOL_END": "</tool_call>",
        },
        transitions={
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.CONTENT, "FUNC_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
                skip_in_token_id_mode=True,
            ),
            (ParserState.TOOL_PREAMBLE, "FUNC_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (),
            ),
            (ParserState.TOOL_NAME, "CLOSE_ANGLE"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
            (ParserState.TOOL_ARGS, "FUNC_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_BETWEEN, "FUNC_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
                skip_in_token_id_mode=True,
            ),
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
    )


def _make_func_prefix_tokenizer():
    return make_mock_tokenizer(
        {
            "<tool_call>": _TOOL_START_ID,
            "</tool_call>": _TOOL_END_ID,
        }
    )


class TestTextOnlyFallbackFiltering:
    """When token IDs are available, transitions marked
    skip_in_token_id_mode should be skipped."""

    def test_func_prefix_in_prose_demoted_in_strict_mode(self):
        """<function=get_time> in prose should NOT trigger a tool call
        when strict mode is active."""
        engine = StreamingParserEngine(
            _func_prefix_config(), _make_func_prefix_tokenizer()
        )
        engine.feed("prefix ", [1])

        events = engine.feed("Use <function=get_time> to check.", [2, 3, 4, 5])
        events.extend(engine.finish())

        types = [e.type for e in events]
        assert EventType.TOOL_CALL_START not in types
        assert EventType.TEXT_CHUNK in types
        text = "".join(e.value for e in events if e.type == EventType.TEXT_CHUNK)
        assert "<function=" in text

    def test_normal_flow_after_tool_start_still_works(self):
        """TOOL_START (special token) -> FUNC_PREFIX (text) should
        still parse a tool call normally in strict mode."""
        engine = StreamingParserEngine(
            _func_prefix_config(), _make_func_prefix_tokenizer()
        )

        events1 = engine.feed("<tool_call>", [_TOOL_START_ID])
        assert any(e.type == EventType.TOOL_CALL_START for e in events1)

        events2 = engine.feed("<function=get_weather>", [2, 3])
        events3 = engine.feed("args", [4])
        events4 = engine.feed("</function>", [5, 6])
        events4.extend(engine.feed("</tool_call>", [_TOOL_END_ID]))
        events4.extend(engine.finish())

        all_events = events1 + events2 + events3 + events4
        assert sum(1 for e in all_events if e.type == EventType.TOOL_CALL_START) == 1
        assert sum(1 for e in all_events if e.type == EventType.TOOL_CALL_END) == 1

    def test_fallback_fires_without_token_ids(self):
        """When no token IDs are provided, fallback transitions should
        still fire normally."""
        engine = StreamingParserEngine(
            _func_prefix_config(), _make_func_prefix_tokenizer()
        )

        events = engine.feed("<function=get_time>args</function>", [])
        events.extend(engine.finish())

        types = [e.type for e in events]
        assert EventType.TOOL_CALL_START in types
        assert EventType.TOOL_CALL_END in types

    def test_tool_between_fallback_blocked_in_strict_mode(self):
        """The (TOOL_BETWEEN, FUNC_PREFIX) fallback should also be
        blocked in strict mode."""
        engine = StreamingParserEngine(
            _func_prefix_config(), _make_func_prefix_tokenizer()
        )

        engine.feed("<tool_call>", [_TOOL_START_ID])
        engine.feed("<function=a>", [2, 3])
        engine.feed("args", [4])
        engine.feed("</function>", [5, 6])
        engine.feed("</tool_call>", [_TOOL_END_ID])

        events = engine.feed("<function=b>more</function>", [7, 8, 9])
        events.extend(engine.finish())

        types = [e.type for e in events]
        assert EventType.TOOL_CALL_START not in types


class TestNoUnusedTokenizerAttr:
    """StreamingParserEngine no longer stores a redundant _tokenizer."""

    def test_no_tokenizer_attribute(self):
        config = ParserEngineConfig(name="test")
        engine = StreamingParserEngine(config, tokenizer=None)
        assert not hasattr(engine, "_tokenizer")


class TestArgsResetOnReentry:
    """When leaving TOOL_ARGS and later re-entering (e.g. two tool
    calls), the entering-TOOL_ARGS block resets args tracking.  The
    redundant reset on exit was removed."""

    @staticmethod
    def _multi_tool_config() -> ParserEngineConfig:
        return ParserEngineConfig(
            name="multi_tool",
            terminals={
                "TOOL_START": "<tool_call>",
                "TOOL_END": "</tool_call>",
                "TOOL_SEP": "<tool_sep>",
            },
            transitions={
                (ParserState.CONTENT, "TOOL_START"): Transition(
                    ParserState.TOOL_ARGS,
                    (EventType.TOOL_CALL_START,),
                ),
                (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                    ParserState.TOOL_BETWEEN,
                    (EventType.TOOL_CALL_END,),
                ),
                (ParserState.TOOL_BETWEEN, "TOOL_SEP"): Transition(
                    ParserState.TOOL_ARGS,
                    (EventType.TOOL_CALL_START,),
                ),
            },
            content_events={
                ParserState.CONTENT: EventType.TEXT_CHUNK,
                ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
            },
        )

    def test_args_tracking_across_reentry(self):
        engine = StreamingParserEngine(self._multi_tool_config(), tokenizer=None)

        events = engine.feed(
            '<tool_call>{"city": "SF"}</tool_call>'
            "<tool_sep>"
            '{"name": "bar"}</tool_call>',
            [],
        )

        tool_starts = [e for e in events if e.type == EventType.TOOL_CALL_START]
        tool_ends = [e for e in events if e.type == EventType.TOOL_CALL_END]
        arg_chunks = [e for e in events if e.type == EventType.ARG_VALUE_CHUNK]

        assert len(tool_starts) == 2
        assert len(tool_ends) == 2
        assert tool_starts[0].tool_index == 0
        assert tool_starts[1].tool_index == 1

        first_args = "".join(e.value for e in arg_chunks if e.tool_index == 0)
        second_args = "".join(e.value for e in arg_chunks if e.tool_index == 1)
        assert '"city"' in first_args
        assert '"name"' in second_args

    def test_brace_depth_resets_on_reentry(self):
        """Verify _args_brace_depth resets when re-entering TOOL_ARGS."""
        engine = StreamingParserEngine(self._multi_tool_config(), tokenizer=None)
        engine.feed("<tool_call>", [])
        assert engine.state == ParserState.TOOL_ARGS
        assert engine._args_brace_depth == 0

        engine.feed('{"a": 1}', [])
        engine.feed("</tool_call>", [])
        assert engine.state == ParserState.TOOL_BETWEEN

        engine.feed("<tool_sep>", [])
        assert engine.state == ParserState.TOOL_ARGS
        assert engine._args_brace_depth == 0
        assert engine._args_in_string is False
        assert engine._args_escape_next is False


class TestToolPreambleFinish:
    """finish() in TOOL_PREAMBLE state emits TOOL_CALL_END when a tool
    call was started (tool_index >= 0), but not when tool_index is -1."""

    @staticmethod
    def _preamble_with_tool_call_start_config() -> ParserEngineConfig:
        return ParserEngineConfig(
            name="preamble_tcs",
            terminals={"TOOL_START": "<tool_call>"},
            transitions={
                (ParserState.CONTENT, "TOOL_START"): Transition(
                    ParserState.TOOL_PREAMBLE,
                    (EventType.TOOL_CALL_START,),
                ),
            },
            content_events={ParserState.CONTENT: EventType.TEXT_CHUNK},
        )

    @staticmethod
    def _preamble_without_tool_call_start_config() -> ParserEngineConfig:
        return ParserEngineConfig(
            name="preamble_no_tcs",
            terminals={"TOOL_CALLS_START": "<tool_calls>"},
            transitions={
                (ParserState.CONTENT, "TOOL_CALLS_START"): Transition(
                    ParserState.TOOL_PREAMBLE,
                    (),
                ),
            },
            content_events={ParserState.CONTENT: EventType.TEXT_CHUNK},
        )

    def test_finish_emits_tool_call_end_with_tool_index(self):
        config = self._preamble_with_tool_call_start_config()
        engine = StreamingParserEngine(config, tokenizer=None)

        engine.feed("<tool_call>", [])
        assert engine.state == ParserState.TOOL_PREAMBLE
        assert engine.tool_index == 0

        finish_events = engine.finish()
        end_events = [e for e in finish_events if e.type == EventType.TOOL_CALL_END]
        assert len(end_events) == 1
        assert end_events[0].tool_index == 0

    def test_finish_no_tool_call_end_without_tool_index(self):
        config = self._preamble_without_tool_call_start_config()
        engine = StreamingParserEngine(config, tokenizer=None)

        engine.feed("<tool_calls>", [])
        assert engine.state == ParserState.TOOL_PREAMBLE
        assert engine.tool_index == -1

        finish_events = engine.finish()
        end_events = [e for e in finish_events if e.type == EventType.TOOL_CALL_END]
        assert len(end_events) == 0
        assert engine.state == ParserState.CONTENT


class TestRegexTerminalInfraRemoved:
    """TerminalDef.priority, LexerShape.regex_terminals, and the regex
    matching loop were removed."""

    def test_terminal_def_no_priority(self):
        import regex as re

        td = TerminalDef(name="X", pattern=re.compile("x"))
        assert not hasattr(td, "priority")

    def test_lexer_shape_no_regex_terminals(self):
        shape = LexerShape([])
        assert not hasattr(shape, "regex_terminals")

    def test_terminals_from_literals_still_works(self):
        literals = {"TOOL_START": "<tool_call>", "TOOL_END": "</tool_call>"}
        defs = terminals_from_literals(literals)
        assert len(defs) == 2
        names = {d.name for d in defs}
        assert names == {"TOOL_START", "TOOL_END"}
        for d in defs:
            assert d.is_literal
            assert d.literal in ("<tool_call>", "</tool_call>")


class TestMultiCharTerminalInArgs:
    """Regression: multi-char terminals falling through in TOOL_ARGS
    must be fed char-by-char via _feed_args_text, not _feed_args_char."""

    @staticmethod
    def _newline_config() -> ParserEngineConfig:
        return ParserEngineConfig(
            name="newline_test",
            terminals={
                "TOOL_START": "<tool_call>",
                "TOOL_END": "</tool_call>",
                "NEWLINE": "\n",
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

    def test_newline_in_args_parsed_correctly(self):
        engine = StreamingParserEngine(self._newline_config(), tokenizer=None)
        text = '<tool_call>{"name": "f",\n"arguments": {"a": 1}}</tool_call>'
        events = engine.parse_complete(text)

        arg_text = "".join(
            e.value for e in events if e.type == EventType.ARG_VALUE_CHUNK
        )
        assert '"name": "f"' in arg_text
        assert '"arguments"' in arg_text

    def test_newline_in_args_streaming(self):
        engine = StreamingParserEngine(self._newline_config(), tokenizer=None)
        all_events = TestStreaming._feed_chars(
            engine, '<tool_call>{"name": "f",\n"a": 1}</tool_call>'
        )

        arg_text = "".join(
            e.value for e in all_events if e.type == EventType.ARG_VALUE_CHUNK
        )
        assert '"name": "f"' in arg_text
        assert '"a": 1' in arg_text


class TestSkipToolParsing:
    """When skip_tool_parsing is set, tool tags become content."""

    def test_tool_tags_emitted_as_content(self):
        engine = StreamingParserEngine(_hermes_config(), tokenizer=None)
        engine.skip_tool_parsing = True

        text = '<tool_call>{"name": "f"}</tool_call>'
        events = engine.parse_complete(text)

        types = [e.type for e in events]
        assert EventType.TOOL_CALL_START not in types
        assert EventType.TOOL_CALL_END not in types

        content = "".join(e.value for e in events if e.type == EventType.TEXT_CHUNK)
        assert "<tool_call>" in content
        assert "</tool_call>" in content

    def test_skip_tool_streaming(self):
        engine = StreamingParserEngine(_hermes_config(), tokenizer=None)
        engine.skip_tool_parsing = True

        all_events = TestStreaming._feed_chars(
            engine, '<tool_call>{"name": "f"}</tool_call>'
        )

        types = [e.type for e in all_events]
        assert EventType.TOOL_CALL_START not in types

        content = "".join(e.value for e in all_events if e.type == EventType.TEXT_CHUNK)
        assert "<tool_call>" in content

    def test_reset_preserves_skip_tool_parsing(self):
        engine = StreamingParserEngine(_hermes_config(), tokenizer=None)
        engine.skip_tool_parsing = True
        engine.reset()
        assert engine.skip_tool_parsing is True
