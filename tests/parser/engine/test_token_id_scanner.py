# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TokenIDScanner, focusing on hold-back text recovery.

Uses gemma4_config for all end-to-end engine tests, covering
reasoning channels, tool calls, and combined flows."""

from unittest.mock import MagicMock

import pytest

from vllm.parser.engine.events import EventType
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
from vllm.parser.engine.token_id_scanner import (
    PreLexedTerminal,
    TextChunk,
    TokenIDScanner,
)
from vllm.parser.gemma4 import gemma4_config

CHANNEL_START = "<|channel>"
CHANNEL_END = "<channel|>"
CHANNEL_START_ID = 100
CHANNEL_END_ID = 101
REGULAR_TOKEN_ID = 200
TOOL_START = "<tool_call>"
TOOL_END = "</tool_call>"
TOOL_START_ID = 110
TOOL_END_ID = 111


@pytest.fixture
def tokenizer():
    tok = MagicMock()
    tok.get_vocab.return_value = {
        CHANNEL_START: CHANNEL_START_ID,
        CHANNEL_END: CHANNEL_END_ID,
    }
    tok.decode.side_effect = lambda ids: {
        CHANNEL_START_ID: CHANNEL_START,
        CHANNEL_END_ID: CHANNEL_END,
        REGULAR_TOKEN_ID: "regular",
    }.get(ids[0], f"<unk:{ids[0]}>")
    return tok


@pytest.fixture
def scanner(tokenizer):
    return TokenIDScanner(
        token_id_to_terminal={
            CHANNEL_START_ID: "THINK_START",
            CHANNEL_END_ID: "THINK_END",
        },
        tokenizer=tokenizer,
    )


class TestJoinDecodedTextReturnsStr:
    """_join_decoded_text now returns str unconditionally (was
    str | None when an isinstance guard made a branch unreachable)."""

    @pytest.fixture
    def bare_scanner(self):
        return TokenIDScanner({}, tokenizer=None, drop_token_ids=set())

    def test_mixed_items(self, bare_scanner):
        items = [
            TextChunk("hello "),
            PreLexedTerminal("TOOL_START", 42, "<tool_call>"),
            TextChunk(" world"),
        ]
        result = bare_scanner._join_decoded_text(items)
        assert isinstance(result, str)
        assert result == "hello <tool_call> world"

    def test_empty_list(self, bare_scanner):
        result = bare_scanner._join_decoded_text([])
        assert isinstance(result, str)
        assert result == ""

    def test_only_text_chunks(self, bare_scanner):
        result = bare_scanner._join_decoded_text([TextChunk("abc"), TextChunk("def")])
        assert result == "abcdef"


class TestHoldbackTextRecovery:
    def test_holdback_text_with_special_token_text_absent(self, scanner):
        """delta_text has hold-back text but the special token's text is
        NOT in delta_text (held back by the detokenizer).  Terminal is
        deferred until the text arrives in a subsequent delta."""
        result = scanner.scan(
            delta_text="processed is appropriate.",
            delta_token_ids=[CHANNEL_END_ID],
        )

        assert len(result) == 0

        # Second scan: terminal text arrives (detokenizer flushes).
        # Deferred terminal resolves with holdback text before it.
        result2 = scanner.scan(
            delta_text="<channel|>Understood.",
            delta_token_ids=[20, 21],
        )
        pre_lexed = [r for r in result2 if isinstance(r, PreLexedTerminal)]
        assert len(pre_lexed) == 1
        assert pre_lexed[0].terminal == "THINK_END"
        texts = [r.text for r in result2 if isinstance(r, TextChunk)]
        combined = "".join(texts)
        assert "processed is appropriate." in combined
        assert "Understood." in combined

    def test_holdback_text_with_special_token_text_present(self, scanner):
        """delta_text includes hold-back text AND the special token text."""
        result = scanner.scan(
            delta_text="holdback text<channel|>",
            delta_token_ids=[CHANNEL_END_ID],
        )

        assert len(result) == 2
        assert isinstance(result[0], TextChunk)
        assert result[0].text == "holdback text"
        assert isinstance(result[1], PreLexedTerminal)
        assert result[1].terminal == "THINK_END"

    def test_no_holdback_text(self, scanner):
        """delta_text is exactly the special token text — no hold-back."""
        result = scanner.scan(
            delta_text="<channel|>",
            delta_token_ids=[CHANNEL_END_ID],
        )

        assert len(result) == 1
        assert isinstance(result[0], PreLexedTerminal)
        assert result[0].terminal == "THINK_END"

    def test_empty_delta_text(self, scanner):
        """delta_text is empty — terminal deferred until text arrives."""
        result = scanner.scan(
            delta_text="",
            delta_token_ids=[CHANNEL_END_ID],
        )

        assert len(result) == 0

        flushed = scanner.flush_pending()
        assert len(flushed) == 1
        assert isinstance(flushed[0], PreLexedTerminal)
        assert flushed[0].terminal == "THINK_END"

    def test_empty_delta_text_drops_individual_decode_text(self, tokenizer):
        """delta_text="" with multiple tokens including special: all
        results deferred — individually-decoded TextChunks are unreliable
        and PreLexedTerminals wait for text confirmation."""
        tool_start_id = 400
        tok_a = 201
        tok_b = 202
        tokenizer.decode.side_effect = lambda ids: {
            tool_start_id: "<|tool_call>",
            tok_a: "call:",
            tok_b: "get_weather",
        }.get(ids[0], "?")

        scanner = TokenIDScanner(
            token_id_to_terminal={tool_start_id: "TOOL_START"},
            tokenizer=tokenizer,
        )

        result = scanner.scan(
            delta_text="",
            delta_token_ids=[tool_start_id, tok_a, tok_b],
        )

        assert len(result) == 0

        flushed = scanner.flush_pending()
        assert len(flushed) == 1
        assert isinstance(flushed[0], PreLexedTerminal)
        assert flushed[0].terminal == "TOOL_START"

    def test_holdback_before_start_tag(self, scanner):
        """Hold-back text before a reasoning start tag."""
        result = scanner.scan(
            delta_text="prefix text<|channel>",
            delta_token_ids=[CHANNEL_START_ID],
        )

        assert len(result) == 2
        assert isinstance(result[0], TextChunk)
        assert result[0].text == "prefix text"
        assert isinstance(result[1], PreLexedTerminal)
        assert result[1].terminal == "THINK_START"

    def test_multi_token_batch_special_in_middle(self, scanner, tokenizer):
        """Stream-interval > 1: batch has regular tokens + special token.
        delta_text differs from individual decodes (context-dependent)."""
        tok_a = 201
        tok_b = 202
        tokenizer.decode.side_effect = lambda ids: {
            tok_a: "wordA",
            tok_b: "wordB",
            CHANNEL_END_ID: CHANNEL_END,
        }.get(ids[0], "?")

        scanner_multi = TokenIDScanner(
            token_id_to_terminal={CHANNEL_END_ID: "THINK_END"},
            tokenizer=tokenizer,
        )

        result = scanner_multi.scan(
            delta_text="holdback wordA<channel|> wordB",
            delta_token_ids=[tok_a, CHANNEL_END_ID, tok_b],
        )

        texts = [r.text for r in result if isinstance(r, TextChunk)]
        terminals = [r.terminal for r in result if isinstance(r, PreLexedTerminal)]
        assert "THINK_END" in terminals
        assert "holdback wordA" in "".join(texts)

    def test_multi_token_batch_special_token_text_absent(self, scanner, tokenizer):
        """Stream-interval > 1: batch has regular + special token, but
        delta_text doesn't contain the special token text at all
        (held back by detokenizer along with trailing regular tokens).
        Terminal is deferred until text arrives."""
        tok_a = 201
        tok_b = 202
        tokenizer.decode.side_effect = lambda ids: {
            tok_a: "alpha",
            tok_b: "beta",
            CHANNEL_END_ID: CHANNEL_END,
        }.get(ids[0], "?")

        scanner_multi = TokenIDScanner(
            token_id_to_terminal={CHANNEL_END_ID: "THINK_END"},
            tokenizer=tokenizer,
        )

        result = scanner_multi.scan(
            delta_text="holdback alpha",
            delta_token_ids=[tok_a, CHANNEL_END_ID, tok_b],
        )

        assert len(result) == 0

        # Next delta: terminal text arrives (detokenizer flushes).
        # Deferred terminal resolves with holdback text before it.
        result2 = scanner_multi.scan(
            delta_text="<channel|> more text",
            delta_token_ids=[300],
        )
        pre_lexed = [r for r in result2 if isinstance(r, PreLexedTerminal)]
        assert len(pre_lexed) == 1
        assert pre_lexed[0].terminal == "THINK_END"
        text_chunks = [r for r in result2 if isinstance(r, TextChunk)]
        combined = "".join(t.text for t in text_chunks)
        assert "holdback alpha" in combined
        assert "more text" in combined

    def test_holdback_with_content_after_special_token(self, tokenizer):
        """delta_text has hold-back + special token + content after,
        with corresponding token IDs for all parts."""
        tok_content = 210
        tokenizer.decode.side_effect = lambda ids: {
            CHANNEL_END_ID: CHANNEL_END,
            tok_content: "content start",
        }.get(ids[0], "?")

        scanner = TokenIDScanner(
            token_id_to_terminal={CHANNEL_END_ID: "THINK_END"},
            tokenizer=tokenizer,
        )

        result = scanner.scan(
            delta_text="reasoning end.<channel|>content start",
            delta_token_ids=[CHANNEL_END_ID, tok_content],
        )

        pre_lexed = [r for r in result if isinstance(r, PreLexedTerminal)]
        assert len(pre_lexed) == 1
        assert pre_lexed[0].terminal == "THINK_END"

        text_chunks = [r for r in result if isinstance(r, TextChunk)]
        combined = "".join(t.text for t in text_chunks)
        assert "reasoning end." in combined


class TestDropTokens:
    def test_drop_token_with_holdback(self, tokenizer):
        """Drop tokens stripped from delta_text, hold-back text preserved.
        Terminal is deferred when its text is absent from delta_text."""
        drop_id = 300
        tokenizer.decode.side_effect = lambda ids: {
            CHANNEL_END_ID: CHANNEL_END,
            drop_id: "<eos>",
        }.get(ids[0], "?")

        scanner = TokenIDScanner(
            token_id_to_terminal={CHANNEL_END_ID: "THINK_END"},
            tokenizer=tokenizer,
            drop_token_ids={drop_id},
        )

        result = scanner.scan(
            delta_text="holdback<eos>",
            delta_token_ids=[drop_id, CHANNEL_END_ID],
        )

        assert len(result) == 0

        # Terminal text arrives in next delta; deferred terminal resolves.
        result2 = scanner.scan(
            delta_text="<channel|>content",
            delta_token_ids=[20],
        )
        pre_lexed = [r for r in result2 if isinstance(r, PreLexedTerminal)]
        assert len(pre_lexed) == 1
        assert pre_lexed[0].terminal == "THINK_END"
        texts = [r.text for r in result2 if isinstance(r, TextChunk)]
        combined = "".join(texts)
        assert "holdback" in combined
        assert "<eos>" not in combined

        assert len(scanner.flush_pending()) == 0


class TestEndToEndReasoningHoldback:
    """End-to-end tests through the full parser engine simulating
    stream-interval > 1 and detokenizer hold-back, using
    gemma4_config."""

    def test_reasoning_content_not_truncated(self):
        config = gemma4_config()
        tok = MagicMock()
        vocab = {
            CHANNEL_START: CHANNEL_START_ID,
            CHANNEL_END: CHANNEL_END_ID,
        }
        tok.get_vocab.return_value = vocab
        tok.decode.side_effect = lambda ids: {
            CHANNEL_START_ID: CHANNEL_START,
            CHANNEL_END_ID: CHANNEL_END,
        }.get(ids[0], f"tok{ids[0]}")

        engine = StreamingParserEngine(config, tok)
        all_events = []

        # Delta 1: channel start token (text includes start tag)
        all_events.extend(engine.feed(CHANNEL_START, [CHANNEL_START_ID]))

        # Delta 2: reasoning text (normal content, no special tokens)
        all_events.extend(
            engine.feed(
                "thought\nThe request was received and ",
                [10, 11, 12, 13, 14],
            )
        )

        # Delta 3: MORE reasoning text, the detokenizer held some back.
        # Then channel end token arrives in token_ids, but its text
        # is NOT in delta_text (held back by detokenizer).
        # delta_text = previously held-back reasoning text only.
        all_events.extend(
            engine.feed(
                "processed is appropriate.",
                [CHANNEL_END_ID],
            )
        )

        # Delta 4: detokenizer flushes held-back channel end text
        # plus new content tokens.
        all_events.extend(
            engine.feed(
                "<channel|>Understood.",
                [20, 21],
            )
        )

        all_events.extend(engine.finish())

        reasoning_text = "".join(
            e.value for e in all_events if e.type == EventType.REASONING_CHUNK
        )
        content_text = "".join(
            e.value for e in all_events if e.type == EventType.TEXT_CHUNK
        )

        assert "processed is appropriate." in reasoning_text
        assert "Understood." in content_text

    def test_backtick_content_not_truncated(self):
        """Reproduces the hostname backtick truncation case."""
        config = gemma4_config()
        tok = MagicMock()
        vocab = {
            CHANNEL_START: CHANNEL_START_ID,
            CHANNEL_END: CHANNEL_END_ID,
        }
        tok.get_vocab.return_value = vocab
        tok.decode.side_effect = lambda ids: {
            CHANNEL_START_ID: CHANNEL_START,
            CHANNEL_END_ID: CHANNEL_END,
        }.get(ids[0], f"tok{ids[0]}")

        engine = StreamingParserEngine(config, tok)
        all_events = []

        all_events.extend(engine.feed(CHANNEL_START, [CHANNEL_START_ID]))
        all_events.extend(
            engine.feed(
                "thought\n1/10 completed. Next: ",
                [10, 11, 12, 13],
            )
        )

        # Hold-back text includes backtick content; channel end text
        # absent from delta_text.
        all_events.extend(
            engine.feed(
                "`hostname`.\n",
                [CHANNEL_END_ID],
            )
        )

        # Next delta flushes channel end + tool call start
        all_events.extend(
            engine.feed(
                "<channel|>tool output",
                [20, 21],
            )
        )

        all_events.extend(engine.finish())

        reasoning_text = "".join(
            e.value for e in all_events if e.type == EventType.REASONING_CHUNK
        )

        assert "`hostname`." in reasoning_text


# ---------------------------------------------------------------------------
# Token IDs for multi-token boundary tests
# ---------------------------------------------------------------------------
_CHANNEL_START_TAG = "<|channel>"
_CHANNEL_END_TAG = "<channel|>"
_TOOL_START_TAG = "<|tool_call>"
_TOOL_END_TAG = "<tool_call|>"
_QUOTE_TAG = '<|"|>'

_CHANNEL_START_TID = 100
_CHANNEL_END_TID = 101
_TOOL_START_TID = 102
_TOOL_END_TID = 103
_QUOTE_TID = 104
_TOK = list(range(200, 215))


def _gemma4_vocab() -> dict[str, int]:
    """Vocab mapping for Gemma4 special tokens."""
    return {
        _CHANNEL_START_TAG: _CHANNEL_START_TID,
        _CHANNEL_END_TAG: _CHANNEL_END_TID,
        _TOOL_START_TAG: _TOOL_START_TID,
        _TOOL_END_TAG: _TOOL_END_TID,
        _QUOTE_TAG: _QUOTE_TID,
    }


def _make_gemma4_tokenizer(
    extra_decode: dict[int, str] | None = None,
) -> MagicMock:
    """Mock tokenizer for Gemma4 with configurable regular-token text."""
    special = {
        _CHANNEL_START_TID: _CHANNEL_START_TAG,
        _CHANNEL_END_TID: _CHANNEL_END_TAG,
        _TOOL_START_TID: _TOOL_START_TAG,
        _TOOL_END_TID: _TOOL_END_TAG,
        _QUOTE_TID: _QUOTE_TAG,
    }
    decode_map = {**special, **(extra_decode or {})}

    tok = MagicMock()
    tok.get_vocab.return_value = _gemma4_vocab()
    tok.decode.side_effect = lambda ids: decode_map.get(ids[0], f"tok{ids[0]}")
    return tok


def _collect_events(engine, deltas):
    """Feed all (delta_text, delta_token_ids) pairs and return events."""
    from vllm.parser.engine.events import SemanticEvent

    all_events: list[SemanticEvent] = []
    for delta_text, delta_token_ids in deltas:
        all_events.extend(engine.feed(delta_text, delta_token_ids))
    all_events.extend(engine.finish())
    return all_events


def _reasoning_text(events) -> str:
    return "".join(e.value for e in events if e.type == EventType.REASONING_CHUNK)


def _content_text(events) -> str:
    return "".join(e.value for e in events if e.type == EventType.TEXT_CHUNK)


def _arg_text(events) -> str:
    return "".join(e.value for e in events if e.type == EventType.ARG_VALUE_CHUNK)


def _has_event(events, event_type) -> bool:
    return any(e.type == event_type for e in events)


class TestMultiTokenBoundaryPreservation:
    """End-to-end tests verifying no text is lost at state boundaries
    when multiple tokens arrive per delta with detokenizer holdback.

    Uses gemma4_config which covers both reasoning and tool calls
    in a single engine."""

    # -- Unique edge cases from channel-only tests -------------------------

    def test_empty_delta_text_at_channel_end_unified(self):
        """delta_text="" when CHANNEL_END arrives; text comes later.

        When delta_text is empty the PreLexedTerminal fires immediately.
        The tag text then appears in the *next* delta's delta_text and
        may be echoed by the lexer — that is accepted.  The invariant
        we enforce is that no reasoning or content text is *lost*."""
        tok = _make_gemma4_tokenizer()
        engine = StreamingParserEngine(gemma4_config(), tok)

        events = _collect_events(
            engine,
            [
                # CHANNEL_START with empty delta_text (detokenizer hasn't
                # flushed yet).
                ("", [_CHANNEL_START_TID]),
                # Detokenizer flushes start tag text + reasoning.
                ("<|channel>thought\nSome reasoning.", [_TOK[0], _TOK[1]]),
                # CHANNEL_END with empty delta_text.
                ("", [_CHANNEL_END_TID]),
                # Detokenizer flushes end tag text + content.
                ("<channel|>Final answer.", [_TOK[2], _TOK[3]]),
            ],
        )

        reasoning = _reasoning_text(events)
        content = _content_text(events)
        assert "Some reasoning." in reasoning
        assert "Final answer." in content
        assert _has_event(events, EventType.REASONING_START)
        assert _has_event(events, EventType.REASONING_END)

    def test_deferred_channel_end_flushed_at_finish_unified(self):
        """Deferred CHANNEL_END flushed at end-of-stream via finish()."""
        tok = _make_gemma4_tokenizer()
        engine = StreamingParserEngine(gemma4_config(), tok)

        events = _collect_events(
            engine,
            [
                (_CHANNEL_START_TAG, [_CHANNEL_START_TID]),
                ("thought\nReasoning text.", [_TOK[0]]),
                # Holdback + deferred — no more deltas after this.
                (" Final thought.", [_CHANNEL_END_TID]),
            ],
        )

        reasoning = _reasoning_text(events)
        assert "Reasoning text. Final thought." in reasoning
        assert _has_event(events, EventType.REASONING_END)

    # -- Cross-engine: reasoning → tool call in single unified engine -----

    def test_reasoning_to_tool_call_handoff_unified(self):
        """Full reasoning → content → tool call through a single engine.

        Verifies the unified config handles the complete flow without
        needing separate reasoning and tool-call engines."""
        tok = _make_gemma4_tokenizer()
        engine = StreamingParserEngine(gemma4_config(), tok)

        events = _collect_events(
            engine,
            [
                # Reasoning section
                (_CHANNEL_START_TAG, [_CHANNEL_START_TID]),
                ("thought\nI need to check the weather.", [_TOK[0], _TOK[1], _TOK[2]]),
                (_CHANNEL_END_TAG, [_CHANNEL_END_TID]),
                # Content + tool call
                ("Let me call a tool.", [_TOK[3], _TOK[4]]),
                (_TOOL_START_TAG, [_TOOL_START_TID]),
                ("call:get_weather{city:", [_TOK[5], _TOK[6]]),
                ('<|"|>SF<|"|>}', [_QUOTE_TID, _TOK[7], _QUOTE_TID, _TOK[8]]),
                (_TOOL_END_TAG, [_TOOL_END_TID]),
            ],
        )

        reasoning = _reasoning_text(events)
        content = _content_text(events)

        assert "I need to check the weather." in reasoning
        assert "Let me call a tool." in content
        assert _has_event(events, EventType.REASONING_START)
        assert _has_event(events, EventType.REASONING_END)
        assert _has_event(events, EventType.TOOL_CALL_START)
        assert _has_event(events, EventType.TOOL_CALL_END)
        assert "SF" in _arg_text(events)

    def test_multiple_tool_calls_rapid_transitions_unified(self):
        """Two back-to-back tool calls in the unified config.

        Verifies tool_index tracking and text integrity — the key behavior
        lost when test_multiple_tool_calls_rapid_transitions was removed."""
        tok = _make_gemma4_tokenizer()
        engine = StreamingParserEngine(gemma4_config(), tok)

        events = _collect_events(
            engine,
            [
                # Tool call 0
                (_TOOL_START_TAG, [_TOOL_START_TID]),
                ("call:get_weather{city:", [_TOK[0], _TOK[1]]),
                ('<|"|>NYC<|"|>}', [_QUOTE_TID, _TOK[2], _QUOTE_TID, _TOK[3]]),
                (_TOOL_END_TAG, [_TOOL_END_TID]),
                # Tool call 1 immediately after
                (_TOOL_START_TAG, [_TOOL_START_TID]),
                ("call:get_time{tz:", [_TOK[4], _TOK[5]]),
                ('<|"|>EST<|"|>}', [_QUOTE_TID, _TOK[6], _QUOTE_TID, _TOK[7]]),
                (_TOOL_END_TAG, [_TOOL_END_TID]),
            ],
        )

        starts = [e for e in events if e.type == EventType.TOOL_CALL_START]
        ends = [e for e in events if e.type == EventType.TOOL_CALL_END]
        assert len(starts) == 2
        assert len(ends) == 2
        assert starts[0].tool_index == 0
        assert starts[1].tool_index == 1

        names = "".join(e.value for e in events if e.type == EventType.TOOL_NAME)
        assert "get_weather" in names
        assert "get_time" in names

    def test_deferred_channel_end_before_tool_call_unified(self):
        """CHANNEL_END deferred (text held back), then tool call follows.

        Covers the case where reasoning ends with holdback at <channel|>
        and a tool call fires in the same unified engine afterward."""
        tok = _make_gemma4_tokenizer()
        engine = StreamingParserEngine(gemma4_config(), tok)

        events = _collect_events(
            engine,
            [
                (_CHANNEL_START_TAG, [_CHANNEL_START_TID]),
                ("thought\nNeed to call a tool.", [_TOK[0], _TOK[1]]),
                # Holdback: reasoning tail in delta_text, CHANNEL_END text absent.
                (" Let me proceed.", [_CHANNEL_END_TID]),
                # Deferred CHANNEL_END resolves.
                (_CHANNEL_END_TAG, [_TOK[2]]),
                # Tool call follows
                (_TOOL_START_TAG, [_TOOL_START_TID]),
                ("call:get_weather{city:", [_TOK[3], _TOK[4]]),
                ('<|"|>Tokyo<|"|>}', [_QUOTE_TID, _TOK[5], _QUOTE_TID, _TOK[6]]),
                (_TOOL_END_TAG, [_TOOL_END_TID]),
            ],
        )

        reasoning = _reasoning_text(events)
        assert "Need to call a tool. Let me proceed." in reasoning
        assert _has_event(events, EventType.REASONING_END)
        assert _has_event(events, EventType.TOOL_CALL_START)
        assert _has_event(events, EventType.TOOL_CALL_END)
        assert "Tokyo" in _arg_text(events)


class TestStreamInterval10:
    """Tests that model ``--stream-interval 10`` behavior.

    With stream_interval=10 the output processor holds tokens until 10
    have accumulated, then emits them all at once.  ``delta_token_ids``
    contains ~10 token IDs and ``delta_text`` is a substring of the
    detokenizer's accumulated output — it includes hold-back text from
    *previous* batches and may or may not include special-token text.

    The critical difference from interval=1: a special token can land
    in the *middle* of a 10-token batch, meaning tokens before it belong
    to one parser state and tokens after belong to another, all arriving
    in a single ``feed()`` call."""

    def test_channel_end_mid_batch_text_present(self):
        """<channel|> lands at position 4 of a 10-token batch.

        delta_text includes all text: holdback from previous batch +
        reasoning text + <channel|> text + content text.  All in one
        feed() call with 10 token IDs."""
        tok = _make_gemma4_tokenizer({_TOK[i]: f"word{i} " for i in range(15)})
        engine = StreamingParserEngine(gemma4_config(), tok)

        events: list = []
        # Batch 1: channel start + first reasoning tokens (10 tokens)
        events.extend(
            engine.feed(
                "<|channel>thought\nword0 word1 word2 word3 word4 "
                "word5 word6 word7 word8 ",
                [
                    _CHANNEL_START_TID,
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                    _TOK[3],
                    _TOK[4],
                    _TOK[5],
                    _TOK[6],
                    _TOK[7],
                    _TOK[8],
                ],
            )
        )

        # Batch 2: 10 tokens, <channel|> at position 4.
        # delta_text includes holdback from previous batch ("word9 ")
        # + reasoning tokens + <channel|> + content tokens.
        events.extend(
            engine.feed(
                "word9 word10 word11 <channel|>word12 word13 word14 word0 word1 word2 ",
                [
                    _TOK[9],
                    _TOK[10],
                    _TOK[11],
                    _CHANNEL_END_TID,
                    _TOK[12],
                    _TOK[13],
                    _TOK[14],
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                ],
            )
        )

        events.extend(engine.finish())

        reasoning = _reasoning_text(events)
        content = _content_text(events)

        # Reasoning must include all text up to <channel|>.
        for w in ("word9", "word10", "word11"):
            assert w in reasoning, f"{w!r} missing from reasoning"

        # Content must include all text after <channel|>.
        for w in ("word12", "word13", "word14"):
            assert w in content, f"{w!r} missing from content"

        assert _has_event(events, EventType.REASONING_END)

    def test_channel_end_and_tool_start_same_batch_unified(self):
        """Both <channel|> AND <|tool_call> in a single 10-token batch,
        handled by the unified config in one engine."""
        tok = _make_gemma4_tokenizer({_TOK[i]: f"w{i} " for i in range(15)})
        engine = StreamingParserEngine(gemma4_config(), tok)

        events: list = []

        # Batch 1: reasoning start + content (10 tokens)
        events.extend(
            engine.feed(
                "<|channel>thought\nw0 w1 w2 w3 w4 w5 w6 w7 w8 ",
                [
                    _CHANNEL_START_TID,
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                    _TOK[3],
                    _TOK[4],
                    _TOK[5],
                    _TOK[6],
                    _TOK[7],
                    _TOK[8],
                ],
            )
        )

        # Batch 2: 10 tokens with <channel|> at pos 2, <|tool_call> at pos 4.
        # Unified engine handles both the reasoning end and tool call start.
        events.extend(
            engine.feed(
                "w9 w10 <channel|>w11 <|tool_call>",
                [
                    _TOK[9],
                    _TOK[10],
                    _CHANNEL_END_TID,
                    _TOK[11],
                    _TOOL_START_TID,
                    _TOK[12],
                    _TOK[13],
                    _TOK[14],
                    _TOK[0],
                    _TOK[1],
                ],
            )
        )
        events.extend(engine.finish())

        reasoning = _reasoning_text(events)

        assert "w9" in reasoning
        assert "w10" in reasoning
        assert _has_event(events, EventType.REASONING_END)
        assert _has_event(events, EventType.TOOL_CALL_START)

    def test_channel_end_mid_batch_text_absent(self):
        """<channel|> at position 4 of 10-token batch, but its text is
        NOT in delta_text — detokenizer held it back.

        The terminal is deferred, and tokens after it in the same batch
        have their individually-decoded text dropped (unreliable without
        delta_text confirmation)."""
        tok = _make_gemma4_tokenizer({_TOK[i]: f"word{i} " for i in range(15)})
        engine = StreamingParserEngine(gemma4_config(), tok)

        events: list = []
        # Batch 1: channel start + reasoning (10 tokens)
        events.extend(
            engine.feed(
                "<|channel>thought\nword0 word1 word2 word3 word4 "
                "word5 word6 word7 word8 ",
                [
                    _CHANNEL_START_TID,
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                    _TOK[3],
                    _TOK[4],
                    _TOK[5],
                    _TOK[6],
                    _TOK[7],
                    _TOK[8],
                ],
            )
        )

        # Batch 2: 10 tokens, <channel|> at position 4.
        # delta_text has holdback + reasoning text but NOT the <channel|>
        # text or anything after — detokenizer held those back.
        events.extend(
            engine.feed(
                "word9 word10 word11 ",
                [
                    _TOK[9],
                    _TOK[10],
                    _TOK[11],
                    _CHANNEL_END_TID,
                    _TOK[12],
                    _TOK[13],
                    _TOK[14],
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                ],
            )
        )

        # Batch 3: detokenizer flushes held-back text.
        events.extend(
            engine.feed(
                "<channel|>word12 word13 word14 word0 word1 word2 ",
                [_TOK[3], _TOK[4], _TOK[5]],
            )
        )

        events.extend(engine.finish())

        reasoning = _reasoning_text(events)
        content = _content_text(events)

        # All reasoning text preserved, including holdback "word9..word11".
        for w in ("word9", "word10", "word11"):
            assert w in reasoning, f"{w!r} missing from reasoning"

        # Content after <channel|> preserved.
        for w in ("word12", "word13", "word14"):
            assert w in content, f"{w!r} missing from content"

        assert _has_event(events, EventType.REASONING_END)

    def test_tool_end_mid_batch_text_absent_unified(self):
        """<tool_call|> at position 5 of 10-token batch, text absent.

        Same pattern as channel_end but for tool calls — verifies
        arg text isn't lost at tool-call end with large batches."""
        tok = _make_gemma4_tokenizer({_TOK[i]: f"w{i}" for i in range(15)})
        engine = StreamingParserEngine(gemma4_config(), tok)

        events: list = []
        events.extend(
            engine.feed(
                _CHANNEL_START_TAG,
                [_CHANNEL_START_TID],
            )
        )
        events.extend(
            engine.feed(
                "thought\nNeed a tool.",
                [_TOK[0], _TOK[1]],
            )
        )
        events.extend(
            engine.feed(
                _TOOL_START_TAG,
                [_TOOL_START_TID],
            )
        )
        events.extend(
            engine.feed(
                "call:get_weather{city:",
                [_TOK[2], _TOK[3], _TOK[4]],
            )
        )

        # Batch 2: 10 tokens, quote + value tokens + quote + close brace
        # + TOOL_END (text absent) + content tokens.
        # delta_text only has the arg text, not <tool_call|> or after.
        events.extend(
            engine.feed(
                '<|"|>San Francisco<|"|>}',
                [
                    _QUOTE_TID,
                    _TOK[5],
                    _TOK[6],
                    _QUOTE_TID,
                    _TOK[7],
                    _TOOL_END_TID,
                    _TOK[8],
                    _TOK[9],
                    _TOK[10],
                    _TOK[11],
                ],
            )
        )

        # Batch 3: detokenizer flushes <tool_call|> text + content.
        events.extend(
            engine.feed(
                "<tool_call|>w8w9w10w11w12",
                [_TOK[12], _TOK[13]],
            )
        )

        events.extend(engine.finish())

        assert _has_event(events, EventType.TOOL_CALL_END)
        assert "San Francisco" in _arg_text(events)

    def test_large_batch_holdback_spans_two_batches(self):
        """Realistic stream_interval=10: reasoning text accumulates
        across two 10-token batches, with <channel|> in the second
        batch and holdback from the first.

        This is the most realistic production scenario: the detokenizer
        has been accumulating text across multiple tokens, holds some
        back at the batch boundary, and the special token arrives in
        the next batch with the held-back text in delta_text."""
        tok = _make_gemma4_tokenizer({_TOK[i]: f"w{i} " for i in range(15)})
        engine = StreamingParserEngine(gemma4_config(), tok)

        events: list = []

        # Batch 1 (10 tokens): channel start + reasoning
        events.extend(
            engine.feed(
                "<|channel>thought\nThe user asked about machine learning "
                "and I need to think about the best approach to",
                [
                    _CHANNEL_START_TID,
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                    _TOK[3],
                    _TOK[4],
                    _TOK[5],
                    _TOK[6],
                    _TOK[7],
                    _TOK[8],
                ],
            )
        )

        # Batch 2 (10 tokens): holdback from batch 1 (" explain")
        # + more reasoning + <channel|> (text absent from delta_text)
        # + content tokens (text also absent).
        events.extend(
            engine.feed(
                " explain this complex topic. Let me organize my thoughts.",
                [
                    _TOK[9],
                    _TOK[10],
                    _TOK[11],
                    _TOK[12],
                    _TOK[13],
                    _TOK[14],
                    _CHANNEL_END_TID,
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                ],
            )
        )

        # Batch 3 (10 tokens): detokenizer flushes <channel|> text
        # + content from batch 2 + new content.
        events.extend(
            engine.feed(
                "<channel|>w0 w1 w2 Here is what I recommend: start with "
                "the fundamentals and build up from there.",
                [
                    _TOK[3],
                    _TOK[4],
                    _TOK[5],
                    _TOK[6],
                    _TOK[7],
                    _TOK[8],
                    _TOK[9],
                    _TOK[10],
                    _TOK[11],
                    _TOK[12],
                ],
            )
        )

        events.extend(engine.finish())

        reasoning = _reasoning_text(events)
        content = _content_text(events)

        # The held-back reasoning text must be preserved.
        assert "organize my thoughts." in reasoning
        assert "explain" in reasoning

        # Content after <channel|> must be present.
        assert "recommend" in content

        assert _has_event(events, EventType.REASONING_START)
        assert _has_event(events, EventType.REASONING_END)


class TestRebuildFromAnchorsLiteralLookalike:
    """When delta_text contains a literal mention of a special token's
    text before the real special token, _rebuild_from_anchors must
    anchor at the real occurrence, not the literal one."""

    @pytest.fixture
    def tool_scanner(self):
        tok = MagicMock()
        tok.get_vocab.return_value = {
            TOOL_START: TOOL_START_ID,
            TOOL_END: TOOL_END_ID,
        }
        tok.decode.side_effect = lambda ids: {
            TOOL_START_ID: TOOL_START,
            TOOL_END_ID: TOOL_END,
        }.get(ids[0], f"t{ids[0]}")
        return TokenIDScanner(
            {TOOL_START_ID: "TOOL_START", TOOL_END_ID: "TOOL_END"},
            tok,
        )

    def test_literal_before_real_anchor(self, tool_scanner):
        """Literal <tool_call> in prose followed by a real <tool_call>
        special token — the scanner must split at the real one."""
        delta_text = 'Use <tool_call> like this: <tool_call>{"name":"f"}</tool_call>'
        delta_token_ids = [1, 2, 3, 4, 5, TOOL_START_ID, 6, 7, TOOL_END_ID]
        items = tool_scanner.scan(delta_text, delta_token_ids)

        text_parts = [it.text for it in items if isinstance(it, TextChunk)]
        terminals = [it for it in items if isinstance(it, PreLexedTerminal)]

        assert len(terminals) == 2
        assert terminals[0].terminal == "TOOL_START"
        assert terminals[1].terminal == "TOOL_END"

        # The literal mention must appear in a text chunk, not be
        # consumed by the TOOL_START anchor.
        joined_text = "".join(text_parts)
        assert "<tool_call>" in joined_text
        assert '{"name":"f"}' in joined_text

    def test_multiple_tool_calls_with_literal_between(self, tool_scanner):
        """Two real tool calls with a literal mention between them."""
        delta_text = (
            '<tool_call>{"name":"a"}</tool_call>'
            " see <tool_call> syntax "
            '<tool_call>{"name":"b"}</tool_call>'
        )
        delta_token_ids = [
            TOOL_START_ID,
            1,
            TOOL_END_ID,
            2,
            3,
            4,
            TOOL_START_ID,
            5,
            TOOL_END_ID,
        ]
        items = tool_scanner.scan(delta_text, delta_token_ids)

        terminals = [it for it in items if isinstance(it, PreLexedTerminal)]
        assert len(terminals) == 4

        text_parts = [it.text for it in items if isinstance(it, TextChunk)]
        joined_text = "".join(text_parts)
        # The literal mention between the two real calls must be in text
        assert "<tool_call> syntax" in joined_text


class TestRebuildFromAnchorsCascadingDeferral:
    """When a middle anchor's text is absent from delta_text,
    only that anchor should be deferred — not subsequent ones
    with valid positions."""

    @pytest.fixture
    def bare_scanner(self):
        tok = MagicMock()
        tok.decode.side_effect = lambda ids: f"t{ids[0]}"
        return TokenIDScanner({}, tok)

    def test_middle_anchor_missing_does_not_cascade(self, bare_scanner):
        a = PreLexedTerminal("TOOL_START", TOOL_START_ID, TOOL_START)
        b = PreLexedTerminal("THINK_END", CHANNEL_END_ID, CHANNEL_END)
        c = PreLexedTerminal("TOOL_END", TOOL_END_ID, TOOL_END)
        delta_text = f"prefix{TOOL_START}middle{TOOL_END}suffix"
        results = [a, b, c]

        rebuilt = bare_scanner._rebuild_from_anchors(delta_text, results)

        terminals = [r for r in rebuilt if isinstance(r, PreLexedTerminal)]
        texts = [r for r in rebuilt if isinstance(r, TextChunk)]
        joined = "".join(t.text for t in texts)

        assert len(terminals) == 2
        assert terminals[0].terminal == "TOOL_START"
        assert terminals[1].terminal == "TOOL_END"
        assert "prefix" in joined
        assert "middle" in joined
        assert "suffix" in joined
        assert len(bare_scanner._deferred_terminals) == 1
        assert bare_scanner._deferred_terminals[0].terminal == "THINK_END"
        assert bare_scanner._deferred_post_text == ""

    def test_first_anchor_missing_rest_still_emitted(self, bare_scanner):
        a = PreLexedTerminal("THINK_END", CHANNEL_END_ID, CHANNEL_END)
        b = PreLexedTerminal("TOOL_START", TOOL_START_ID, TOOL_START)
        delta_text = f"text{TOOL_START}more"
        results = [a, b]

        rebuilt = bare_scanner._rebuild_from_anchors(delta_text, results)

        terminals = [r for r in rebuilt if isinstance(r, PreLexedTerminal)]
        assert len(terminals) == 1
        assert terminals[0].terminal == "TOOL_START"
        assert len(bare_scanner._deferred_terminals) == 1
        assert bare_scanner._deferred_terminals[0].terminal == "THINK_END"

    def test_last_anchor_missing_preceding_still_emitted(self, bare_scanner):
        a = PreLexedTerminal("TOOL_START", TOOL_START_ID, TOOL_START)
        b = PreLexedTerminal("THINK_END", CHANNEL_END_ID, CHANNEL_END)
        delta_text = f"text{TOOL_START}more"
        results = [a, b]

        rebuilt = bare_scanner._rebuild_from_anchors(delta_text, results)

        terminals = [r for r in rebuilt if isinstance(r, PreLexedTerminal)]
        assert len(terminals) == 1
        assert terminals[0].terminal == "TOOL_START"
        texts = [r for r in rebuilt if isinstance(r, TextChunk)]
        joined = "".join(t.text for t in texts)
        assert "text" in joined
        # "more" is deferred along with the missing terminal —
        # it will be resolved in the next scan when the terminal
        # text arrives.
        assert bare_scanner._deferred_post_text == "more"
        assert len(bare_scanner._deferred_terminals) == 1
        assert bare_scanner._deferred_terminals[0].terminal == "THINK_END"
