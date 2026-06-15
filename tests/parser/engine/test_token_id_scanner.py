# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TokenIDScanner, focusing on hold-back text recovery.

Uses gemma4_config for all end-to-end engine tests, covering
reasoning channels, tool calls, and combined flows."""

from unittest.mock import MagicMock

import pytest

from vllm.parser.engine.events import EventType
from vllm.parser.engine.token_id_scanner import (
    PreLexedTerminal,
    TextChunk,
    TokenIDScanner,
)

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
        from vllm.parser.engine.parser_engine_config import (
            ParserEngineConfig,
            ParserState,
            Transition,
        )
        from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine

        config = ParserEngineConfig(
            name="test_channel",
            initial_state=ParserState.CONTENT,
            terminals={
                "THINK_START": CHANNEL_START,
                "THINK_END": CHANNEL_END,
            },
            token_id_terminals={
                "THINK_START": CHANNEL_START,
                "THINK_END": CHANNEL_END,
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
        from vllm.parser.engine.parser_engine_config import (
            ParserEngineConfig,
            ParserState,
            Transition,
        )
        from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine

        config = ParserEngineConfig(
            name="test_channel",
            initial_state=ParserState.CONTENT,
            terminals={
                "THINK_START": CHANNEL_START,
                "THINK_END": CHANNEL_END,
            },
            token_id_terminals={
                "THINK_START": CHANNEL_START,
                "THINK_END": CHANNEL_END,
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
