# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scan delta token IDs for special tokens and split the stream into
pre-lexed terminals and plain text chunks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class TextChunk:
    text: str


@dataclass(slots=True)
class PreLexedTerminal:
    terminal: str
    token_id: int
    text: str


LexerInput = TextChunk | PreLexedTerminal


class TokenIDScanner:
    """Maps special token IDs in the delta to terminals.

    Before text-based lexing happens, the scanner checks each token ID
    in the delta against a mapping of ``{token_id: terminal_name}``.
    Matched tokens are emitted as :class:`PreLexedTerminal` items;
    everything else is grouped into :class:`TextChunk` items for the
    incremental lexer to process.

    When a terminal's text is not yet in ``delta_text`` (held back by
    the detokenizer), the terminal is deferred until the text arrives
    in a subsequent delta.
    """

    def __init__(
        self,
        token_id_to_terminal: dict[int, str],
        tokenizer,
    ) -> None:
        self.token_id_to_terminal = token_id_to_terminal
        self.tokenizer = tokenizer
        self._token_text_cache: dict[int, str] = {}
        self._deferred_terminals: list[PreLexedTerminal] = []
        self._deferred_post_text: str = ""

    def reset(self) -> None:
        """Clear mutable state for reuse. Preserves the token text cache."""
        self._deferred_terminals.clear()
        self._deferred_post_text = ""

    def _decode_token(self, token_id: int) -> str:
        if token_id not in self._token_text_cache:
            self._token_text_cache[token_id] = self.tokenizer.decode([token_id])
        return self._token_text_cache[token_id]

    _EMPTY: tuple[LexerInput, ...] = ()

    def scan(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> Sequence[LexerInput]:
        prefix_items: list[LexerInput] = []
        effective_text = delta_text

        if self._deferred_terminals:
            prefix_items, effective_text = self._resolve_deferred(delta_text)

        if not self.token_id_to_terminal:
            if effective_text:
                prefix_items.append(TextChunk(effective_text))
            return prefix_items

        has_special = False
        token_id_to_terminal = self.token_id_to_terminal
        for tid in delta_token_ids:
            if tid in token_id_to_terminal:
                has_special = True
                break

        if not has_special:
            if effective_text:
                if not prefix_items:
                    return [TextChunk(effective_text)]
                prefix_items.append(TextChunk(effective_text))
            return prefix_items or self._EMPTY

        token_texts = [self._decode_token(tid) for tid in delta_token_ids]

        results: list[LexerInput] = []
        text_accum: list[str] = []

        for idx, tid in enumerate(delta_token_ids):
            terminal = self.token_id_to_terminal.get(tid)
            if terminal is not None:
                if text_accum:
                    joined = "".join(text_accum)
                    if joined:
                        results.append(TextChunk(joined))
                    text_accum.clear()
                results.append(PreLexedTerminal(terminal, tid, token_texts[idx]))
            else:
                text_accum.append(token_texts[idx])

        if text_accum:
            joined = "".join(text_accum)
            if joined:
                results.append(TextChunk(joined))

        if effective_text:
            results = self._recover_holdback_text(effective_text, results)
        else:
            # No detokenizer text to validate against — individually-decoded
            # TextChunks are unreliable (context-dependent decoding).
            # Defer PreLexedTerminals so the state machine doesn't
            # transition before the preceding text has arrived.  The
            # deferred terminals will be resolved against the actual
            # delta_text in a subsequent scan() or flushed by finish().
            for r in results:
                if isinstance(r, PreLexedTerminal):
                    self._deferred_terminals.append(r)
            results = []

        return prefix_items + results

    def flush_pending(self) -> list[LexerInput]:
        if not self._deferred_terminals and not self._deferred_post_text:
            return []
        results: list[LexerInput] = []
        if self._deferred_post_text:
            results.append(TextChunk(self._deferred_post_text))
            self._deferred_post_text = ""
        results.extend(self._deferred_terminals)
        self._deferred_terminals.clear()
        return results

    def _resolve_deferred(
        self,
        delta_text: str,
    ) -> tuple[list[LexerInput], str]:
        """Resolve deferred terminals against new delta_text.

        When a previous ``scan()`` deferred a terminal (its text hadn't
        arrived yet), the next delta's text should contain that terminal's
        text.  Split delta_text at the terminal boundary: text before
        belongs to the previous parser state, the terminal triggers the
        state transition, and text after belongs to the new state.

        Returns ``(prefix_items, remaining_text)`` where prefix_items
        are the resolved deferred terminals (with any preceding text)
        and remaining_text is the unconsumed portion of delta_text that
        should be scanned with the current delta's token IDs.
        """
        deferred = self._deferred_terminals
        self._deferred_terminals = []

        results: list[LexerInput] = []
        remaining = delta_text

        if self._deferred_post_text:
            remaining = self._deferred_post_text + remaining
            self._deferred_post_text = ""

        # Duplicate-text deferred terminals resolve left-to-right via
        # find(); correct when each terminal text appears once in sequence.
        for terminal in deferred:
            pos = remaining.find(terminal.text)
            if pos > 0:
                results.append(TextChunk(remaining[:pos]))
                results.append(terminal)
                remaining = remaining[pos + len(terminal.text) :]
            elif pos == 0:
                results.append(terminal)
                remaining = remaining[len(terminal.text) :]
            else:
                # Accumulate text until terminal text arrives —
                # only the terminal provides a reliable split point.
                if remaining:
                    self._deferred_post_text += remaining
                    remaining = ""
                self._deferred_terminals.append(terminal)

        return results, remaining

    def _recover_holdback_text(
        self,
        delta_text: str,
        results: list[LexerInput],
    ) -> list[LexerInput]:
        """Recover detokenizer hold-back text not in delta_token_ids.

        The detokenizer may flush previously held-back text in
        ``delta_text`` that has no corresponding token ID in
        ``delta_token_ids``.  This hold-back text always appears as a
        prefix of ``delta_text``.
        """
        if not results:
            return [TextChunk(delta_text)]

        reconstructed = self._join_decoded_text(results)

        if not reconstructed:
            return [TextChunk(delta_text)] + results

        pos = delta_text.find(reconstructed)
        if pos > 0:
            return [TextChunk(delta_text[:pos])] + results
        if pos == 0:
            return results

        # Fallback: SentencePiece context-dependent decoding mismatch.
        # Rebuild from delta_text using PreLexedTerminals as split anchors.
        return self._rebuild_from_anchors(delta_text, results)

    def _join_decoded_text(self, results: list[LexerInput]) -> str:
        """Join TextChunk and PreLexedTerminal text into one string."""
        parts: list[str] = []
        for item in results:
            if isinstance(item, (TextChunk, PreLexedTerminal)):
                parts.append(item.text)
        return "".join(parts)

    def _rebuild_from_anchors(
        self,
        delta_text: str,
        results: list[LexerInput],
    ) -> list[LexerInput]:
        """Rebuild results from delta_text using terminals as anchors.

        When context-dependent decoding creates a mismatch between
        individually-decoded tokens and delta_text, use
        PreLexedTerminals as split points and reallocate text from
        delta_text.  If a terminal's text is not found in delta_text,
        it is deferred to the next scan() call.

        Anchors are resolved right-to-left with ``rfind`` so that each
        anchor binds to the *rightmost* available occurrence of its
        text.  This prevents earlier literal lookalikes (e.g. a user
        mentioning ``<tool_call>`` in prose) from stealing the position
        of a real special-token anchor that appears later.

        If the same anchor text appears multiple times as real special
        tokens (not prose), the rightmost-first binding could misalign.
        In practice this doesn't happen: each special token ID maps to
        a distinct PreLexedTerminal, and duplicates in prose are resolved
        by the token-ID filtering layer above.
        """
        anchors = [item for item in results if isinstance(item, PreLexedTerminal)]
        if not anchors:
            return [TextChunk(delta_text)]

        # Resolve positions right-to-left: each anchor gets the
        # rightmost occurrence that is still before the next anchor.
        positions: list[int] = [-1] * len(anchors)
        search_end = len(delta_text)
        for i in range(len(anchors) - 1, -1, -1):
            pos = delta_text.rfind(anchors[i].text, 0, search_end)
            if pos >= 0:
                positions[i] = pos
                search_end = pos

        # Build results left-to-right using the resolved positions.
        new_results: list[LexerInput] = []
        consumed = 0
        for i, anchor in enumerate(anchors):
            pos = positions[i]
            if pos >= consumed:
                if pos > consumed:
                    new_results.append(TextChunk(delta_text[consumed:pos]))
                new_results.append(anchor)
                consumed = pos + len(anchor.text)
            else:
                has_later_valid = any(p >= 0 for p in positions[i + 1 :])
                if not has_later_valid and consumed < len(delta_text):
                    self._deferred_post_text += delta_text[consumed:]
                    consumed = len(delta_text)
                self._deferred_terminals.append(anchor)
        if consumed < len(delta_text):
            new_results.append(TextChunk(delta_text[consumed:]))
        return new_results
