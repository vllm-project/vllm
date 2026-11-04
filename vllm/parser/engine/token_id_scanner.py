# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scan delta token IDs for special tokens and split the stream into
pre-lexed terminals and plain text chunks."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

DROP_TERMINAL = "__DROP__"
_BYTE_LEVEL_ALPHABET = frozenset(
    chr(code)
    for start, end in ((33, 127), (161, 173), (174, 324))
    for code in range(start, end)
)


@dataclass(slots=True)
class TextChunk:
    text: str
    token_texts: tuple[str, ...] = ()
    token_count: int = 0


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
        *,
        strict_token_spans: bool = False,
    ) -> None:
        self.token_id_to_terminal = token_id_to_terminal
        self.tokenizer = tokenizer
        self._token_text_cache: dict[int, str] = {}
        self._deferred_terminals: list[PreLexedTerminal] = []
        self._deferred_prefix_token_counts: list[int] = []
        self._deferred_trailing_token_count = 0
        self._deferred_post_text: str = ""
        self._strict_token_spans = strict_token_spans
        self._span_mode = False
        self._pending_ids: list[int] = []
        self._pending_text = ""
        self._byte_level = False
        self._added_ids: set[int] = set()
        self._token_byte_lengths: dict[int, int | None] = {}
        if strict_token_spans:
            backend = getattr(tokenizer, "backend_tokenizer", None)
            decoder: Any = getattr(backend, "decoder", None)
            try:
                self._byte_level = (
                    json.loads(decoder.__getstate__()).get("type") == "ByteLevel"
                )
                if self._byte_level:
                    self._added_ids = set(tokenizer.get_added_vocab().values())
            except (AttributeError, TypeError, ValueError):
                self._byte_level = False

    def _byte_prefix(self, token_ids: list[int], text: str) -> tuple[int, str] | None:
        byte_budget = len(text.encode("utf-8"))
        byte_count = 0
        prefix_length = 0
        for tid in token_ids:
            if tid not in self._token_byte_lengths:
                length = None
                try:
                    if tid in self._added_ids:
                        length = len(self._decode_token(tid).encode("utf-8"))
                    else:
                        piece = self.tokenizer.convert_ids_to_tokens(tid)
                        # ByteLevel vocab characters each represent one source byte;
                        # added tokens instead carry their literal UTF-8 spelling.
                        if isinstance(piece, str) and _BYTE_LEVEL_ALPHABET.issuperset(
                            piece
                        ):
                            length = len(piece)
                except (AttributeError, TypeError, ValueError, UnicodeError):
                    pass
                self._token_byte_lengths[tid] = length
            length = self._token_byte_lengths[tid]
            if length is None:
                prefix = (
                    self._decode_token(token_ids[0])
                    if len(token_ids) == 1
                    else self.tokenizer.decode(token_ids, skip_special_tokens=False)
                )
                if prefix and text.startswith(prefix):
                    return len(token_ids), prefix
                return None
            if byte_count + length > byte_budget:
                break
            byte_count += length
            prefix_length += 1
        # Replacement characters can expand invalid source bytes and hide holdback.
        if (
            prefix_length == len(token_ids)
            and byte_count == byte_budget
            and "\ufffd" not in text
        ):
            return prefix_length, text
        # A token boundary can bisect a four-byte UTF-8 character. Only the
        # decoded prefix proves alignment for a partial run or trailing text.
        for length in range(prefix_length, max(0, prefix_length - 4), -1):
            prefix = (
                self._decode_token(token_ids[0])
                if length == 1
                else self.tokenizer.decode(
                    token_ids[:length], skip_special_tokens=False
                )
            )
            if prefix and text.startswith(prefix):
                return length, prefix
        return None

    def reset(self) -> None:
        """Clear mutable state for reuse. Preserves the token text cache."""
        self._deferred_terminals.clear()
        self._deferred_prefix_token_counts.clear()
        self._deferred_trailing_token_count = 0
        self._deferred_post_text = ""
        self._span_mode = False
        self._pending_ids.clear()
        self._pending_text = ""

    @property
    def has_pending(self) -> bool:
        return bool(self._deferred_terminals or self._pending_ids or self._pending_text)

    def _scan_token_spans(
        self, delta_text: str, delta_token_ids: Sequence[int]
    ) -> list[LexerInput]:
        self._pending_ids.extend(delta_token_ids)
        self._pending_text += delta_text
        results: list[LexerInput] = []
        consumed = 0
        ids = self._pending_ids
        while consumed < len(ids):
            tid = ids[consumed]
            terminal = self.token_id_to_terminal.get(tid)
            if terminal is not None:
                text = self._decode_token(tid)
                if text and self._pending_text.startswith(text):
                    results.append(PreLexedTerminal(terminal, tid, text))
                    self._pending_text = self._pending_text[len(text) :]
                    consumed += 1
                    continue
                if terminal == DROP_TERMINAL and not text.startswith(
                    self._pending_text
                ):
                    consumed += 1
                    continue
                break

            if not self._pending_text:
                break
            end = consumed + 1
            while end < len(ids) and ids[end] not in self.token_id_to_terminal:
                end += 1
            run = ids[consumed:end]
            if self._byte_level:
                matched = self._byte_prefix(run, self._pending_text)
                if matched is None:
                    break
                length, text = matched
                run = run[:length]
            else:
                text = (
                    self._decode_token(run[0])
                    if len(run) == 1
                    else self.tokenizer.decode(run, skip_special_tokens=False)
                )
            if not self._pending_text.startswith(text):
                # Detokenizers may withhold text, including incomplete UTF-8.
                # Match source prefixes only; repeated marker spellings are
                # never used as positional anchors.
                for length in range(len(run) - 1, 0, -1):
                    prefix = self.tokenizer.decode(
                        run[:length], skip_special_tokens=False
                    )
                    if prefix and self._pending_text.startswith(prefix):
                        run = run[:length]
                        text = prefix
                        break
                else:
                    break
            if not text:
                break
            results.append(TextChunk(text, token_count=len(run)))
            consumed += len(run)
            self._pending_text = self._pending_text[len(text) :]
        if consumed:
            del ids[:consumed]
        return results

    def _decode_token(self, token_id: int) -> str:
        if token_id not in self._token_text_cache:
            token_text = self.tokenizer.decode([token_id])
            self._token_text_cache[token_id] = (
                token_text if isinstance(token_text, str) else ""
            )
        return self._token_text_cache[token_id]

    def _decode_tokens(self, token_ids: Sequence[int]) -> tuple[str, ...]:
        if self.tokenizer is None:
            return ()
        return tuple(self._decode_token(tid) for tid in token_ids)

    _EMPTY: tuple[LexerInput, ...] = ()

    def scan(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> Sequence[LexerInput]:
        if self._strict_token_spans and (delta_token_ids or self._span_mode):
            self._span_mode = True
            return self._scan_token_spans(delta_text, delta_token_ids)
        prefix_items: list[LexerInput] = []
        effective_text = delta_text
        deferred_trailing_count = 0

        if self._deferred_terminals:
            prefix_items, effective_text, deferred_trailing_count = (
                self._resolve_deferred(delta_text, len(delta_token_ids))
            )

        if not self.token_id_to_terminal:
            if effective_text:
                token_texts = self._decode_tokens(delta_token_ids)
                prefix_items.append(
                    TextChunk(
                        effective_text,
                        token_texts,
                        deferred_trailing_count + len(token_texts),
                    )
                )
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
                    token_texts = self._decode_tokens(delta_token_ids)
                    return [TextChunk(effective_text, token_texts, len(token_texts))]
                token_texts = self._decode_tokens(delta_token_ids)
                prefix_items.append(
                    TextChunk(
                        effective_text,
                        token_texts,
                        deferred_trailing_count + len(token_texts),
                    )
                )
            return prefix_items or self._EMPTY

        decoded_token_texts = [self._decode_token(tid) for tid in delta_token_ids]

        results: list[LexerInput] = []
        text_accum: list[str] = []
        token_text_accum: list[str] = []

        for idx, tid in enumerate(delta_token_ids):
            terminal = self.token_id_to_terminal.get(tid)
            if terminal is not None:
                if text_accum:
                    joined = "".join(text_accum)
                    if joined:
                        results.append(
                            TextChunk(
                                joined,
                                tuple(token_text_accum),
                                len(token_text_accum),
                            )
                        )
                    text_accum.clear()
                    token_text_accum.clear()
                results.append(
                    PreLexedTerminal(terminal, tid, decoded_token_texts[idx])
                )
            else:
                text_accum.append(decoded_token_texts[idx])
                token_text_accum.append(decoded_token_texts[idx])

        if text_accum:
            joined = "".join(text_accum)
            if joined:
                results.append(
                    TextChunk(joined, tuple(token_text_accum), len(token_text_accum))
                )

        if deferred_trailing_count:
            results.insert(0, TextChunk("", token_count=deferred_trailing_count))

        if effective_text:
            results = self._recover_holdback_text(effective_text, results)
        else:
            # No detokenizer text to validate against — individually-decoded
            # TextChunks are unreliable (context-dependent decoding).
            # Defer PreLexedTerminals so the state machine doesn't
            # transition before the preceding text has arrived.  The
            # deferred terminals will be resolved against the actual
            # delta_text in a subsequent scan() or flushed by finish().
            prefix_token_count = 0
            for r in results:
                if isinstance(r, TextChunk):
                    prefix_token_count += r.token_count
                if isinstance(r, PreLexedTerminal):
                    self._deferred_terminals.append(r)
                    self._deferred_prefix_token_counts.append(prefix_token_count)
                    prefix_token_count = 0
            self._deferred_trailing_token_count += prefix_token_count
            results = []

        return prefix_items + results

    def flush_pending(self) -> list[LexerInput]:
        if self._span_mode:
            aligned = self._scan_token_spans("", ())
            # Unaligned text has no provable token origin. Preserve the text
            # supplied by the detokenizer without inventing special tokens.
            if self._pending_text:
                aligned.append(TextChunk(self._pending_text))
            self._pending_ids.clear()
            self._pending_text = ""
            return aligned
        if not self._deferred_terminals and not self._deferred_post_text:
            return []
        results: list[LexerInput] = []
        if self._deferred_post_text:
            prefix_count = (
                self._deferred_prefix_token_counts[0]
                if self._deferred_prefix_token_counts
                else 0
            )
            results.append(
                TextChunk(self._deferred_post_text, token_count=prefix_count)
            )
            self._deferred_post_text = ""
        results.extend(self._deferred_terminals)
        if self._deferred_trailing_token_count:
            results.append(
                TextChunk("", token_count=self._deferred_trailing_token_count)
            )
        self._deferred_terminals.clear()
        self._deferred_prefix_token_counts.clear()
        self._deferred_trailing_token_count = 0
        return results

    def _resolve_deferred(
        self,
        delta_text: str,
        current_token_count: int = 0,
    ) -> tuple[list[LexerInput], str, int]:
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
        prefix_token_counts = self._deferred_prefix_token_counts
        trailing_token_count = self._deferred_trailing_token_count
        self._deferred_terminals = []
        self._deferred_prefix_token_counts = []
        self._deferred_trailing_token_count = 0

        results: list[LexerInput] = []
        remaining = delta_text

        if self._deferred_post_text:
            remaining = self._deferred_post_text + remaining
            self._deferred_post_text = ""

        # Duplicate-text deferred terminals resolve left-to-right via
        # find(); correct when each terminal text appears once in sequence.
        for idx, terminal in enumerate(deferred):
            prefix_token_count = prefix_token_counts[idx]
            pos = remaining.find(terminal.text)
            if pos > 0:
                results.append(
                    TextChunk(remaining[:pos], token_count=prefix_token_count)
                )
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
                    prefix_token_count += current_token_count
                    remaining = ""
                self._deferred_terminals.append(terminal)
                self._deferred_prefix_token_counts.append(prefix_token_count)

        if self._deferred_terminals:
            self._deferred_trailing_token_count = trailing_token_count
            trailing_token_count = 0

        return results, remaining, trailing_token_count

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

        token_groups: list[list[str]] = [[] for _ in range(len(anchors) + 1)]
        count_groups = [0] * (len(anchors) + 1)
        group_idx = 0
        for item in results:
            if isinstance(item, PreLexedTerminal):
                group_idx += 1
            else:
                token_groups[group_idx].extend(item.token_texts)
                count_groups[group_idx] += item.token_count

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
                    new_results.append(
                        TextChunk(
                            delta_text[consumed:pos],
                            tuple(token_groups[i]),
                            count_groups[i],
                        )
                    )
                new_results.append(anchor)
                consumed = pos + len(anchor.text)
            else:
                has_later_valid = any(p >= 0 for p in positions[i + 1 :])
                # DROP anchors (EOS, etc.) may have text that never
                # arrives in delta_text (stripped by detokenizer).
                # Don't defer remaining content waiting for text
                # that will never come.
                if (
                    not has_later_valid
                    and consumed < len(delta_text)
                    and anchor.terminal != DROP_TERMINAL
                ):
                    self._deferred_post_text += delta_text[consumed:]
                    consumed = len(delta_text)
                self._deferred_terminals.append(anchor)
                self._deferred_prefix_token_counts.append(count_groups[i])
        if consumed < len(delta_text):
            new_results.append(
                TextChunk(
                    delta_text[consumed:],
                    tuple(token_groups[-1]),
                    count_groups[-1],
                )
            )
        return new_results
