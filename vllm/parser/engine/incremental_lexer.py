# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Incremental text lexer that converts text chunks into terminal
tokens, with prefix-match buffering for ambiguous boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import regex as re

CONTENT_TERMINAL = "__CONTENT__"


@dataclass(slots=True)
class TerminalDef:
    name: str
    pattern: re.Pattern[str]
    is_literal: bool = False
    literal: str = ""
    prefix: str = ""


@dataclass(slots=True)
class LexToken:
    terminal: str
    value: str
    token_count: int = 0


class LexerShape:
    """Immutable pre-computed data derived from terminal definitions.

    Created once per :class:`ParserEngineConfig` and shared across all
    :class:`IncrementalLexer` instances that use the same config.
    """

    __slots__ = (
        "terminals",
        "literal_strings",
        "max_literal_len",
        "literal_first_chars",
        "has_only_literals",
        "prefix_set",
        "literals_by_first",
        "regex_terminals",
        "boundary_probes",
    )

    def __init__(self, terminals: list[TerminalDef]) -> None:
        self.terminals = sorted(
            terminals,
            key=lambda t: (not t.is_literal, -len(t.pattern.pattern)),
        )
        literal_strings: list[tuple[str, str]] = []
        for t in self.terminals:
            if t.is_literal:
                literal_strings.append((t.literal, t.name))

        self.regex_terminals = [t for t in self.terminals if not t.is_literal]

        self.literal_strings = literal_strings
        max_len = 0
        for lit, _ in literal_strings:
            if len(lit) > max_len:
                max_len = len(lit)
        self.max_literal_len = max_len
        self.literal_first_chars = frozenset(
            lit[0] for lit, _ in literal_strings if lit
        ) | frozenset(t.prefix[0] for t in self.regex_terminals)
        self.has_only_literals = all(t.is_literal for t in terminals)

        # Prefixes that could still grow into a terminal: for a regex
        # terminal, only the proper prefixes of its literal prefix -- once
        # the full prefix is present, partial regex matching takes over.
        prefix_set: set[str] = set()
        for lit, _ in literal_strings:
            for i in range(1, len(lit)):
                prefix_set.add(lit[:i])
        for t in self.regex_terminals:
            for i in range(1, len(t.prefix)):
                prefix_set.add(t.prefix[:i])
        self.prefix_set = frozenset(prefix_set)

        self.boundary_probes = [lit for lit, _ in literal_strings] + [
            t.prefix for t in self.regex_terminals
        ]

        by_first: dict[str, list[tuple[str, str]]] = {}
        for lit, name in literal_strings:
            if lit:
                by_first.setdefault(lit[0], []).append((lit, name))
        self.literals_by_first = by_first


class IncrementalLexer:
    """Converts streaming text into terminal tokens.

    The key feature is **prefix-match buffering**: when the text in the
    buffer could be the start of a multi-character terminal (e.g.
    ``"<tool_"`` that could become ``"<tool_call>"``), the lexer holds
    the text rather than emitting it.  When the next chunk arrives, it
    either completes the terminal or flushes the buffered text as
    content.

    Terminals are tried in priority order (literals first, then by
    descending priority, then by pattern length).
    """

    def __init__(
        self,
        terminals: list[TerminalDef] | LexerShape,
        content_terminal: str = CONTENT_TERMINAL,
    ) -> None:
        if isinstance(terminals, LexerShape):
            shape = terminals
        else:
            shape = LexerShape(terminals)
        self._shape = shape
        self.terminals = shape.terminals
        self.content_terminal = content_terminal
        self.buffer = ""
        self._token_counts: list[int] = []

        self._literal_strings = shape.literal_strings
        self._max_literal_len = shape.max_literal_len
        self._literal_first_chars = shape.literal_first_chars
        self._has_only_literals = shape.has_only_literals
        self._prefix_set = shape.prefix_set
        self._literals_by_first = shape.literals_by_first
        self._regex_terminals = shape.regex_terminals
        self._boundary_probes = shape.boundary_probes

    def reset(self) -> None:
        self.buffer = ""
        self._token_counts.clear()

    def feed(
        self,
        text: str,
        token_texts: tuple[str, ...] = (),
        token_count: int = 0,
    ) -> list[LexToken]:
        char_token_counts = self._char_token_counts(text, token_texts, token_count)
        if not self.buffer and self._has_only_literals and self._literal_first_chars:
            for ch in text:
                if ch in self._literal_first_chars:
                    break
            else:
                return [LexToken(self.content_terminal, text, sum(char_token_counts))]
        self.buffer += text
        self._token_counts.extend(char_token_counts)
        return self._drain()

    def flush(self) -> list[LexToken]:
        tokens: list[LexToken] = []
        if self.buffer:
            tokens.extend(self._drain(final=True))
        if self.buffer:
            tokens.append(
                LexToken(self.content_terminal, self.buffer, sum(self._token_counts))
            )
            self.buffer = ""
            self._token_counts.clear()
        return tokens

    @staticmethod
    def _char_token_counts(
        text: str,
        token_texts: tuple[str, ...],
        token_count: int,
    ) -> list[int]:
        counts = [0] * len(text)
        if not text:
            return counts
        if token_texts:
            pos = 0
            assigned = 0
            for token_text in token_texts:
                if not token_text:
                    continue
                found = text.find(token_text, pos)
                if found < 0:
                    continue
                counts[found] += 1
                assigned += 1
                pos = found + len(token_text)
            missing = token_count - assigned
            if missing > 0:
                counts[0] += missing
        elif token_count:
            counts[0] = token_count
        return counts

    def _pop_token_count(self, length: int) -> int:
        token_count = sum(self._token_counts[:length])
        del self._token_counts[:length]
        return token_count

    def _drain(self, *, final: bool = False) -> list[LexToken]:
        tokens: list[LexToken] = []
        first_chars = self._literal_first_chars
        content_terminal = self.content_terminal
        has_only_literals = self._has_only_literals
        literals_by_first = self._literals_by_first
        prefix_set = self._prefix_set

        while self.buffer:
            if has_only_literals and first_chars:
                has_potential = False
                for ch in self.buffer:
                    if ch in first_chars:
                        has_potential = True
                        break
                if not has_potential:
                    tokens.append(
                        LexToken(content_terminal, self.buffer, sum(self._token_counts))
                    )
                    self.buffer = ""
                    self._token_counts.clear()
                    break

            best_match: tuple[str, str, int] | None = None

            first = self.buffer[0]
            for lit, name in literals_by_first.get(first, ()):
                if self.buffer.startswith(lit) and (
                    best_match is None or len(lit) > best_match[2]
                ):
                    best_match = (name, lit, len(lit))

            # Regex terminals match only at the buffer start, gated by their
            # literal prefix. A partial match (the buffer could still grow
            # into a full match) holds the buffer, mirroring literal prefix
            # buffering; a literal match of the same length wins the tie.
            regex_hold = False
            for t in self._regex_terminals:
                if not self.buffer.startswith(t.prefix):
                    continue
                m = t.pattern.match(self.buffer, partial=not final)
                if m is None:
                    continue
                if m.partial:
                    regex_hold = True
                    continue
                length = m.end()
                if best_match is None or length > best_match[2]:
                    best_match = (t.name, m.group(0), length)
            if regex_hold:
                break

            # If the current buffer is both a complete literal and the prefix
            # of a longer literal, wait for the next chunk. For example,
            # "<invoke name=" should not be emitted before the next chunk
            # proves whether this is the quoted form '<invoke name="'.
            if self.buffer in prefix_set and not final:
                if best_match is not None:
                    longer_match = False
                    for lit, _ in literals_by_first.get(first, ()):
                        if len(lit) > best_match[2] and lit.startswith(self.buffer):
                            longer_match = True
                            break
                    if not longer_match:
                        tokens.append(
                            LexToken(
                                best_match[0],
                                best_match[1],
                                self._pop_token_count(best_match[2]),
                            )
                        )
                        self.buffer = self.buffer[best_match[2] :]
                        continue
                    break
                else:
                    break

            if best_match is not None:
                tokens.append(
                    LexToken(
                        best_match[0],
                        best_match[1],
                        self._pop_token_count(best_match[2]),
                    )
                )
                self.buffer = self.buffer[best_match[2] :]
            else:
                content_end = self._find_content_boundary()
                if content_end > 0:
                    tokens.append(
                        LexToken(
                            content_terminal,
                            self.buffer[:content_end],
                            self._pop_token_count(content_end),
                        )
                    )
                    self.buffer = self.buffer[content_end:]
                else:
                    tokens.append(
                        LexToken(
                            content_terminal,
                            self.buffer[0],
                            self._pop_token_count(1),
                        )
                    )
                    self.buffer = self.buffer[1:]

        return tokens

    def _find_content_boundary(self) -> int:
        buf = self.buffer
        n = len(buf)
        first_chars = self._literal_first_chars
        for i in range(1, n):
            if buf[i] not in first_chars:
                continue
            remaining = n - i
            for probe in self._boundary_probes:
                check_len = min(remaining, len(probe))
                if buf[i : i + check_len] == probe[:check_len]:
                    return i
        return n


_REGEX_METACHARS = frozenset("[](){}?*+|^$.\\")


def terminal_from_regex(name: str, pattern: str) -> TerminalDef:
    """Build a non-literal terminal from a regex *pattern*.

    The pattern must start with a run of plain literal characters (its
    prefix); the lexer only attempts the regex once the buffer starts with
    that prefix, and holds shorter buffers that could still complete it.
    The pattern should have an unambiguous end: a complete match must not
    be extensible by further input, or the lexer may emit it early.
    """
    prefix_chars: list[str] = []
    for ch in pattern:
        if ch in _REGEX_METACHARS:
            break
        prefix_chars.append(ch)
    prefix = "".join(prefix_chars)
    if not prefix:
        raise ValueError(
            f"regex terminal {name!r} must start with a literal prefix: {pattern!r}"
        )
    return TerminalDef(name=name, pattern=re.compile(pattern), prefix=prefix)


def terminals_from_literals(literals: dict[str, str]) -> list[TerminalDef]:
    return [
        TerminalDef(
            name=name,
            pattern=re.compile(re.escape(lit)),
            is_literal=True,
            literal=lit,
        )
        for name, lit in literals.items()
    ]
