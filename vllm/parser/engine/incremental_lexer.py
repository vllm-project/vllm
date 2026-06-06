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
    priority: int = 0
    is_literal: bool = False
    literal: str = ""


@dataclass(slots=True)
class LexToken:
    terminal: str
    value: str


class LexerShape:
    """Immutable pre-computed data derived from terminal definitions.

    Created once per :class:`ParserEngineConfig` and shared across all
    :class:`IncrementalLexer` instances that use the same config.
    """

    __slots__ = (
        "terminals",
        "literal_strings",
        "regex_terminals",
        "max_literal_len",
        "literal_first_chars",
        "has_only_literals",
    )

    def __init__(self, terminals: list[TerminalDef]) -> None:
        self.terminals = sorted(
            terminals,
            key=lambda t: (not t.is_literal, -t.priority, -len(t.pattern.pattern)),
        )
        literal_strings: list[tuple[str, str]] = []
        regex_terminals: list[TerminalDef] = []
        for t in self.terminals:
            if t.is_literal:
                literal_strings.append((t.literal, t.name))
            else:
                regex_terminals.append(t)

        self.literal_strings = literal_strings
        self.regex_terminals = regex_terminals
        max_len = 0
        for lit, _ in literal_strings:
            if len(lit) > max_len:
                max_len = len(lit)
        self.max_literal_len = max_len
        self.literal_first_chars = frozenset(
            lit[0] for lit, _ in literal_strings if lit
        )
        self.has_only_literals = not regex_terminals


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

        self._literal_strings = shape.literal_strings
        self._regex_terminals = shape.regex_terminals
        self._max_literal_len = shape.max_literal_len
        self._literal_first_chars = shape.literal_first_chars
        self._has_only_literals = shape.has_only_literals

    def reset(self) -> None:
        self.buffer = ""

    def feed(self, text: str) -> list[LexToken]:
        if not self.buffer and self._has_only_literals and self._literal_first_chars:
            for ch in text:
                if ch in self._literal_first_chars:
                    break
            else:
                return [LexToken(self.content_terminal, text)]
        self.buffer += text
        return self._drain()

    def flush(self) -> list[LexToken]:
        tokens: list[LexToken] = []
        if self.buffer:
            tokens.append(LexToken(self.content_terminal, self.buffer))
            self.buffer = ""
        return tokens

    def _drain(self) -> list[LexToken]:
        tokens: list[LexToken] = []
        first_chars = self._literal_first_chars
        literals = self._literal_strings
        regex_terminals = self._regex_terminals
        content_terminal = self.content_terminal
        has_only_literals = self._has_only_literals

        while self.buffer:
            if has_only_literals and first_chars:
                has_potential = False
                for ch in self.buffer:
                    if ch in first_chars:
                        has_potential = True
                        break
                if not has_potential:
                    tokens.append(LexToken(content_terminal, self.buffer))
                    self.buffer = ""
                    break

            best_match: tuple[str, str, int] | None = None

            for lit, name in literals:
                if self.buffer.startswith(lit) and (
                    best_match is None or len(lit) > best_match[2]
                ):
                    best_match = (name, lit, len(lit))

            for tdef in regex_terminals:
                m = tdef.pattern.match(self.buffer)
                if m and m.start() == 0:
                    matched = m.group()
                    if best_match is None or len(matched) > best_match[2]:
                        best_match = (tdef.name, matched, len(matched))

            if self._has_prefix_match():
                if best_match is not None:
                    tokens.append(LexToken(best_match[0], best_match[1]))
                    self.buffer = self.buffer[best_match[2] :]
                    continue
                else:
                    break

            if best_match is not None:
                tokens.append(LexToken(best_match[0], best_match[1]))
                self.buffer = self.buffer[best_match[2] :]
            else:
                content_end = self._find_content_boundary()
                if content_end > 0:
                    tokens.append(LexToken(content_terminal, self.buffer[:content_end]))
                    self.buffer = self.buffer[content_end:]
                else:
                    tokens.append(LexToken(content_terminal, self.buffer[0]))
                    self.buffer = self.buffer[1:]

        return tokens

    def _has_prefix_match(self) -> bool:
        buf_len = len(self.buffer)
        if buf_len >= self._max_literal_len:
            return False
        buf = self.buffer
        for lit, _ in self._literal_strings:
            if buf_len < len(lit) and lit.startswith(buf):
                return True
        return False

    def _find_content_boundary(self) -> int:
        buf = self.buffer
        n = len(buf)
        first_chars = self._literal_first_chars
        for i in range(1, n):
            if buf[i] not in first_chars:
                continue
            remaining = n - i
            for lit, _ in self._literal_strings:
                check_len = min(remaining, len(lit))
                if buf[i : i + check_len] == lit[:check_len]:
                    return i
        return n


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
