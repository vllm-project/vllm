# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resumable structural walk over (possibly incomplete) JSON text.

Shared escape / string / depth state for one-shot scanners and resumable
scans that must agree on every prefix of growing input. Callers react via
``on_string_closed`` / ``on_outside_char``; the walker never rewinds
``pos`` -- only append to ``raw`` between ``advance`` calls.

Also home to the generic incomplete/truncated-JSON scanning primitives
built on top of the walker (:class:`ValueScan`, :class:`KeyScan`,
:func:`closeable_prefix`, :func:`object_members`/:func:`array_items`, ...).
None of these know about any particular model's tool-call shape; a parser
whose format is bare/truncatable JSON (currently only ``llama_json``) wraps
them with its own envelope-specific glue.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Literal

import regex as re

# Handler returns False to stop the walk early (pos lands past the char
# that triggered the stop). Any other return continues.
OutsideCharHandler = Callable[[int, str, int], bool | None]
StringClosedHandler = Callable[[int, int, str | None], bool | None]

_STRUCTURAL = frozenset("{}[]:,")

WS = " \t\r\n"
_HEX = "0123456789abcdefABCDEF"
_STRUCTURAL_RE = re.compile(r'["{}\[\]]')
_PRIMITIVE_END_RE = re.compile(r"[,}\] \t\r\n]")
_ESCAPES = '"\\/bfnrt'
# Ends a JSON string token: closing quote, an escape, or an unescaped control char.
_STR_STOP = re.compile(r'["\\\x00-\x1f]')


class JsonWalker:
    """Left-to-right fold over JSON structure outside string literals."""

    __slots__ = (
        "_pos",
        "_depth",
        "_in_string",
        "_escape",
        "_string_start",
        "_last_key",
        "_brackets",
        "_decode_keys",
        "_key_depth",
    )

    def __init__(
        self,
        start: int = 0,
        depth: int = 0,
        *,
        brackets: Literal["object", "any"] = "any",
        decode_keys: Callable[[str], str] | None = None,
        key_depth: int | None = 1,
    ) -> None:
        self._pos = start
        self._depth = depth
        self._in_string = False
        self._escape = False
        self._string_start = -1
        self._last_key: str | None = None
        self._brackets = brackets
        self._decode_keys = decode_keys
        self._key_depth = key_depth

    @property
    def pos(self) -> int:
        return self._pos

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def last_key(self) -> str | None:
        return self._last_key

    @property
    def suspend_pos(self) -> int:
        """Index where a later ``advance`` resumes (same as :attr:`pos`)."""
        return self._pos

    @property
    def incomplete_string_at(self) -> int:
        """Opening ``"`` index while inside a string, else ``-1``."""
        return self._string_start if self._in_string else -1

    def clear_last_key(self) -> None:
        self._last_key = None

    def advance(
        self,
        raw: str,
        *,
        limit: int | None = None,
        on_string_closed: StringClosedHandler | None = None,
        on_outside_char: OutsideCharHandler | None = None,
    ) -> bool:
        """Consume ``raw[self.pos:limit]``.

        Returns ``False`` if a handler stopped the walk early; ``True``
        when the slice was fully consumed.
        """
        end = len(raw) if limit is None else min(limit, len(raw))
        i = self._pos
        while i < end:
            ch = raw[i]
            if self._escape:
                self._escape = False
                i += 1
                continue
            if self._in_string:
                if ch == "\\":
                    self._escape = True
                elif ch == '"':
                    self._in_string = False
                    key_text: str | None = None
                    if self._key_depth is not None and self._depth == self._key_depth:
                        slice_ = raw[self._string_start + 1 : i]
                        key_text = (
                            self._decode_keys(slice_)
                            if self._decode_keys is not None
                            else slice_
                        )
                        self._last_key = key_text
                    if (
                        on_string_closed is not None
                        and on_string_closed(i, self._depth, key_text) is False
                    ):
                        self._pos = i + 1
                        return False
                i += 1
                continue
            if ch == '"':
                self._in_string = True
                self._string_start = i
                i += 1
                continue
            if self._brackets == "object":
                if ch == "{":
                    self._depth += 1
                elif ch == "}":
                    self._depth -= 1
            elif ch in "{[":
                self._depth += 1
            elif ch in "}]":
                self._depth -= 1
            if (
                on_outside_char is not None
                and ch in _STRUCTURAL
                and on_outside_char(i, ch, self._depth) is False
            ):
                self._pos = i + 1
                return False
            i += 1
        self._pos = i
        return True


def walk_json(
    raw: str,
    start: int = 0,
    *,
    on_string_closed: StringClosedHandler | None = None,
    on_outside_char: OutsideCharHandler | None = None,
    brackets: Literal["object", "any"] = "any",
    decode_keys: Callable[[str], str] | None = None,
    key_depth: int | None = 1,
    depth: int = 0,
) -> int:
    """One-shot walk from ``start``; return the final cursor position."""
    walker = JsonWalker(
        start=start,
        depth=depth,
        brackets=brackets,
        decode_keys=decode_keys,
        key_depth=key_depth,
    )
    walker.advance(
        raw,
        on_string_closed=on_string_closed,
        on_outside_char=on_outside_char,
    )
    return walker.pos


def decode_key(raw_slice: str) -> str:
    """JSON-decode a completed object-key slice (e.g. ``na\\u006de`` →
    ``name``) so escaped keys match; fall back to the raw text."""
    if "\\" not in raw_slice:
        return raw_slice
    try:
        return json.loads(f'"{raw_slice}"')
    except json.JSONDecodeError:
        return raw_slice


def scan_json_value(raw: str, start: int) -> int | None:
    """End index (exclusive) of the JSON value at ``raw[start]``, or ``None``
    while unterminated. Handles objects/arrays/strings/primitives (numbers
    and literals end at a top-level ``,``/``}``/``]`` or whitespace).
    """
    first = raw[start]
    if first in "{[":
        end: list[int] = []

        def on_outside(i: int, ch: str, depth: int) -> bool:
            if ch in "}]" and depth == 0:
                end.append(i + 1)
                return False
            return True

        JsonWalker(start=start, brackets="any", key_depth=None).advance(
            raw, on_outside_char=on_outside
        )
        return end[0] if end else None
    if first == '"':
        escape = False
        for i in range(start + 1, len(raw)):
            ch = raw[i]
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                return i + 1
        return None
    for i in range(start, len(raw)):
        if raw[i] in ",}]" or raw[i] in WS:
            return i
    return None


def string_close(raw: str, start: int, pos: int) -> tuple[int, int]:
    """End of the string opening at ``raw[start]``, searching from ``raw[pos]``.

    Returns ``(end, resume)``: index past the closing quote, or -1 and the
    resume point while unterminated. Even trailing backslashes mean the
    quote closes (same as toggling escape char by char), so resuming
    mid-scan matches a scan from ``start``.
    """
    i = pos
    while True:
        quote = raw.find('"', i)
        if quote < 0:
            return -1, len(raw)
        back = quote - 1
        while back >= start and raw[back] == "\\":
            back -= 1
        if (quote - back) % 2:
            return quote + 1, quote + 1
        i = quote + 1


class ValueScan:
    """Resumable :func:`scan_json_value` for one growing value.

    :meth:`end` may be called again as *raw* grows by appending: a
    left-to-right fold with no lookahead, so resuming from the saved cursor
    matches a rescan from ``start`` -- linear rather than quadratic per
    value. Runs between structural chars use ``str.find`` since a value is
    looked at once per feed.
    """

    __slots__ = ("start", "_pos", "_depth", "_str_start", "_end")

    def __init__(self, start: int) -> None:
        self.start = start
        self._pos = start
        self._depth = 0
        self._str_start = -1
        self._end: int | None = None

    def resumable(self, raw: str) -> bool:
        """Whether *raw* still extends the text this scan has consumed."""
        return len(raw) >= (self._pos if self._end is None else self._end)

    def end(self, raw: str) -> int | None:
        if self._end is not None:
            return self._end
        first = raw[self.start]
        if first in "{[":
            self._scan_container(raw)
        elif first == '"':
            if self._str_start < 0:
                self._str_start = self.start
                self._pos = self.start + 1
            close, self._pos = string_close(raw, self.start, self._pos)
            if close >= 0:
                self._end = close
        else:
            match = _PRIMITIVE_END_RE.search(raw, self._pos)
            if match is None:
                self._pos = len(raw)
            else:
                self._end = match.start()
        return self._end

    def _scan_container(self, raw: str) -> None:
        pos = self._pos
        depth = self._depth
        if self._str_start >= 0:
            close, pos = string_close(raw, self._str_start, pos)
            if close < 0:
                self._pos = pos
                return
            self._str_start = -1
            pos = close
        while True:
            match = _STRUCTURAL_RE.search(raw, pos)
            if match is None:
                self._pos = len(raw)
                self._depth = depth
                return
            at = match.start()
            ch = raw[at]
            if ch == '"':
                close, pos = string_close(raw, at, at + 1)
                if close < 0:
                    self._str_start = at
                    self._pos = pos
                    self._depth = depth
                    return
                pos = close
            elif ch in "{[":
                depth += 1
                pos = at + 1
            else:
                depth -= 1
                pos = at + 1
                if depth == 0:
                    self._pos = self._end = pos
                    self._depth = depth
                    return


def scan_top_level_start(raw: str, key_names: tuple[str, ...]) -> int:
    """Return the offset of the first top-level *key_names* value in a
    (possibly incomplete) JSON envelope, or ``-1`` when it has not
    started."""
    found: list[int] = []

    def on_outside(i: int, ch: str, depth: int) -> bool:
        if ch != ":" or depth != 1 or walker.last_key not in key_names:
            return True
        value_start = i + 1
        while value_start < len(raw) and raw[value_start] in WS:
            value_start += 1
        found.append(-1 if value_start >= len(raw) else value_start)
        return False

    walker = JsonWalker(brackets="any", decode_keys=decode_key, key_depth=1)
    walker.advance(raw, on_outside_char=on_outside)
    return found[0] if found else -1


class KeyScan:
    """Resumable :func:`scan_top_level_start` for one growing envelope.

    A left-to-right fold with no lookahead, so resuming from the saved
    cursor matches a rescan from 0. Without this, an envelope whose key
    hasn't arrived yet -- prose containing a stray ``{`` -- is rescanned
    in full on every feed: quadratic in its length. Structural state lives
    on a long-lived :class:`JsonWalker`; whitespace after ``:`` is settled
    separately so a truncated ``"name":`` can resume.
    """

    __slots__ = ("_walker", "_value_from", "_result")

    def __init__(self) -> None:
        self._walker = JsonWalker(brackets="any", decode_keys=decode_key, key_depth=1)
        # Set once the matching key's ``:`` is seen; the value is the next
        # non-whitespace char, which may not have arrived yet.
        self._value_from = -1
        self._result: int | None = None

    def resumable(self, raw: str) -> bool:
        """Whether *raw* still extends the text this scan has consumed."""
        return len(raw) >= self._walker.pos

    def start(self, raw: str, key_names: tuple[str, ...]) -> int:
        if self._result is not None:
            return self._result
        n = len(raw)
        if self._value_from >= 0:
            return self._settle(raw, n)

        def on_outside(i: int, ch: str, depth: int) -> bool:
            if ch == ":" and depth == 1 and self._walker.last_key in key_names:
                self._value_from = i + 1
                return False
            return True

        self._walker.advance(raw, on_outside_char=on_outside)
        if self._value_from >= 0:
            return self._settle(raw, n)
        return -1

    def _settle(self, raw: str, n: int) -> int:
        """Skip the whitespace between ``:`` and the value, resumably."""
        i = self._value_from
        while i < n and raw[i] in WS:
            i += 1
        self._value_from = i
        if i >= n:
            return -1
        self._result = i
        return i


def scan_top_level(raw: str, key_names: tuple[str, ...]) -> str | None:
    """Raw text span of the first top-level *key_names* value in a
    (possibly incomplete) envelope. Verbatim substring (prefix-stable,
    required by the engine's arg-delta diffing), possibly unterminated;
    ``None`` when not started.
    """
    value_start = scan_top_level_start(raw, key_names)
    if value_start < 0:
        return None
    # A ``None`` end means the value is unterminated: it runs to the end.
    return raw[value_start : scan_json_value(raw, value_start)]


def top_level_keys(raw: str) -> set[str]:
    """Return the completed top-level keys of a (possibly incomplete)
    JSON object: strings at depth 1 followed by ``:``."""
    keys: set[str] = set()

    def on_outside(i: int, ch: str, depth: int) -> bool:
        if ch == ":" and depth == 1:
            if walker.last_key is not None:
                keys.add(walker.last_key)
                walker.clear_last_key()
        elif ch == ",":
            walker.clear_last_key()
        return True

    walker = JsonWalker(brackets="any", decode_keys=decode_key, key_depth=1)
    walker.advance(raw, on_outside_char=on_outside)
    return keys


def scan_string_end(raw: str, start: int) -> tuple[int, int]:
    """Scan the JSON string opening at ``raw[start]``.

    Returns ``(end, safe_end)``: ``end`` past the closing quote (-1 while
    unterminated, on an invalid escape, or a raw control char), and
    ``safe_end`` the longest prefix a closing quote can be appended to.
    """
    n = len(raw)
    i = start + 1
    while True:
        # JSON forbids raw control chars in strings, so one ends the
        # string just as an invalid escape does.
        match = _STR_STOP.search(raw, i)
        if match is None:
            return -1, n
        stop = match.start()
        ch = raw[stop]
        if ch == '"':
            return stop + 1, stop
        if ch != "\\":
            return -1, stop
        if stop + 1 >= n:
            return -1, stop
        nxt = raw[stop + 1]
        if nxt == "u":
            if stop + 6 > n:
                return -1, stop
            if any(c not in _HEX for c in raw[stop + 2 : stop + 6]):
                return -1, stop
            i = stop + 6
        elif nxt in _ESCAPES:
            i = stop + 2
        else:
            return -1, stop


def scan_number(raw: str, start: int) -> tuple[int, int]:
    """Return ``(token_end, complete_end)`` for the number token at
    ``raw[start]``: where the token stops, and how much of it is already a
    complete JSON number (``start`` when none of it is)."""
    n = len(raw)
    i = start
    if i < n and raw[i] == "-":
        i += 1
    int_start = i
    while i < n and "0" <= raw[i] <= "9":
        i += 1
    if i == int_start:
        return i, start
    if raw[int_start] == "0" and i - int_start > 1:
        return int_start + 1, int_start + 1
    complete = i
    if i < n and raw[i] == ".":
        i += 1
        frac_start = i
        while i < n and "0" <= raw[i] <= "9":
            i += 1
        if i == frac_start:
            # No digit after the point; an exponent can't rescue "1.e5".
            return i, complete
        complete = i
    if i < n and raw[i] in "eE":
        i += 1
        if i < n and raw[i] in "+-":
            i += 1
        exp_start = i
        while i < n and "0" <= raw[i] <= "9":
            i += 1
        if i > exp_start:
            complete = i
    return i, complete


def closeable_prefix(raw: str) -> tuple[int, str]:
    """Return ``(end, closers)``: the longest prefix of *raw* that becomes
    valid JSON when *closers* (only ``"``/``}``/``]``) is appended.

    Truncated/malformed model output leaves a JSON span as a fragment;
    emitting it verbatim is invalid and retracting already-sent text is
    impossible, so it's cut back to the last completable point and closed
    by appending -- an append-only repair, keeping streaming and
    non-streaming byte-identical. ``end`` never moves backwards as *raw*
    grows, which keeps the streamed text prefix-stable.

    Lexical errors are cut back the same way: a raw control char ends its
    string, and a fractionless ``1.e5`` keeps only ``1``.
    """
    stack: list[str] = []
    closers = ""
    best = 0
    best_closers = ""
    expect = "value"
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if ch in WS:
            i += 1
            continue
        if expect == "colon":
            if ch != ":":
                break
            i += 1
            expect = "value"
            continue
        if expect == "comma_or_close":
            if ch == "," and stack:
                expect = "key" if stack[-1] == "}" else "value"
                i += 1
                continue
            if not stack or ch != stack[-1]:
                break
            stack.pop()
            closers = closers[1:]
            i += 1
            best, best_closers = i, closers
            continue
        if expect in ("key", "key_or_close"):
            if ch == "}" and expect == "key_or_close":
                stack.pop()
                closers = closers[1:]
                i += 1
                best, best_closers = i, closers
                expect = "comma_or_close"
                continue
            if ch != '"':
                break
            key_end, _ = scan_string_end(raw, i)
            if key_end < 0:
                break
            i = key_end
            expect = "colon"
            continue
        # expect is "value" or "value_or_close"
        if ch == "]" and expect == "value_or_close":
            stack.pop()
            closers = closers[1:]
            i += 1
            best, best_closers = i, closers
            expect = "comma_or_close"
            continue
        if ch in "{[":
            closer = "}" if ch == "{" else "]"
            stack.append(closer)
            closers = closer + closers
            i += 1
            best, best_closers = i, closers
            expect = "key_or_close" if ch == "{" else "value_or_close"
            continue
        if ch == '"':
            str_end, safe = scan_string_end(raw, i)
            if str_end < 0:
                # An unterminated string value closes with a quote; an
                # incomplete escape at the very end does not.
                if safe > best:
                    best, best_closers = safe, '"' + closers
                break
            i = str_end
            best, best_closers = i, closers
            expect = "comma_or_close"
            continue
        if ch == "-" or "0" <= ch <= "9":
            token_end, complete = scan_number(raw, i)
            if complete > i:
                best, best_closers = complete, closers
            if complete != token_end:
                break
            i = token_end
            expect = "comma_or_close"
            continue
        for literal in ("true", "false", "null"):
            if raw.startswith(literal, i):
                i += len(literal)
                best, best_closers = i, closers
                expect = "comma_or_close"
                break
        else:
            break
    return best, best_closers


def object_members(raw: str, start: int, end: int) -> list[tuple[str, int, int]]:
    """Return ``(key, value_start, value_end)`` for each complete member of
    the object ``raw[start:end]``, stopping at the first incomplete one."""
    members: list[tuple[str, int, int]] = []
    i = start + 1
    while i < end:
        while i < end and (raw[i] in WS or raw[i] == ","):
            i += 1
        if i >= end or raw[i] != '"':
            break
        key_end = scan_json_value(raw, i)
        if key_end is None or key_end > end:
            break
        key = decode_key(raw[i + 1 : key_end - 1])
        i = key_end
        while i < end and raw[i] in WS:
            i += 1
        if i >= end or raw[i] != ":":
            break
        i += 1
        while i < end and raw[i] in WS:
            i += 1
        if i >= end:
            break
        value_end = scan_json_value(raw, i)
        if value_end is None or value_end > end:
            break
        members.append((key, i, value_end))
        i = value_end
    return members


def array_items(raw: str, start: int, end: int) -> list[tuple[int, int]]:
    """Return ``(value_start, value_end)`` for each complete element of the
    array ``raw[start:end]``."""
    items: list[tuple[int, int]] = []
    i = start + 1
    while i < end:
        while i < end and (raw[i] in WS or raw[i] == ","):
            i += 1
        if i >= end or raw[i] == "]":
            break
        value_end = scan_json_value(raw, i)
        if value_end is None or value_end > end:
            break
        if value_end <= i:
            # No progress: a closer at an element position (``[1}``) scans
            # as a zero-length value, and the loop would spin. Guarded here
            # since callers are expected to sanitise through closeable_prefix.
            break
        items.append((i, value_end))
        i = value_end
    return items
