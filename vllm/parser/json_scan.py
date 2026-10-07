# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resumable structural walk over (possibly incomplete) JSON text.

Shared escape / string / depth state for one-shot scanners and resumable
scans that must agree on every prefix of growing input. Callers react via
``on_string_closed`` / ``on_outside_char``; the walker never rewinds
``pos`` -- only append to ``raw`` between ``advance`` calls.

``_ValueScan`` in ``llama_json`` still uses a ``find``/regex fast path for
container streaming. A later migration can reuse this walker's depth and
string state (``incomplete_string_at`` / ``suspend_pos``) with
``_string_close`` as a string-skip plugin, or share only those fields via
a mixin -- without changing the one-shot / ``_KeyScan`` contract here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

# Handler returns False to stop the walk early (pos lands past the char
# that triggered the stop). Any other return continues.
OutsideCharHandler = Callable[[int, str, int], bool | None]
StringClosedHandler = Callable[[int, int, str | None], bool | None]

_STRUCTURAL = frozenset("{}[]:,")


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
