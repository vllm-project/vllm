# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Llama 3.x/4 JSON tool-call parser (llama3_json / llama4_json).

Format: ``[<|python_tag|>]{"name": "f", "parameters": {...}}; {"name": "g", ...}``
-- a bare JSON envelope with no tool-name state and no end marker. A ``{``
opens a call; it closes when the envelope's JSON balances
(``tool_call_ends_on_args_balance``); the name is the top-level ``"name"``;
``parameters``/``arguments`` is carved out as a verbatim prefix-stable
substring. Separators (``;``, newlines, or none under the xgrammar "llama"
tag) are dropped like all text after the first completed call, matching
legacy. A balanced ``{...}`` with no top-level ``"name"`` is prose JSON,
not a call; its text is restored as content in place.

Two deliberate contract changes vs. legacy, both forced by the streaming
contract (append-only, must match non-streaming exactly):

* Prose before an envelope is returned as ``content`` alongside the calls.
  Legacy returned ``content=None`` non-streaming and the whole output as
  content with no call at all when streaming. Prose before the opening
  ``{`` can't be retracted, so reporting it in both modes is the only
  parity-preserving option, and OpenAI's schema allows both together.
* When an envelope carries both ``parameters`` and ``arguments``, the one
  appearing first wins. Legacy preferred ``arguments`` non-streaming but
  asserted on the duplicate while streaming. Honoring a later key would
  mean retracting already-streamed text, which the append-only guard
  can't do.
"""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING

import regex as re

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine import ParserEngine, ToolCallSlot
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.tool_parsers.utils import find_tool_properties

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

PYTHON_TAG = "<|python_tag|>"
# Llama 4 wraps calls in these instead of <|python_tag|>; must be consumed,
# not left as content -- skip_special_tokens=False means the detokenizer
# no longer strips them, and drop machinery only covers all_special_tokens.
PYTHON_START = "<|python_start|>"
PYTHON_END = "<|python_end|>"
_WS = " \t\r\n"
_HEX = "0123456789abcdefABCDEF"
_STRUCTURAL_RE = re.compile(r'["{}\[\]]')
_PRIMITIVE_END_RE = re.compile(r"[,}\] \t\r\n]")
_ESCAPES = '"\\/bfnrt'
# Ends a JSON string token: closing quote, an escape, or an unescaped control char.
_STR_STOP = re.compile(r'["\\\x00-\x1f]')
# Matched by position, not this order: the first alias in the text wins.
_ARG_KEYS = ("arguments", "parameters")


def _string_close(raw: str, start: int, pos: int) -> tuple[int, int]:
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


def _scan_json_value(raw: str, start: int) -> int | None:
    """End index (exclusive) of the JSON value at ``raw[start]``, or ``None``
    while unterminated. Handles objects/arrays/strings/primitives (numbers
    and literals end at a top-level ``,``/``}``/``]`` or whitespace).
    """
    first = raw[start]
    if first in "{[":
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if escape:
                escape = False
                continue
            if in_string:
                if ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    return i + 1
        return None
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
        if raw[i] in ",}]" or raw[i] in _WS:
            return i
    return None


class _ValueScan:
    """Resumable :func:`_scan_json_value` for one growing value.

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
            close, self._pos = _string_close(raw, self.start, self._pos)
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
            close, pos = _string_close(raw, self._str_start, pos)
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
                close, pos = _string_close(raw, at, at + 1)
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


def _decode_key(raw_slice: str) -> str:
    """JSON-decode a completed object-key slice (e.g. ``na\\u006de`` →
    ``name``) so escaped keys match; fall back to the raw text."""
    if "\\" not in raw_slice:
        return raw_slice
    try:
        return json.loads(f'"{raw_slice}"')
    except json.JSONDecodeError:
        return raw_slice


def _scan_top_level_start(raw: str, key_names: tuple[str, ...]) -> int:
    """Return the offset of the first top-level *key_names* value in a
    (possibly incomplete) JSON envelope, or ``-1`` when it has not
    started."""
    depth = 0
    in_string = False
    escape = False
    string_start = -1
    last_string: str | None = None
    for i, ch in enumerate(raw):
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                if depth == 1:
                    last_string = _decode_key(raw[string_start + 1 : i])
            continue
        if ch == '"':
            in_string = True
            string_start = i
        elif ch == ":" and depth == 1 and last_string in key_names:
            value_start = i + 1
            while value_start < len(raw) and raw[value_start] in _WS:
                value_start += 1
            return -1 if value_start >= len(raw) else value_start
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
    return -1


class _KeyScan:
    """Resumable :func:`_scan_top_level_start` for one growing envelope.

    A left-to-right fold with no lookahead, so resuming from the saved
    cursor matches a rescan from 0. Without this, an envelope whose key
    hasn't arrived yet -- prose containing a stray ``{`` -- is rescanned
    in full on every feed: quadratic in its length.
    """

    __slots__ = (
        "_pos",
        "_depth",
        "_in_string",
        "_escape",
        "_string_start",
        "_last_string",
        "_value_from",
        "_result",
    )

    def __init__(self) -> None:
        self._pos = 0
        self._depth = 0
        self._in_string = False
        self._escape = False
        self._string_start = -1
        self._last_string: str | None = None
        # Set once the matching key's ``:`` is seen; the value is the next
        # non-whitespace char, which may not have arrived yet.
        self._value_from = -1
        self._result: int | None = None

    def resumable(self, raw: str) -> bool:
        """Whether *raw* still extends the text this scan has consumed."""
        return len(raw) >= self._pos

    def start(self, raw: str, key_names: tuple[str, ...]) -> int:
        if self._result is not None:
            return self._result
        n = len(raw)
        if self._value_from >= 0:
            return self._settle(raw, n)
        i = self._pos
        while i < n:
            ch = raw[i]
            if self._escape:
                self._escape = False
            elif self._in_string:
                if ch == "\\":
                    self._escape = True
                elif ch == '"':
                    self._in_string = False
                    if self._depth == 1:
                        self._last_string = _decode_key(raw[self._string_start + 1 : i])
            elif ch == '"':
                self._in_string = True
                self._string_start = i
            elif ch == ":" and self._depth == 1 and self._last_string in key_names:
                self._value_from = i + 1
                self._pos = i + 1
                return self._settle(raw, n)
            elif ch in "{[":
                self._depth += 1
            elif ch in "}]":
                self._depth -= 1
            i += 1
        self._pos = i
        return -1

    def _settle(self, raw: str, n: int) -> int:
        """Skip the whitespace between ``:`` and the value, resumably."""
        i = self._value_from
        while i < n and raw[i] in _WS:
            i += 1
        self._value_from = i
        if i >= n:
            return -1
        self._result = i
        return i


def _scan_top_level(raw: str, key_names: tuple[str, ...]) -> str | None:
    """Raw text span of the first top-level *key_names* value in a
    (possibly incomplete) envelope. Verbatim substring (prefix-stable,
    required by the engine's arg-delta diffing), possibly unterminated;
    ``None`` when not started.
    """
    value_start = _scan_top_level_start(raw, key_names)
    if value_start < 0:
        return None
    # A ``None`` end means the value is unterminated: it runs to the end.
    return raw[value_start : _scan_json_value(raw, value_start)]


def _args_value_span(raw: str) -> str | None:
    return _scan_top_level(raw, _ARG_KEYS)


def _top_level_keys(raw: str) -> set[str]:
    """Return the completed top-level keys of a (possibly incomplete)
    JSON envelope: strings at depth 1 followed by ``:``."""
    keys: set[str] = set()
    depth = 0
    in_string = False
    escape = False
    string_start = -1
    pending: str | None = None
    for i, ch in enumerate(raw):
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                if depth == 1:
                    pending = _decode_key(raw[string_start + 1 : i])
            continue
        if ch == '"':
            in_string = True
            string_start = i
        elif ch == ":" and depth == 1:
            if pending is not None:
                keys.add(pending)
                pending = None
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        elif ch in ",":
            pending = None
    return keys


def _top_level_name(raw: str) -> str | None:
    """Completed top-level ``"name"`` value from a (possibly incomplete)
    envelope, or ``None``. Unlike the engine's regex extraction, never
    picks up a nested ``"name"``, and only accepts a terminated string
    (an unterminated span ending in an escaped quote isn't complete).
    """
    return _name_from_span(_scan_top_level(raw, ("name",)))


def _name_from_span(span: str | None) -> str | None:
    """Validate a top-level ``"name"`` span and decode it."""
    if not span or not span.startswith('"'):
        return None
    if _scan_json_value(span, 0) != len(span):
        return None
    try:
        name = json.loads(span)
    except json.JSONDecodeError:
        return None
    return name or None


def _envelope_name(raw: str) -> str | None:
    """Classify *raw* as a tool-call envelope and return its name.

    Needs a ``parameters``/``arguments`` key alongside ``name``: legacy
    raised KeyError on bare ``{"name": "f"}`` and fell back to content,
    and prose JSON merely containing a ``name`` field is likewise not a call.
    """
    name = _top_level_name(raw)
    if name is None:
        return None
    keys = _top_level_keys(raw)
    if "arguments" in keys or "parameters" in keys:
        return name
    return None


def _scan_number(raw: str, start: int) -> tuple[int, int]:
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


def _scan_string_end(raw: str, start: int) -> tuple[int, int]:
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


def _closeable_prefix(raw: str) -> tuple[int, str]:
    """Return ``(end, closers)``: the longest prefix of *raw* that becomes
    valid JSON when *closers* (only ``"``/``}``/``]``) is appended.

    Truncated/malformed model output leaves the argument span as a JSON
    fragment; emitting it verbatim is invalid and retracting already-sent
    text is impossible, so it's cut back to the last completable point and
    closed by appending -- an append-only repair, keeping streaming and
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
        if ch in _WS:
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
            key_end, _ = _scan_string_end(raw, i)
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
            str_end, safe = _scan_string_end(raw, i)
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
            token_end, complete = _scan_number(raw, i)
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


def _llama_bare_arg_converter(raw_args: str, partial: bool) -> str:
    """Convert a named choice's bare parameter object to arguments."""
    end, closers = _closeable_prefix(raw_args)
    if partial:
        return raw_args[:end]
    if end == 0:
        return "{}"
    return raw_args[:end] + closers


def _llama_arg_converter(raw_args: str, partial: bool) -> str:
    """Carve the arguments value out of the envelope (verbatim, prefix-
    stable; see ``inkling._inkling_arg_converter`` for the rationale), cut
    back to what still closes into valid JSON, and closed once final."""
    span = _args_value_span(raw_args)
    if span is None:
        return "" if partial else "{}"
    end, closers = _closeable_prefix(span)
    if partial:
        return span[:end]
    if end == 0:
        return "{}"
    return span[:end] + closers


def _object_members(raw: str, start: int, end: int) -> list[tuple[str, int, int]]:
    """Return ``(key, value_start, value_end)`` for each complete member of
    the object ``raw[start:end]``, stopping at the first incomplete one."""
    members: list[tuple[str, int, int]] = []
    i = start + 1
    while i < end:
        while i < end and (raw[i] in _WS or raw[i] == ","):
            i += 1
        if i >= end or raw[i] != '"':
            break
        key_end = _scan_json_value(raw, i)
        if key_end is None or key_end > end:
            break
        key = _decode_key(raw[i + 1 : key_end - 1])
        i = key_end
        while i < end and raw[i] in _WS:
            i += 1
        if i >= end or raw[i] != ":":
            break
        i += 1
        while i < end and raw[i] in _WS:
            i += 1
        if i >= end:
            break
        value_end = _scan_json_value(raw, i)
        if value_end is None or value_end > end:
            break
        members.append((key, i, value_end))
        i = value_end
    return members


def _array_items(raw: str, start: int, end: int) -> list[tuple[int, int]]:
    """Return ``(value_start, value_end)`` for each complete element of the
    array ``raw[start:end]``."""
    items: list[tuple[int, int]] = []
    i = start + 1
    while i < end:
        while i < end and (raw[i] in _WS or raw[i] == ","):
            i += 1
        if i >= end or raw[i] == "]":
            break
        value_end = _scan_json_value(raw, i)
        if value_end is None or value_end > end:
            break
        if value_end <= i:
            # No progress: a closer at an element position (``[1}``) scans
            # as a zero-length value, and the loop would spin. Guarded here
            # since callers are expected to sanitise through _closeable_prefix.
            break
        items.append((i, value_end))
        i = value_end
    return items


_ALL_JSON_TYPES = frozenset(
    {"null", "boolean", "integer", "number", "string", "array", "object"}
)


_CONSTRAINING_KEYS = (
    "properties",
    "additionalProperties",
    "patternProperties",
    "required",
    "anyOf",
    "oneOf",
    "allOf",
    "$ref",
    "enum",
    "const",
)


def _constrains_object(schema: dict) -> bool:
    """Whether *schema* restricts which objects are valid.

    ``{}``, ``{"type": "object"}`` and an empty ``properties`` accept
    anything, including the ``{"name", "parameters"}`` envelope. Anything
    else narrows the shape, so the model emits bare parameters.
    """
    for key in _CONSTRAINING_KEYS:
        value = schema.get(key)
        if key == "properties":
            if isinstance(value, dict) and value:
                return True
        elif value is not None:
            return True
    return False


def _json_type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _declared_types(schema: object) -> set[str]:
    """JSON types permitted by *schema*.

    Sibling constraints and ``allOf`` intersect; ``anyOf``/``oneOf`` union.
    No type constraint permits every type, unlike
    ``extract_types_from_schema``'s string fallback -- why an integer
    under ``{}``, a ``const`` or a ``$ref`` used to be rewritten as a string.
    """
    all_types = set(_ALL_JSON_TYPES)
    if not isinstance(schema, dict):
        return all_types

    def widen(types: set[str]) -> set[str]:
        # Every integer is a valid number, so a "number" schema accepts one.
        if "number" in types:
            types.add("integer")
        return types

    constraints: list[set[str]] = []
    declared: set[str] = set()
    type_value = schema.get("type")
    if isinstance(type_value, str):
        declared.add(type_value)
    elif isinstance(type_value, list):
        declared.update(t for t in type_value if isinstance(t, str))
    if declared:
        constraints.append(widen(declared))

    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        constraints.append(widen({_json_type_name(v) for v in enum}))

    if "const" in schema:
        constraints.append(widen({_json_type_name(schema["const"])}))

    for field in ("anyOf", "oneOf"):
        choices = schema.get(field)
        if isinstance(choices, list) and choices:
            union: set[str] = set()
            for choice in choices:
                union |= _declared_types(choice)
            constraints.append(union)

    choices = schema.get("allOf")
    if isinstance(choices, list) and choices:
        conjunction = set(all_types)
        for choice in choices:
            conjunction &= _declared_types(choice)
        constraints.append(conjunction)

    allowed = all_types
    for constraint in constraints:
        allowed &= constraint
    return allowed


def _is_valid_for_schema_type(value: object, schema: dict) -> bool:
    """Whether *value* already has a JSON type *schema* permits."""
    return _json_type_name(value) in _declared_types(schema)


def _collect_type_edits(
    raw: str,
    start: int,
    end: int,
    schema: dict,
    edits: list[tuple[int, int, str]],
) -> None:
    """Record ``(start, end, replacement)`` for every scalar span in
    ``raw[start:end]`` whose literal type disagrees with *schema*.

    Mirrors ``ParserEngine._coerce_dict``/``_coerce_value`` but works on
    text spans (objects recurse through ``properties``, arrays ``items``).
    """
    if raw[start] == "{":
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, value_start, value_end in _object_members(raw, start, end):
                sub_schema = properties.get(key)
                if isinstance(sub_schema, dict):
                    _collect_type_edits(raw, value_start, value_end, sub_schema, edits)
        return
    if raw[start] == "[":
        items = schema.get("items")
        if isinstance(items, dict):
            for value_start, value_end in _array_items(raw, start, end):
                _collect_type_edits(raw, value_start, value_end, items, edits)
        return
    try:
        value = json.loads(raw[start:end])
    except (json.JSONDecodeError, ValueError):
        return
    if _is_valid_for_schema_type(value, schema):
        return
    try:
        coerced, changed = ParserEngine._coerce_value(value, schema)
    except RecursionError:
        # json.loads above only covers this span; _coerce_value re-enters
        # json.loads on the decoded string via coerce_to_schema_type, whose
        # deep nesting can raise RecursionError too -- left uncoerced, as legacy did.
        return
    if changed:
        edits.append((start, end, json.dumps(coerced, ensure_ascii=False)))


def _splice_types(raw: str, start: int, end: int, schema: dict) -> str:
    """``raw[start:end]`` with schema-coerced scalars substituted in place,
    every other byte kept verbatim.

    Re-serialising the decoded object (what the engine does) rewrites the
    model's separators, so the append-only guard drops the correction.
    Splicing only rewrites values that actually change, keeping the
    result prefix-compatible.
    """
    edits: list[tuple[int, int, str]] = []
    _collect_type_edits(raw, start, end, schema, edits)
    if not edits:
        return raw[start:end]
    # Edits arrive in document order and never overlap, so the result is
    # assembled in one pass instead of copying the whole string per edit.
    out: list[str] = []
    cursor = start
    for edit_start, edit_end, text in edits:
        out.append(raw[cursor:edit_start])
        out.append(text)
        cursor = edit_end
    out.append(raw[cursor:end])
    return "".join(out)


@functools.cache
def llama_json_config(
    markers: tuple[str, ...] = (PYTHON_TAG, PYTHON_START, PYTHON_END),
) -> ParserEngineConfig:
    """Config for the Llama JSON format.

    *markers* is the subset of tool markers present in the tokenizer's
    vocab; only those become **text** terminals -- a marker the model
    can't emit as a token can only have been written as prose, and
    consuming it there deletes the client's text (Llama 3 lacks the
    Llama 4 wrappers, so declaring all three unconditionally ate one
    typed as literal text).
    """
    return ParserEngineConfig(
        name="llama_json",
        terminals={
            **{
                name: text
                for name, text in (
                    ("PYTHON_TAG", PYTHON_TAG),
                    ("PYTHON_START", PYTHON_START),
                    ("PYTHON_END", PYTHON_END),
                )
                if text in markers
            },
            "OPEN_BRACE": "{",
        },
        # Llama 3 tokenizers lack the Llama 4 markers and vice versa;
        # token-id resolution silently skips the ones it cannot resolve.
        token_id_terminals={
            "PYTHON_TAG": PYTHON_TAG,
            "PYTHON_START": PYTHON_START,
            "PYTHON_END": PYTHON_END,
        },
        transitions={
            # The tag alone opens nothing: ipython-style non-JSON after
            # <|python_tag|> stays content; the "{" starts the call.
            (ParserState.CONTENT, "PYTHON_TAG"): Transition(ParserState.CONTENT, ()),
            # Llama 4's wrappers are consumed the same way, never surfacing
            # as content in either mode.
            (ParserState.CONTENT, "PYTHON_START"): Transition(ParserState.CONTENT, ()),
            (ParserState.CONTENT, "PYTHON_END"): Transition(ParserState.CONTENT, ()),
            # Transitions for markers absent from the vocab are harmless:
            # the terminal is never produced, so the key is never matched.
            (ParserState.CONTENT, "OPEN_BRACE"): Transition(
                ParserState.TOOL_ARGS,
                # ARG_VALUE_CHUNK re-injects the consumed "{" into the
                # slot so the accumulated envelope stays valid JSON.
                (EventType.TOOL_CALL_START, EventType.ARG_VALUE_CHUNK),
            ),
        },
        initial_state=ParserState.CONTENT,
        arg_converter=_llama_arg_converter,
        stream_arg_deltas=True,
        tool_args_json=True,
        tool_call_ends_on_args_balance=True,
        validate_tool_names=False,
    )


class _ArgScan:
    """Per-slot incremental state for the argument scans.

    The envelope and its carved-out argument text only ever grow by
    appending, so each scan resumes rather than rescans. ``slot`` identity
    catches the one way text can shrink: recycled as prose JSON, or reset
    between requests.
    """

    __slots__ = (
        "slot",
        "key",
        "value",
        "name_key",
        "name_value",
        "spliced",
        "cursor",
        "member",
    )

    def __init__(self, slot: ToolCallSlot) -> None:
        self.slot = slot
        self.key: _KeyScan | None = None  # args-key scan
        # Where the envelope's top-level "name" is; same resumability need.
        self.name_key: _KeyScan | None = None
        self.name_value: _ValueScan | None = None
        self.value: _ValueScan | None = None  # args value scan
        self.spliced: str = ""  # members already spliced/streamed
        self.cursor: int = 0
        self.member: _ValueScan | None = None  # trailing member's scan


class LlamaJsonParser(ParserEngine):
    """Llama 3.x/4 JSON tool-call parser backed by the parser engine."""

    CONFIG_NAME = "llama_json"

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        # No vocab check for <|python_tag|>: Llama 4 tokenizers lack it
        # (legacy raised RuntimeError there); the text terminal covers both.
        self._engine_to_dense: dict[int, int] = {}
        self._phantom_count: int = 0
        self._drop_content: bool = False
        # Set per request from tool_choice; not cleared by _reset, since
        # non-streaming resets *after* _check_skip_tool_parsing runs.
        self._forced_tool_choice: bool = False
        # Named choices can be constrained to emit only bare parameters.
        self._named_bare_args_name: str | None = None
        self._held_ws: list[str] = []
        # Content withheld because a forced tool choice is active.  Released
        # at finish if no call was ever promoted -- see _filter_events.
        self._held_forced: list[str] = []
        self._arg_scans: dict[int, _ArgScan] = {}
        self._properties_cache: dict[str, dict] = {}
        self._properties_tools: list[Tool] | None = None
        if "parser_engine_config" not in kwargs:
            vocab = kwargs.get("vocab")
            if vocab is None:
                try:
                    vocab = tokenizer.get_vocab()
                except (AttributeError, NotImplementedError):
                    vocab = {}
            present = tuple(
                m for m in (PYTHON_TAG, PYTHON_START, PYTHON_END) if m in vocab
            )
            kwargs["parser_engine_config"] = llama_json_config(present)
        super().__init__(tokenizer, tools, **kwargs)

    def _reset(self, initial_state: ParserState | None = None) -> None:
        super()._reset(initial_state=initial_state)
        self._engine_to_dense.clear()
        self._phantom_count = 0
        self._drop_content = False
        self._held_ws.clear()
        self._held_forced.clear()
        self._arg_scans.clear()

    def _check_skip_tool_parsing(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> None:
        super()._check_skip_tool_parsing(request)
        # Base only suppresses tool_choice="none" WITH tools, but no-tools
        # requests also default to "none"; a bare "{" would else be eaten.
        tool_choice = getattr(request, "tool_choice", None)
        if not self._suppress_tool_calls and tool_choice == "none":
            self._suppress_tool_calls = True
        # Required/named choice emits only tool calls; without this, the
        # array schema those choices apply (``[{...}, {...}]``) leaks its
        # opening bracket as content before the first call completes.
        self._forced_tool_choice = False
        if tool_choice is not None and tool_choice != "none":
            from openai.types.responses import ToolChoiceFunction

            from vllm.entrypoints.openai.chat_completion.protocol import (
                ChatCompletionNamedToolChoiceParam,
            )

            self._forced_tool_choice = tool_choice == "required" or isinstance(
                tool_choice, (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction)
            )
        self._named_bare_args_name = self._bare_args_tool_name(request)
        self._arg_converter = (
            _llama_bare_arg_converter
            if self._named_bare_args_name is not None
            else llama_json_config().arg_converter
        )

    @staticmethod
    def _bare_args_tool_name(
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> str | None:
        """Return the selected name when guided output is bare parameters."""
        tool_choice = getattr(request, "tool_choice", None)
        if tool_choice is None or isinstance(tool_choice, str):
            return None

        from openai.types.responses import ToolChoiceFunction

        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionNamedToolChoiceParam,
        )

        if isinstance(tool_choice, ToolChoiceFunction):
            name = tool_choice.name
        elif isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):
            name = tool_choice.function.name
        else:
            return None

        structured_outputs = getattr(request, "structured_outputs", None)
        if getattr(structured_outputs, "structural_tag", None) is not None:
            return None
        text_format = getattr(getattr(request, "text", None), "format", None)
        schema = getattr(structured_outputs, "json", None)
        if schema is None:
            schema = getattr(text_format, "schema_", None)
        if schema is None:
            return None
        # An unconstrained schema doesn't forbid the {"name",...,
        # "parameters",...} envelope, so a free model writes the envelope
        # the chat template asks for -- reading it whole would hand the
        # executor the envelope. Any schema that constrains the object at
        # all makes bare parameters the shape actually generated.
        #
        # A literal "properties" check gets both ends wrong: it arms for
        # {"type":"object","properties":{}} (constrains nothing; what
        # adjust_request installs for a parameterless tool), and refuses to
        # arm for anyOf/oneOf/allOf/$ref/additionalProperties/enum, which
        # do constrain.
        if not isinstance(schema, dict) or not _constrains_object(schema):
            return None
        return name or None

    def finish_streaming(self) -> DeltaMessage | None:
        delta = super().finish_streaming()
        # The engine skips _events_to_delta entirely when it has nothing
        # buffered, leaving held whitespace unflushed.
        if self._held_ws and not self._suppress_content:
            ws = "".join(self._held_ws)
            self._held_ws.clear()
            if delta is None:
                delta = DeltaMessage(content=ws)
            else:
                delta.content = (delta.content or "") + ws
        # Same gap for text held under a forced tool choice: if no call was
        # promoted, it's the whole response and must not be lost here.
        if self._held_forced and not self._drop_content:
            held = "".join(self._held_forced)
            self._held_forced.clear()
            if delta is None:
                delta = DeltaMessage(content=held)
            else:
                delta.content = (delta.content or "") + held
        return delta

    def _try_extract_name(self, idx: int) -> str | None:
        # Top-level only (the engine's regex would match a nested "name"),
        # gated on an args key proving this is actually a call.
        if self._named_bare_args_name is not None:
            return self._named_bare_args_name
        raw = self._tool_slots[idx].args
        span = self._resumable_name_span(idx, raw)
        if span is None:
            return None
        name = _name_from_span(span)
        if name is None:
            return None
        keys = _top_level_keys(raw)
        return name if ("arguments" in keys or "parameters" in keys) else None

    def _resumable_name_span(self, idx: int, raw: str) -> str | None:
        """``_scan_top_level(raw, ("name",))`` carried across feeds."""
        scan = self._arg_scan(idx)
        key = scan.name_key
        if key is None or not key.resumable(raw):
            key = scan.name_key = _KeyScan()
            scan.name_value = None
        start = key.start(raw, ("name",))
        if start < 0:
            return None
        value = scan.name_value
        if value is None or not value.resumable(raw):
            value = scan.name_value = _ValueScan(start)
        return raw[start : value.end(raw)]

    def _slot_args(self, dense_idx: int) -> str:
        if dense_idx < len(self._tool_slots):
            return self._tool_slots[dense_idx].args
        return ""

    @property
    def _suppress_content(self) -> bool:
        """Whether no further content may be emitted: either a real call
        has completed, or the request forced a tool choice and the whole
        output is tool calls by construction."""
        return self._drop_content or self._forced_tool_choice

    def _flush_held_ws(self, out: list[SemanticEvent]) -> None:
        if self._held_ws:
            out.append(SemanticEvent(EventType.TEXT_CHUNK, "".join(self._held_ws)))
            self._held_ws.clear()

    def _filter_events(
        self,
        events: list[SemanticEvent],
        finished: bool = False,
    ) -> list[SemanticEvent]:
        """Classify tool-call closures as real calls or prose JSON.

        A balanced ``{...}`` with no top-level ``"name"`` is ordinary
        content: its events become a TEXT_CHUNK restoring the swallowed
        text (legacy fell back to content), and its slot is recycled so
        real calls stream with dense indices. After the first real call,
        further content is dropped (legacy: content=None with calls;
        leading prose kept per engine convention).

        Whitespace-only text is held, not forwarded: with a candidate
        call open, the engine would drop it as whitespace-before-tools
        even if the call turns out to be prose -- held text is flushed
        before the next content, discarded once a real call completes.
        """
        out: list[SemanticEvent] = []
        pending: dict[int, list[str]] = {}
        # Where each call's first event of this batch landed in ``out``, so a
        # retraction rescans only that call rather than everything before it.
        call_start: dict[int, int] = {}
        for event in events:
            if event.type == EventType.TEXT_CHUNK:
                if self._drop_content:
                    self._held_ws.clear()
                    self._held_forced.clear()
                elif self._forced_tool_choice:
                    # Forced choice means the wire scaffolding (required-
                    # mode's "[") must not reach the client -- but if cut
                    # before any call is promoted, that text is the whole
                    # response. Hold it; release at finish only if nothing
                    # was promoted.
                    self._held_ws.clear()
                    self._held_forced.append(event.value)
                elif not event.value.strip():
                    # Every whitespace run is held, not just the leading
                    # one: non-streaming strips whitespace around a call, so
                    # forwarding it here ("Sure. {call}") made the routes
                    # disagree on the most ordinary shape there is.
                    self._held_ws.append(event.value)
                else:
                    self._flush_held_ws(out)
                    # A delta can carry text plus trailing whitespace in one
                    # chunk; splitting keeps the hold above chunk-size independent.
                    body = event.value.rstrip()
                    if body != event.value:
                        if body:
                            out.append(SemanticEvent(EventType.TEXT_CHUNK, body))
                        self._held_ws.append(event.value[len(body) :])
                    else:
                        out.append(event)
                continue
            if event.tool_index < 0:
                out.append(event)
                continue
            engine_idx = event.tool_index
            dense_idx = self._engine_to_dense.setdefault(
                engine_idx, engine_idx - self._phantom_count
            )
            if event.type == EventType.ARG_VALUE_CHUNK:
                pending.setdefault(engine_idx, []).append(event.value)
            if event.type == EventType.TOOL_CALL_END:
                accumulated = self._slot_args(dense_idx) + "".join(
                    pending.get(engine_idx, [])
                )
                slot_named = (
                    dense_idx < len(self._tool_slots)
                    and self._tool_slots[dense_idx].name_sent
                )
                if (
                    not slot_named
                    and self._named_bare_args_name is None
                    and _envelope_name(accumulated) is None
                ):
                    # Phantom: retract this batch's (remapped) events for
                    # the call and restore the full text as content.
                    self._retract_call(
                        out, call_start.pop(dense_idx, len(out)), dense_idx
                    )
                    if dense_idx < len(self._tool_slots):
                        self._tool_slots[dense_idx] = ToolCallSlot()
                    self._phantom_count += 1
                    del self._engine_to_dense[engine_idx]
                    # Gated on _drop_content, NOT _suppress_content: forced
                    # choice must not suppress this restore -- if cut before
                    # any call is promoted, this text is the whole response.
                    if accumulated and not self._drop_content:
                        if self._forced_tool_choice:
                            # Hold, don't emit: text under a forced choice
                            # is already held, so emitting here would jump
                            # ahead and leak the required-mode "[" later.
                            self._held_forced.append(accumulated)
                        else:
                            self._flush_held_ws(out)
                            out.append(SemanticEvent(EventType.TEXT_CHUNK, accumulated))
                    continue
                self._drop_content = True
                self._held_ws.clear()
                self._held_forced.clear()
            call_start.setdefault(dense_idx, len(out))
            out.append(SemanticEvent(event.type, event.value, dense_idx))
        if finished and not self._suppress_content:
            self._flush_held_ws(out)
        if finished and not self._drop_content and self._held_forced:
            out.append(SemanticEvent(EventType.TEXT_CHUNK, "".join(self._held_forced)))
            self._held_forced.clear()
        return out

    @staticmethod
    def _retract_call(out: list[SemanticEvent], start: int, dense_idx: int) -> None:
        """Drop this batch's events for a call that turned out to be prose.

        A call's events are contiguous from *start*, so only that suffix is
        examined. Rebuilding the whole list instead made N prose-JSON
        objects cost O(N^2) -- 128 KB of JSON lines took seconds of CPU.
        """
        if start >= len(out):
            return
        kept = [
            e
            for e in out[start:]
            if e.type == EventType.TEXT_CHUNK or e.tool_index != dense_idx
        ]
        del out[start:]
        out.extend(kept)

    @staticmethod
    def _coalesce_arg_events(
        events: list[SemanticEvent],
    ) -> list[SemanticEvent]:
        """Merge each run of consecutive same-index ARG_VALUE_CHUNK events.

        The engine emits ~one arg event per char, each triggering a full
        O(len) rescan (arg converter plus the engine's safe-prefix scan),
        making a call O(n^2). Coalescing runs the rescan once per feed
        instead of once per char, so non-streaming becomes O(n). Boundary
        events (START/NAME/END) break the run, preserving call indices.
        """
        out: list[SemanticEvent] = []
        buf: list[str] = []
        buf_idx: int | None = None

        def flush() -> None:
            nonlocal buf_idx
            if buf_idx is not None:
                out.append(
                    SemanticEvent(EventType.ARG_VALUE_CHUNK, "".join(buf), buf_idx)
                )
                buf.clear()
                buf_idx = None

        for event in events:
            if event.type == EventType.ARG_VALUE_CHUNK:
                if buf_idx is not None and event.tool_index != buf_idx:
                    flush()
                buf_idx = event.tool_index
                buf.append(event.value)
            else:
                flush()
                out.append(event)
        flush()
        return out

    def _tool_properties(self, func_name: str) -> dict:
        """Cache the tool's schema properties by name: ``find_tool_properties``
        walks the whole tool list, and streaming resolves the schema again
        per argument chunk. Dropped when a request swaps the tool list.
        """
        if not func_name:
            return {}
        if self._properties_tools is not self._tools:
            self._properties_tools = self._tools
            self._properties_cache = {}
        properties = self._properties_cache.get(func_name)
        if properties is None:
            properties = find_tool_properties(self._tools, func_name)
            self._properties_cache[func_name] = properties
        return properties

    def _fix_arg_types(self, args_json: str, func_name: str) -> str:
        """Coerce argument values in place instead of re-serialising.

        The engine's ``json.dumps`` of the coerced object needn't match
        the verbatim model text Llama streams, so it fails the append-only
        ``startswith`` guard and is dropped -- leaving truncated JSON.
        Splicing also works on a truncated envelope, keeping non-streaming
        byte-identical to streaming there too.
        """
        properties = self._tool_properties(func_name)
        if not properties or not args_json.startswith("{"):
            return args_json
        return _splice_types(args_json, 0, len(args_json), {"properties": properties})

    def _stable_arg_prefix(
        self,
        raw: str,
        properties: dict,
        string_keys: set[str] | None,
        scan: _ArgScan | None = None,
    ) -> str:
        """Longest prefix of the final spliced arguments already decided.

        Every byte outside a value span survives splicing verbatim, and a
        coerced value depends only on itself, so a completed value can be
        spliced and streamed at once. Only an unfinished value is held
        back -- unless it's a string under a string-typed key, which
        coercion never rewrites.

        With a *scan*, members already settled aren't walked again:
        everything before ``scan.cursor`` is in ``scan.spliced``, and
        *raw* only grows by appending, so resuming matches a full rescan.
        """
        if not raw.startswith("{"):
            return self._safe_arg_prefix(raw, string_keys)
        streamable = string_keys or set()
        out: list[str] = []
        n = len(raw)
        cursor = 0
        i = 1
        member: _ValueScan | None = None
        if scan is not None and scan.cursor <= n:
            out.append(scan.spliced)
            cursor = scan.cursor
            i = cursor or 1
            member = scan.member
        tail_end = n
        while True:
            while i < n and (raw[i] in _WS or raw[i] == ","):
                i += 1
            if i >= n or raw[i] == "}":
                break
            if raw[i] != '"':
                tail_end = i
                break
            key_end = _scan_json_value(raw, i)
            if key_end is None:
                break
            key = _decode_key(raw[i + 1 : key_end - 1])
            j = key_end
            while j < n and raw[j] in _WS:
                j += 1
            if j >= n:
                break
            if raw[j] != ":":
                tail_end = j
                break
            j += 1
            while j < n and raw[j] in _WS:
                j += 1
            if j >= n:
                break
            if member is None or member.start != j or not member.resumable(raw):
                member = _ValueScan(j)
            value_end = member.end(raw)
            schema = properties.get(key)
            if not isinstance(schema, dict):
                schema = None
            if value_end is None:
                stable = raw[j] == '"' and (schema is None or key in streamable)
                tail_end = n if stable else j
                break
            out.append(raw[cursor:j])
            out.append(
                raw[j:value_end]
                if schema is None
                else _splice_types(raw, j, value_end, schema)
            )
            cursor = i = value_end
            member = None
        settled = "".join(out)
        if scan is not None:
            scan.spliced = settled
            scan.cursor = cursor
            scan.member = member
        return settled + raw[cursor:tail_end]

    def _arg_scan(self, idx: int) -> _ArgScan:
        slot = self._tool_slots[idx]
        scan = self._arg_scans.get(idx)
        if scan is None or scan.slot is not slot:
            scan = self._arg_scans[idx] = _ArgScan(slot)
        return scan

    def _partial_args(self, scan: _ArgScan, raw: str) -> str:
        """``_llama_arg_converter(raw, True)`` with the envelope scan
        carried across feeds instead of restarted on every one."""
        value = scan.value
        if value is None:
            key = scan.key
            if key is None or not key.resumable(raw):
                key = scan.key = _KeyScan()
            start = key.start(raw, _ARG_KEYS)
            if start < 0:
                return ""
            value = scan.value = _ValueScan(start)
        elif not value.resumable(raw):
            value = scan.value = _ValueScan(value.start)
        # A ``None`` end means the value is still open, i.e. runs to the
        # end of the envelope.
        span = raw[value.start : value.end(raw)]
        return span[: _closeable_prefix(span)[0]]

    def _compute_arg_delta(self, idx: int, raw_delta: str) -> str | None:
        """Stream a schema-spliced prefix instead of raw model text.

        The engine streams verbatim and only coerces at flush; the
        re-serialised result then isn't an extension of what was already
        sent, so the correction is silently dropped. Splicing each value
        as it streams keeps the stream append-only and non-streaming-convergent.
        """
        if self._arg_converter is None or not self._stream_arg_deltas:
            return super()._compute_arg_delta(idx, raw_delta)
        slot = self._tool_slots[idx]
        # The incremental scan knows this parser's own conversion; anything
        # else configured keeps the plain per-feed call.
        incremental = self._arg_converter is _llama_arg_converter
        scan = self._arg_scan(idx) if incremental else None
        try:
            current = (
                self._partial_args(scan, slot.args)
                if scan is not None
                else self._arg_converter(slot.args, True)
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
        if not current:
            return None
        properties = self._tool_properties(slot.name)
        if properties:
            safe = self._stable_arg_prefix(current, properties, slot.string_keys, scan)
        else:
            safe = self._safe_arg_prefix(current, slot.string_keys)
        prev = slot.streamed_json
        if not safe or safe == prev or (prev and not safe.startswith(prev)):
            return None
        slot.streamed_json = safe
        return safe[len(prev) :]

    def _events_to_delta(
        self,
        events: list[SemanticEvent],
        finished: bool = False,
    ) -> DeltaMessage | None:
        events = self._coalesce_arg_events(events)
        if self._suppress_tool_calls:
            # Bare-JSON markup is indistinguishable from ordinary JSON
            # content, so tool_choice="none" returns it as content, not dropped.
            events = [
                SemanticEvent(EventType.TEXT_CHUNK, e.value, e.tool_index)
                if e.value and e.type == EventType.ARG_VALUE_CHUNK
                else e
                for e in events
            ]
        else:
            events = self._filter_events(events, finished=finished)
        return super()._events_to_delta(events, finished=finished)
