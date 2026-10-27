# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-place rewriting and incremental correction of tool-call argument JSON.

:class:`_ArgsRewriter` performs one full rewrite; :class:`_StreamingArgs`
caches the result per tool-call slot and rewrites only the tail a new delta
can change.  Both keep every byte they do not need to replace, so the emitted
text is always a prefix of the final arguments.
"""

from __future__ import annotations

import json

import regex as re

from vllm.parser.utils import JSON_WS
from vllm.tool_parsers.utils import (
    MAX_NESTING_DEPTH,
    canonical_schema_types,
    coerce_to_schema_type,
    is_escaped,
    schema_coercion_types,
    value_changed,
)

_NON_FINITE_LITERALS = ("NaN", "Infinity", "-Infinity")
_NUMBER_CHARS = frozenset("0123456789+-.eE")
# Approximates a bare non-finite literal token: the lookarounds reject
# identifiers ("banaNaNa") and quoted occurrences, but not a token inside a
# longer string.  Fragments cannot be string-scanned, so callers that hold a
# document use ``_document_has_non_finite_literal`` instead.
_NON_FINITE_RE = re.compile(
    r'(?<![\w"])(?:' + "|".join(_NON_FINITE_LITERALS) + r')(?![\w"])'
)
# Characters of a non-finite literal that can precede a delta boundary.
_NON_FINITE_OVERLAP = max(map(len, _NON_FINITE_LITERALS)) - 1
_JSON_STRING_RE = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"', re.DOTALL)


def _has_non_finite_literal(text: str) -> bool:
    """Whether *text* may contain a bare non-finite literal.

    Over-approximates inside strings, which is all a fragment permits: a
    false positive only costs a structural pass, a false negative would miss
    a rewrite.
    """
    # Cheap C-level pre-filter; only a text containing a candidate substring
    # pays for the regex scan that rejects non-token occurrences.
    if "NaN" not in text and "Infinity" not in text:
        return False
    return _NON_FINITE_RE.search(text) is not None


def _document_has_non_finite_literal(text: str) -> bool:
    """Whether the document *text* contains a bare non-finite literal.

    Unlike :func:`_has_non_finite_literal`, text inside strings is ignored,
    so an argument naming ``-Infinity`` does not disable the no-schema fast
    path.  An unmatched quote opens the unterminated trailing string, whose
    content cannot contain a bare token yet.
    """
    if not _has_non_finite_literal(text):
        return False
    outside = _JSON_STRING_RE.sub("", text)
    quote = outside.find('"')
    if quote != -1:
        outside = outside[:quote]
    return _NON_FINITE_RE.search(outside) is not None


def _is_literal(literal: str) -> bool:
    """Whether *literal* is a complete bare boolean, null, or number token."""
    if literal in ("true", "false", "null"):
        return True
    if all(c in _NUMBER_CHARS for c in literal):
        try:
            float(literal)
        except ValueError:
            return False
        return True
    return False


def _next_non_ws(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i] in JSON_WS:
        i += 1
    return i


def _changes_value(literal: str, coerced: object) -> bool:
    """Whether coercing *literal* to *coerced* changes its value or type."""
    try:
        parsed = json.loads(literal)
    except ValueError:
        return True
    return value_changed(parsed, coerced)


class _ArgsRewriter:
    """Rewrite tool-call argument JSON in place.

    Only complete value tokens that a property schema coerces, plus
    ``NaN``/``Infinity`` literals (invalid JSON), are replaced; every other
    byte — separators, whitespace, number formats, escapes, key order — is
    kept verbatim.  A rewritten token is final; a token still undelimited
    may be rewritten once its delimiter arrives, but ``safe_arg_prefix``
    never emits it before then, so streamed argument deltas stay
    prefix-consistent however the model formats its JSON: the final
    arguments are always the streamed text plus a suffix.  Re-serializing
    with ``json.dumps`` instead would normalize separators and numbers,
    breaking that prefix.  A truncated tail is passed through unchanged.
    """

    def __init__(
        self,
        text: str,
        properties: dict | None,
        *,
        final: bool,
        force_parse: bool = False,
    ) -> None:
        self._text = text
        self._props = properties
        # Whether no more text can arrive.  A trailing literal is otherwise
        # treated as still growing: it is only rewritten once delimited.
        self._final = final
        # Parse even when the identity fast path would apply, so streaming
        # callers can learn where structural parsing stops (see ``stop``).
        self._force_parse = force_parse
        self._out: list[str] = []
        # Schema id -> coercion types.  Every schema is reachable from
        # ``self._props``, which the rewriter holds, so no live id is reused.
        self._types_cache: dict[int, list[str]] = {}
        # Offset where the first unparsable token made the rewriter copy the
        # rest verbatim, or ``None`` if it parsed to the end.
        self.stop: int | None = None

    def rewrite(self) -> str:
        text = self._text
        if (
            not self._props
            and not self._force_parse
            and not _document_has_non_finite_literal(text)
        ):
            # Without a property schema only non-finite literals are rewritten.
            return text
        i = self._skip_ws(0)
        if i >= len(text) or text[i] != "{":
            return text
        j = self._object(i, self._props, 0)
        if j < len(text):
            # Text past the closed top-level object is copied verbatim; a
            # tail scan must not resume inside it and correct it differently.
            if self.stop is None:
                self.stop = j
            self._out.append(text[j:])
        return "".join(self._out)

    def _skip_ws(self, i: int) -> int:
        j = _next_non_ws(self._text, i)
        if j > i:
            self._out.append(self._text[i:j])
        return j

    def _scan_string(self, i: int) -> tuple[str | None, int] | None:
        """Return ``(value, end)`` for the closed string at *i*, else None.

        *value* is ``None`` when the string is closed but not valid JSON
        (e.g. an invalid escape); callers copy it verbatim instead of
        giving up on everything after it.
        """
        text = self._text
        m = _JSON_STRING_RE.match(text, i)
        if m is None:
            return None
        j = m.end()
        if text.find("\\", i + 1, j) != -1:
            try:
                return json.loads(text[i:j]), j
            except ValueError:
                return None, j
        return text[i + 1 : j - 1], j

    def _truncated(self, i: int, *, give_up: bool = True) -> int:
        # A give-up is final; a wait for more text is not.  Both copy the
        # remainder verbatim, but only the former bounds incremental resuming.
        if give_up and self.stop is None:
            self.stop = i
        self._out.append(self._text[i:])
        return len(self._text)

    def _delimited(self, j: int) -> bool:
        k = _next_non_ws(self._text, j)
        return k < len(self._text) and self._text[k] in ",}]"

    def _coercion_types(self, schema: dict) -> list[str]:
        """Types a literal under *schema* coerces to; ``[]`` for none."""
        cached = self._types_cache.get(id(schema))
        if cached is not None:
            return cached
        types = schema_coercion_types(schema)
        self._types_cache[id(schema)] = types
        return types

    def _object(self, i: int, props: dict | None, depth: int) -> int:
        text, out = self._text, self._out
        out.append("{")
        i = self._skip_ws(i + 1)
        while True:
            if i >= len(text):
                return i
            if text[i] == "}":
                out.append("}")
                return i + 1
            scanned = self._scan_string(i) if text[i] == '"' else None
            if scanned is None:
                # A non-string key is a give-up; an unterminated one may close.
                return self._truncated(i, give_up=text[i] != '"')
            key, j = scanned
            out.append(text[i:j])
            i = self._skip_ws(j)
            if i >= len(text) or text[i] != ":":
                return self._truncated(i, give_up=i < len(text))
            out.append(":")
            i = self._skip_ws(i + 1)
            i = self._value(
                i, props.get(key) if props and key is not None else None, depth
            )
            if i >= len(text):
                return i
            i = self._skip_ws(i)
            if i < len(text) and text[i] == ",":
                out.append(",")
                i = self._skip_ws(i + 1)
                continue
            if i < len(text) and text[i] == "}":
                out.append("}")
                return i + 1
            # Only a pending input is a wait; trailing whitespace after a
            # complete member is not a structural give-up.
            return self._truncated(i, give_up=i < len(text))

    def _array(self, i: int, items: dict | None, depth: int) -> int:
        text, out = self._text, self._out
        out.append("[")
        i = self._skip_ws(i + 1)
        while True:
            if i >= len(text):
                return i
            if text[i] == "]":
                out.append("]")
                return i + 1
            i = self._value(i, items, depth)
            if i >= len(text):
                return i
            i = self._skip_ws(i)
            if i < len(text) and text[i] == ",":
                out.append(",")
                i = self._skip_ws(i + 1)
                continue
            if i < len(text) and text[i] == "]":
                out.append("]")
                return i + 1
            return self._truncated(i, give_up=i < len(text))

    def _value(self, i: int, schema: object, depth: int) -> int:
        text, out = self._text, self._out
        if i >= len(text):
            return i
        c = text[i]
        if c == "{" or c == "[":
            if depth >= MAX_NESTING_DEPTH:
                return self._truncated(i)
            if c == "{":
                nested = schema.get("properties") if isinstance(schema, dict) else None
                return self._object(
                    i, nested if isinstance(nested, dict) else None, depth + 1
                )
            items = schema.get("items") if isinstance(schema, dict) else None
            return self._array(i, items if isinstance(items, dict) else None, depth + 1)
        if c == '"':
            scanned = self._scan_string(i)
            if scanned is None:
                return self._truncated(i, give_up=False)
            value, j = scanned
            if value is not None and isinstance(schema, dict):
                types = self._coercion_types(schema)
                if types:
                    coerced = coerce_to_schema_type(value, types)
                    if value_changed(value, coerced):
                        out.append(json.dumps(coerced, ensure_ascii=False))
                        return j
            out.append(text[i:j])
            return j
        return self._literal(i, schema)

    def _literal(self, i: int, schema: object) -> int:
        text, out = self._text, self._out
        j = i + 1 if text[i] == "-" else i
        while j < len(text) and (text[j] in _NUMBER_CHARS or text[j].isalpha()):
            j += 1
        literal = text[i:j]
        if not literal:
            return self._truncated(i)
        if literal in _NON_FINITE_LITERALS and (self._final or j < len(text)):
            # Not valid JSON as a bare literal; the quoted string form is.
            # A token at the end of the text may still grow, so wait for one
            # more character; once it cannot, quote it even when the text
            # that follows is malformed, since the final pass will too.
            out.append(f'"{literal}"')
            return j
        if isinstance(schema, dict) and self._delimited(j):
            types = self._coercion_types(schema)
            if types and _is_literal(literal):
                # Keep value-preserving re-encodings (5e3 under integer) and
                # unrepresentable numbers (1e999 under number) verbatim.
                coerced = coerce_to_schema_type(literal, types)
                dumped = json.dumps(coerced, ensure_ascii=False)
                if (
                    dumped != literal
                    and _changes_value(literal, coerced)
                    and not (
                        isinstance(coerced, str)
                        and "string" not in canonical_schema_types(types)
                    )
                ):
                    out.append(dumped)
                    return j
        out.append(text[i:j])
        return j


def _top_level_commas(text: str, depth: int, limit: int | None = None) -> list[int]:
    """Offsets just past each comma at container *depth*, string-aware.

    Stops after *limit* commas when given.
    """
    in_string = False
    escape = False
    commas: list[int] = []
    for i, c in enumerate(text):
        if escape:
            escape = False
        elif in_string:
            if c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
        elif c == "," and depth == 1:
            commas.append(i + 1)
            if len(commas) == limit:
                break
    return commas


def _member_start(text: str, i: int) -> bool:
    """Whether the member following offset *i* begins with a quoted key."""
    i = _next_non_ws(text, i)
    return i < len(text) and text[i] == '"'


def _resume_point(
    raw: str, corrected: str, emitted: int, depth: int
) -> tuple[int, int] | None:
    """Resume offsets in *raw* and *corrected*, or ``None``.

    Every top-level comma is copied verbatim, so the k-th comma in *raw* pairs
    with the k-th comma in *corrected* -- their indices differ because a
    coerced value may change length.  Only a comma whose member begins with a
    quoted key is safe to resume from: scanning from there reproduces the key
    attribution of a full scan, which a malformed member (no quoted key) would
    not.  The last such comma at or before the emitted prefix is the last
    member that is final in both.
    """
    corrected_commas = _top_level_commas(corrected[:emitted], depth)
    if not corrected_commas:
        return None
    # Only the first len(corrected_commas) raw commas can pair with one.
    raw_commas = _top_level_commas(raw, depth, limit=len(corrected_commas))
    if len(raw_commas) < len(corrected_commas):
        return None
    for k in range(len(corrected_commas) - 1, -1, -1):
        if _member_start(raw, raw_commas[k]) and _member_start(
            corrected, corrected_commas[k]
        ):
            return raw_commas[k], corrected_commas[k]
    return None


def _unescaped_quote(text: str, escape: bool) -> bool:
    """Whether *text*, starting after an open escape if *escape*, has a quote."""
    for c in text:
        if escape:
            escape = False
        elif c == "\\":
            escape = True
        elif c == '"':
            return True
    return False


class _StreamingArgs:
    """Incremental correction of streamed argument JSON.

    A corrected document prefix is final once every member in it is
    delimited, so each delta rewrites only the tail from the last top-level
    comma at or before the emitted prefix instead of the whole document.
    The emitted boundary matches a full :class:`_ArgsRewriter` pass plus
    ``safe_arg_prefix``, so delta boundaries are unchanged; text that does
    not extend the cached document falls back to a full pass.  A trailing
    open string under a streamable key streams its content verbatim, so a
    long value is not rescanned on every delta.

    The tail is only resumed from a comma whose member starts with a quoted
    key, and never past the offset where the rewriter first structurally
    gave up (an unterminated trailing value is not a give-up): scanning
    either region from a different start would infer different corrections
    than the full pass.
    """

    __slots__ = (
        "_properties",
        "_string_keys",
        "_raw",
        "_raw_boundary",
        "_corrected_boundary",
        "_emitted",
        "streams",
        "_at_end",
        "_stop",
        "_nonfinite",
        "safe",
    )

    def __init__(
        self, args_json: str, properties: dict | None, string_keys: set[str] | None
    ) -> None:
        """Run a full pass; ``safe`` is the emittable corrected prefix."""
        self._properties = properties
        self._string_keys = string_keys
        self._stop: int | None = None
        self._nonfinite = _document_has_non_finite_literal(args_json)
        corrected, self.safe, self.streams = self._rewrite(args_json)
        self._raw = args_json
        self._raw_boundary = 0
        self._corrected_boundary = 0
        self._emitted = len(self.safe)
        self._at_end = len(self.safe) == len(corrected)
        point = _resume_point(args_json, corrected, len(self.safe), 0)
        if point is not None and (self._stop is None or point[0] <= self._stop):
            self._raw_boundary, self._corrected_boundary = point

    def _rewrite(
        self, text: str, *, force_parse: bool = False, stop_shift: int = 0
    ) -> tuple[str, str, bool]:
        """Correct *text*; fold its give-up offset, shifted by *stop_shift*."""
        rewriter = _ArgsRewriter(
            text, self._properties, final=False, force_parse=force_parse
        )
        out = rewriter.rewrite()
        safe, streams = safe_arg_prefix(out, self._string_keys)
        # The first structural give-up is final: everything after it is
        # copied verbatim, so a tail scan may not resume beyond it.
        if rewriter.stop is not None:
            stop = rewriter.stop + stop_shift
            self._stop = stop if self._stop is None else min(self._stop, stop)
        return out, safe, streams

    def _extends(self, args_json: str) -> bool:
        """Whether *args_json* still has the cached document as its prefix."""
        raw = self._raw
        if not raw.startswith("{") or len(args_json) < len(raw):
            return False
        # Compare the whole prefix: a converter that rewrites text far from
        # the end is invisible to a tail-only check when the tail repeats,
        # and its deltas would be sliced from stale offsets.  ``startswith``
        # is a C-level memcmp, so this stays far cheaper than a rewriter pass.
        return args_json.startswith(raw)

    def delta(self, args_json: str) -> tuple[str, bool] | None:
        """Return (newly safe text, value_streams), or ``None`` for a full pass."""
        raw = self._raw
        if not self._extends(args_json):
            return None
        if self.streams and self._at_end:
            delta = args_json[len(raw) :]
            # No unescaped quote means the delta stays inside the cached open
            # string, so a non-finite-looking token in it is string content
            # and cannot need the rewrite the fragment check guards against.
            if not _unescaped_quote(delta, is_escaped(raw, len(raw))):
                self._raw = args_json
                self._emitted += len(delta)
                return delta, True
        if not self._properties and not self._nonfinite:
            # A non-finite literal introduced after the cache was built may
            # rewrite text the full pass left verbatim after a give-up.
            added = args_json[max(0, len(raw) - _NON_FINITE_OVERLAP) :]
            if _has_non_finite_literal(added):
                return None
        skip = 1 if self._raw_boundary else 0
        base = self._raw_boundary
        raw_tail = args_json[base:]
        tail = "{" + raw_tail if skip else args_json
        tail_out, safe_tail, streams = self._rewrite(
            tail, force_parse=self._nonfinite, stop_shift=base - skip
        )
        body = tail_out[skip:]
        emit_rel = max(len(safe_tail) - skip, 0)
        end = self._corrected_boundary + emit_rel
        delta = ""
        point = None
        if end > self._emitted:
            delta = body[self._emitted - self._corrected_boundary : emit_rel]
            self._emitted = end
            point = _resume_point(raw_tail, body, emit_rel, skip)
        self._raw = args_json
        self.streams = streams
        self._at_end = end == self._corrected_boundary + len(body)
        if point is not None and (self._stop is None or base + point[0] <= self._stop):
            self._raw_boundary = base + point[0]
            self._corrected_boundary += point[1]
        return delta, streams


def safe_arg_prefix(
    json_str: str, string_keys: set[str] | None = None
) -> tuple[str, bool]:
    """Return the prefix of *json_str* up to the last top-level value.

    Middle values (followed by a comma) are stable across streaming
    ticks and included.  The trailing value is excluded for non-string
    values because type coercion may change its serialised form between
    ticks, which would violate the ``startswith(prev)`` prefix invariant.
    String values for keys in ``string_keys`` are prefix-stable, so stream
    their unterminated content instead of buffering long arguments until
    the closing tag arrives.

    Also returns whether the returned prefix stops inside that trailing
    string, so streamed text may be appended to it.  This holds even when
    *json_str* closes the string: converters serialise an open raw value
    as a closed JSON string.
    """
    last_colon = -1
    last_key: str | None = None
    pending_key: str | None = None
    in_string = False
    escape = False
    string_start = -1
    depth = 0
    for i, c in enumerate(json_str):
        if escape:
            escape = False
            continue
        if in_string:
            if c == "\\":
                escape = True
            elif c == '"':
                in_string = False
                if depth == 1 and string_start >= 0:
                    pending_key = json_str[string_start + 1 : i]
            continue
        if c == '"':
            in_string = True
            string_start = i
        elif c in ("{", "["):
            depth += 1
        elif c in ("}", "]"):
            depth -= 1
        elif c == ":" and depth == 1:
            last_colon = i
            last_key = pending_key
            pending_key = None
    if last_colon < 0:
        return "", False
    end = _next_non_ws(json_str, last_colon + 1)
    if end >= len(json_str) or json_str[end] != '"':
        return json_str[:end], False
    if string_keys is not None and last_key not in string_keys:
        return json_str[:end], False

    escape = False
    for i in range(end + 1, len(json_str)):
        c = json_str[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            return json_str[:i], True
    return json_str, True
