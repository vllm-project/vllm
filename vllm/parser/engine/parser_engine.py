# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parser engine base that handles both reasoning and tool call
extraction with a single :class:`StreamingParserEngine`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

import regex as re

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.parser.abstract_parser import Parser, StreamState
from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine_config import (
    ArgValueExtender,
    ParserEngineConfig,
    ParserState,
)
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
from vllm.parser.utils import _JSON_WS
from vllm.renderers.chat_utils import get_tool_call_id_type, make_tool_call_id
from vllm.tool_parsers.utils import (
    MAX_NESTING_DEPTH,
    _is_escaped,
    canonical_schema_types,
    coerce_to_schema_type,
    coerce_typed_value,
    extract_types_from_schema,
    find_tool_name,
    find_tool_properties,
    schema_admits_any_value,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

logger = init_logger(__name__)


class ReasoningEnd(NamedTuple):
    """Where reasoning ends in a token sequence."""

    offset: int  # index of the first end token, or the sequence length
    implicit: bool  # the end token is also content, e.g. a tool-call start


class ToolCallSlot:
    __slots__ = (
        "id",
        "name",
        "_args_parts",
        "_args_joined",
        "name_sent",
        "properties",
        "string_keys",
        "arg_cache",
        "_streamed_parts",
        "_streamed_joined",
        "extend_open_value",
        "open_value_streams",
    )

    def __init__(self) -> None:
        self.id: str = ""
        self.name: str = ""
        self._args_parts: list[str] = []
        self._args_joined: str | None = ""
        self.name_sent: bool = False
        self.properties: dict | None = None
        self.string_keys: set[str] | None = None
        self.arg_cache: _StreamingArgs | None = None
        self._streamed_parts: list[str] = []
        self._streamed_joined: str | None = ""
        self.extend_open_value: ArgValueExtender | None = None
        self.open_value_streams: bool = False

    @property
    def args(self) -> str:
        if self._args_joined is None:
            self._args_joined = "".join(self._args_parts)
        return self._args_joined

    def append_args(self, value: str) -> None:
        self._args_parts.append(value)
        self._args_joined = None

    @property
    def streamed_json(self) -> str:
        if self._streamed_joined is None:
            self._streamed_joined = "".join(self._streamed_parts)
        return self._streamed_joined

    @streamed_json.setter
    def streamed_json(self, value: str) -> None:
        self._streamed_parts = [value]
        self._streamed_joined = value

    def append_streamed(self, value: str) -> None:
        self._streamed_parts.append(value)
        self._streamed_joined = None


_NON_FINITE_LITERALS = ("NaN", "Infinity", "-Infinity")
_NUMBER_CHARS = frozenset("0123456789+-.eE")
# A bare non-finite literal token, not the same text inside a string (e.g.
# "banaNaNa"): the no-schema fast path only needs to parse when one exists.
_NON_FINITE_RE = re.compile(r'(?<![\w"])(?:NaN|Infinity)(?![\w"])')
# Characters of a non-finite literal that can precede a delta boundary.
_NON_FINITE_OVERLAP = len("Infinity") - 1
_JSON_STRING_RE = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"', re.DOTALL)


def _has_non_finite_literal(text: str) -> bool:
    # Cheap C-level pre-filter; only a text containing a candidate substring
    # pays for the regex scan that rejects non-token occurrences.
    if "NaN" not in text and "Infinity" not in text:
        return False
    return _NON_FINITE_RE.search(text) is not None


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
    while i < n and text[i] in _JSON_WS:
        i += 1
    return i


def _changes_value(literal: str, coerced: object) -> bool:
    """Whether coercing *literal* to *coerced* changes its value or type."""
    try:
        parsed = json.loads(literal)
    except ValueError:
        return True
    return type(coerced) is not type(parsed) or coerced != parsed


class _ArgsRewriter:
    """Rewrite tool-call argument JSON in place.

    Only complete value tokens that a property schema coerces, plus
    ``NaN``/``Infinity`` literals (invalid JSON), are replaced; every other
    byte — separators, whitespace, number formats, escapes, key order — is
    kept verbatim.  A rewritten token is final; a token still undelimited
    may be rewritten once its delimiter arrives, but ``_safe_arg_prefix``
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
        self._depth = 0
        # Schema id -> (schema, types).  Holding the schema keeps its id()
        # from being reused by a different dict.
        self._types_cache: dict[int, tuple[dict, list[str]]] = {}
        # Offset where the first unparsable token made the rewriter copy the
        # rest verbatim, or ``None`` if it parsed to the end.
        self.stop: int | None = None

    def rewrite(self) -> str:
        text = self._text
        if (
            not self._props
            and not self._force_parse
            and not _has_non_finite_literal(text)
        ):
            # Without a property schema only non-finite literals are rewritten.
            return text
        i = self._skip_ws(0)
        if i >= len(text) or text[i] != "{":
            return text
        j = self._object(i, self._props)
        if j < len(text):
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
            return cached[1]
        types = (
            []
            if schema_admits_any_value(schema)
            else extract_types_from_schema(schema, fallback=[])
        )
        self._types_cache[id(schema)] = (schema, types)
        return types

    def _object(self, i: int, props: dict | None) -> int:
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
            i = self._value(i, props.get(key) if props and key is not None else None)
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
            return self._truncated(i)

    def _array(self, i: int, items: dict | None) -> int:
        text, out = self._text, self._out
        out.append("[")
        i = self._skip_ws(i + 1)
        while True:
            if i >= len(text):
                return i
            if text[i] == "]":
                out.append("]")
                return i + 1
            i = self._value(i, items)
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
            return self._truncated(i)

    def _value(self, i: int, schema: object) -> int:
        text, out = self._text, self._out
        if i >= len(text):
            return i
        c = text[i]
        if c == "{" or c == "[":
            if self._depth >= MAX_NESTING_DEPTH:
                return self._truncated(i)
            self._depth += 1
            try:
                if c == "{":
                    nested = (
                        schema.get("properties") if isinstance(schema, dict) else None
                    )
                    return self._object(i, nested if isinstance(nested, dict) else None)
                items = schema.get("items") if isinstance(schema, dict) else None
                return self._array(i, items if isinstance(items, dict) else None)
            finally:
                self._depth -= 1
        if c == '"':
            scanned = self._scan_string(i)
            if scanned is None:
                return self._truncated(i, give_up=False)
            value, j = scanned
            if value is not None and isinstance(schema, dict):
                coerced, changed = coerce_typed_value(value, schema)
                if changed:
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
        if literal in _NON_FINITE_LITERALS and (self._final or self._delimited(j)):
            # Not valid JSON as a bare literal; the quoted string form is.
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


def _top_level_commas(text: str, depth: int) -> list[int]:
    """Offsets just past each comma at container *depth*, string-aware."""
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
    raw_commas = _top_level_commas(raw, depth)
    corrected_commas = _top_level_commas(corrected[:emitted], depth)
    if not corrected_commas or len(corrected_commas) > len(raw_commas):
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
    ``_safe_arg_prefix``, so delta boundaries are unchanged; text that does
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
        rewriter = _ArgsRewriter(args_json, properties, final=False)
        corrected = rewriter.rewrite()
        self.safe, self.streams = ParserEngine._safe_arg_prefix(corrected, string_keys)
        self._raw = args_json
        self._raw_boundary = 0
        self._corrected_boundary = 0
        self._emitted = len(self.safe)
        self._at_end = len(self.safe) == len(corrected)
        self._stop = rewriter.stop
        self._nonfinite = _has_non_finite_literal(args_json)
        point = _resume_point(args_json, corrected, len(self.safe), 0)
        if point is not None and (self._stop is None or point[0] <= self._stop):
            self._raw_boundary, self._corrected_boundary = point

    def delta(self, args_json: str) -> tuple[str, bool] | None:
        """Return (newly safe text, value_streams), or ``None`` for a full pass."""
        raw = self._raw
        if not args_json.startswith(raw) or not raw.startswith("{"):
            return None
        if not self._properties and not self._nonfinite:
            # A non-finite literal introduced after the cache was built may
            # rewrite text the full pass left verbatim after a give-up.
            added = args_json[max(0, len(raw) - _NON_FINITE_OVERLAP) :]
            if _has_non_finite_literal(added):
                return None
        if self.streams and self._at_end:
            delta = args_json[len(raw) :]
            if not _unescaped_quote(delta, _is_escaped(raw, len(raw))):
                self._raw = args_json
                self._emitted += len(delta)
                return delta, True
        skip = 1 if self._raw_boundary else 0
        base = self._raw_boundary
        tail = "{" + args_json[base:] if skip else args_json
        rewriter = _ArgsRewriter(
            tail, self._properties, final=False, force_parse=self._nonfinite
        )
        tail_out = rewriter.rewrite()
        if rewriter.stop is not None:
            stop = base + rewriter.stop - skip
            # The first structural give-up is final: everything after it is
            # copied verbatim, so a tail scan may not resume beyond it.
            self._stop = stop if self._stop is None else min(self._stop, stop)
        safe_tail, streams = ParserEngine._safe_arg_prefix(tail_out, self._string_keys)
        body = tail_out[skip:]
        emit_rel = max(len(safe_tail) - skip, 0)
        end = self._corrected_boundary + emit_rel
        point = _resume_point(args_json[base:], body, emit_rel, skip)
        delta = ""
        if end > self._emitted:
            delta = body[self._emitted - self._corrected_boundary : emit_rel]
            self._emitted = end
        self._raw = args_json
        self.streams = streams
        self._at_end = end == self._corrected_boundary + len(body)
        if point is not None and (self._stop is None or base + point[0] <= self._stop):
            self._raw_boundary = base + point[0]
            self._corrected_boundary += point[1]
        return delta, streams


class ParserEngine(Parser):
    """A :class:`Parser` backed by a single declarative engine config.

    Subclasses set the ``ParserEngineConfig`` in ``__init__`` to define the
    complete output format for a model (reasoning + tool calls).
    """

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        *,
        parser_engine_config: ParserEngineConfig,
        model_config=None,
        **kwargs,
    ) -> None:
        self.model_tokenizer = tokenizer
        self._tools = tools
        self._stream_state = StreamState(
            tool_call_id_type=(
                get_tool_call_id_type(model_config)
                if model_config is not None
                else "random"
            ),
        )
        self._reasoning_parser = None
        self._tool_parser = None
        self.parser_engine_config = parser_engine_config
        self._engine = StreamingParserEngine(
            parser_engine_config, tokenizer, vocab=self.vocab
        )

        self._has_reasoning = (
            "THINK_END" in parser_engine_config.token_id_terminals
            or "THINK_START" in parser_engine_config.terminals
            or "THINK_END" in parser_engine_config.terminals
            or parser_engine_config.initial_state == ParserState.REASONING
        )
        self._reasoning_ended: bool = not self._has_reasoning
        self._prompt_closes_reasoning: bool = False
        self._streaming_initialized: bool = False
        self._prompt_streaming_prepared: bool = False

        self._tool_slots: list[ToolCallSlot] = []
        self._deferred_content: str = ""
        self._deferred_reasoning: str = ""
        self._content_has_nonws: bool = False
        self._suppress_tool_calls: bool = False

        self._arg_converter = parser_engine_config.arg_converter
        self._arg_structural_chars = parser_engine_config.arg_structural_chars
        self._arg_value_extender = parser_engine_config.arg_value_extender
        self._stream_arg_deltas = parser_engine_config.stream_arg_deltas
        self._strip_trailing_reasoning_ws = (
            parser_engine_config.strip_trailing_reasoning_whitespace
        )
        self._drop_ws_only_content_before_tools = (
            parser_engine_config.drop_whitespace_only_content_before_tools
        )
        self._strip_content_ws_with_tools = (
            parser_engine_config.strip_content_whitespace_with_tools
        )

        vocab = self.vocab
        self._reasoning_start_token_id: int | None = None
        self._reasoning_end_token_id: int | None = None

        start_text = parser_engine_config.token_id_terminals.get("THINK_START")
        end_text = parser_engine_config.token_id_terminals.get("THINK_END")
        if start_text:
            self._reasoning_start_token_id = vocab.get(start_text)
        if end_text:
            self._reasoning_end_token_id = vocab.get(end_text)
        self._turn_boundary_token_ids: frozenset[int] = frozenset(
            token_id
            for token in parser_engine_config.turn_boundary_tokens
            if (token_id := vocab.get(token)) is not None
        )
        self._reasoning_end_token_ids: frozenset[int] = (
            self._derive_reasoning_end_token_ids(parser_engine_config, vocab)
        )

    @property
    def reasoning_start_str(self) -> str | None:
        return self.parser_engine_config.terminal_literal("THINK_START")

    @property
    def reasoning_end_str(self) -> str | None:
        return self.parser_engine_config.terminal_literal("THINK_END")

    @cached_property
    def vocab(self) -> dict[str, int]:
        return self.model_tokenizer.get_vocab()

    # ── Engine lifecycle ──────────────────────────────────────────────

    @property
    def skip_tool_parsing(self) -> bool:
        return self._engine.skip_tool_parsing

    @skip_tool_parsing.setter
    def skip_tool_parsing(self, value: bool) -> None:
        self._engine.skip_tool_parsing = value

    @property
    def skip_reasoning_parsing(self) -> bool:
        return self._engine.skip_reasoning_parsing

    @skip_reasoning_parsing.setter
    def skip_reasoning_parsing(self, value: bool) -> None:
        self._engine.skip_reasoning_parsing = value

    @property
    def reasoning_ended(self) -> bool:
        return self._reasoning_ended

    def initialize_streaming(
        self,
        initial_state: ParserState | None = None,
    ) -> None:
        if not self._streaming_initialized:
            self._streaming_initialized = True
            self._reset(initial_state=initial_state)

    def set_prompt_token_ids(self, prompt_token_ids: Sequence[int]) -> None:
        """Start parsing in content when the prompt already closes reasoning,
        as with continue_final_message or a template that switches thinking
        off. Only configs with turn boundaries can tell a closed block in
        this turn from one in an earlier turn, so others are left as is.
        """
        self._prompt_closes_reasoning = (
            self.parser_engine_config.initial_state is ParserState.REASONING
            and bool(self._turn_boundary_token_ids)
            and self.is_reasoning_end(list(prompt_token_ids))
        )

    def adjust_initial_state_from_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        """See :meth:`ReasoningParser.adjust_initial_state_from_prompt`."""
        return

    def finish_streaming(self) -> DeltaMessage | None:
        events = self._engine.finish()
        if events or self._deferred_content:
            return self._events_to_delta(events, finished=True)
        return None

    def _reset(self, initial_state: ParserState | None = None) -> None:
        prompt_closed = initial_state is None and self._prompt_closes_reasoning
        self._engine.reset(
            initial_state=ParserState.CONTENT if prompt_closed else initial_state
        )
        self._reasoning_ended = not self._has_reasoning or prompt_closed
        self._tool_slots.clear()
        self._deferred_content = ""
        self._deferred_reasoning = ""
        self._content_has_nonws = False
        self._prompt_streaming_prepared = False

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request.skip_special_tokens = False
        return request

    def _preprocess_feed(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> tuple[str, Sequence[int]]:
        return delta_text, delta_token_ids

    def _feed(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> list[SemanticEvent]:
        delta_text, delta_token_ids = self._preprocess_feed(delta_text, delta_token_ids)
        return self._engine.feed(delta_text, delta_token_ids)

    # ── Schema-aware type correction ─────────────────────────────────

    @staticmethod
    def _safe_arg_prefix(
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
        end = last_colon + 1
        while end < len(json_str) and json_str[end] in (" ", "\t", "\n", "\r"):
            end += 1
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

    @staticmethod
    def _streamable_string_keys(properties: dict) -> set[str] | None:
        """Return keys whose trailing string values can safely stream.

        ``None`` means there is no schema, so all string values keep their
        JSON representation as strings.  With a schema, only fields that can
        remain strings are safe to emit before the value is closed; fields
        coerced to bool/number/null/object/array may serialize differently.
        """
        if not properties:
            return None

        streamable: set[str] = set()
        for key, schema in properties.items():
            if canonical_schema_types(extract_types_from_schema(schema)) == {"string"}:
                streamable.add(key)
        return streamable

    def _fix_arg_types(self, args_json: str, func_name: str) -> str:
        """Correct argument values in place using the tool schema.

        See :class:`_ArgsRewriter`; non-finite literals are quoted even
        without a schema.
        """
        return _ArgsRewriter(
            args_json, find_tool_properties(self._tools, func_name), final=True
        ).rewrite()

    def _is_valid_tool_name(self, name: str) -> bool:
        if not self.parser_engine_config.validate_tool_names:
            return True
        if not self._tools:
            return True
        return find_tool_name(self._tools, name)

    def _accept_tool_name(self, name: str) -> bool:
        return bool(name) and self._is_valid_tool_name(name)

    # ── Private helpers ─────────────────────────────────────────────

    def _check_skip_tool_parsing(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> None:
        tools = getattr(request, "tools", None)
        if tools:
            self._tools = tools
        if not self.skip_tool_parsing and not self._suppress_tool_calls:
            tool_choice = getattr(request, "tool_choice", None)
            if tool_choice == "none" and tools:
                self._suppress_tool_calls = True

    def _strip_content_whitespace(
        self,
        content: str,
        tools_called: bool,
    ) -> str | None:
        if tools_called:
            if self._strip_content_ws_with_tools:
                content = content.strip()
            elif self._drop_ws_only_content_before_tools and not content.strip():
                content = ""
        return content or None

    # ── Streaming: parse_delta ────────────────────────────────────────

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        self._initialize_history_tool_call_cnt(request)
        if not self._prompt_streaming_prepared and prompt_token_ids is not None:
            # NOTE: call the hook BEFORE setting the flag, because the hook
            # may invoke ``_reset`` (e.g. via ``initialize_streaming``) which
            # clears ``_prompt_streaming_prepared``.
            self.adjust_initial_state_from_prompt(prompt_token_ids)
            self._prompt_streaming_prepared = True
        self._check_skip_tool_parsing(request)
        events = self._feed(delta_text, delta_token_ids)
        if finished:
            events.extend(self._engine.finish())
        result = self._events_to_delta(events, finished=finished)
        result = self._strip_trailing_reasoning(result)

        # Suppress reasoning deltas if not requested
        if result and not request.include_reasoning:
            result.reasoning = None
            if not result.content and not result.tool_calls:
                result = None

        return result

    def _strip_trailing_reasoning(
        self,
        delta: DeltaMessage | None,
    ) -> DeltaMessage | None:
        """Strip trailing whitespace from reasoning, deferring it until we
        know whether more reasoning follows or reasoning has ended.

        Runs in ``parse_delta`` *after* ``_events_to_delta`` (and any
        subclass overrides) so that overrides see the raw reasoning text.

        Gated by ``strip_trailing_reasoning_whitespace``; when disabled,
        passes through unchanged.
        """
        if not self._strip_trailing_reasoning_ws:
            return delta
        if delta is not None and delta.reasoning is not None:
            combined = self._deferred_reasoning + delta.reasoning
            trimmed = combined.rstrip()
            self._deferred_reasoning = combined[len(trimmed) :]
            delta.reasoning = trimmed or None
            if (
                delta.reasoning is None
                and delta.content is None
                and not delta.tool_calls
            ):
                return None
        elif self._deferred_reasoning and self._reasoning_ended:
            self._deferred_reasoning = ""
        return delta

    # ── Non-streaming: extract_reasoning ──────────────────────────────

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        self._reset()
        events = self._feed(model_output, [])
        events.extend(self._engine.finish())

        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        for event in events:
            if event.type == EventType.REASONING_CHUNK:
                reasoning_parts.append(event.value)
            elif event.type == EventType.TEXT_CHUNK:
                content_parts.append(event.value)
            elif event.type == EventType.REASONING_END:
                self._reasoning_ended = True

        raw_reasoning = "".join(reasoning_parts)
        if self._strip_trailing_reasoning_ws:
            raw_reasoning = raw_reasoning.rstrip()
        reasoning = raw_reasoning or None
        content = "".join(content_parts) or None
        return reasoning, content

    # ── Non-streaming: extract_reasoning_streaming ────────────────────

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        self.initialize_streaming()
        events = self._feed(delta_text, delta_token_ids)
        return self._strip_trailing_reasoning(self._events_to_delta(events))

    # ── Non-streaming: extract_tool_calls ─────────────────────────────

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ExtractedToolCallInformation:
        self._reset()
        self._streaming_initialized = True
        result = self.extract_tool_calls_streaming(
            previous_text="",
            current_text=model_output,
            delta_text=model_output,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        finish_delta = self.finish_streaming()
        return self._build_extracted_result(result, finish_delta)

    def extract_tool_calls_from_content(
        self,
        content: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from reasoning-stripped content.

        Unlike :meth:`extract_tool_calls` which re-parses the full model
        output, this method starts the parser engine in ``CONTENT`` state
        so it can parse content that has already had reasoning stripped.
        """
        self._check_skip_tool_parsing(request)
        _, parsed_content, tool_call_info = self._single_pass_parse(
            content,
            [],
            initial_state=ParserState.CONTENT,
        )
        if parsed_content is not None and tool_call_info.content is None:
            tool_call_info = ExtractedToolCallInformation(
                tools_called=tool_call_info.tools_called,
                tool_calls=tool_call_info.tool_calls,
                content=parsed_content,
            )
        return tool_call_info

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> DeltaMessage | None:
        self.initialize_streaming()
        self._check_skip_tool_parsing(request)
        events = self._feed(delta_text, delta_token_ids)
        return self._strip_trailing_reasoning(self._events_to_delta(events))

    # ── Reasoning state queries ───────────────────────────────────────

    @staticmethod
    def _derive_reasoning_end_token_ids(
        config: ParserEngineConfig, vocab: dict[str, int]
    ) -> frozenset[int]:
        end_terminals: dict[str, ParserState] = {}
        for (state, terminal), transition in config.transitions.items():
            if state != ParserState.REASONING:
                continue
            if EventType.REASONING_END in transition.events:
                end_terminals.setdefault(terminal, transition.next_state)
            elif transition.next_state != ParserState.REASONING:
                return frozenset()

        token_ids: set[int] = set()
        for terminal, next_state in end_terminals.items():
            text = config.token_id_terminals.get(terminal)
            token_id = vocab.get(text) if text is not None else None
            if token_id is not None:
                token_ids.add(token_id)
            elif next_state == ParserState.CONTENT:
                return frozenset()
        return frozenset(token_ids)

    @property
    def reasoning_end_token_ids(self) -> frozenset[int]:
        return self._reasoning_end_token_ids

    @property
    def implicit_reasoning_end_token_ids(self) -> frozenset[int]:
        """Reasoning-end tokens that are also content, e.g. a tool-call start."""
        return self._reasoning_end_token_ids - {self._reasoning_end_token_id}

    def find_reasoning_end(self, token_ids: Sequence[int]) -> ReasoningEnd | None:
        """First reasoning end in `token_ids`, and if its token is content."""
        end_ids = self._reasoning_end_token_ids
        if not end_ids:
            return None
        for offset, token_id in enumerate(token_ids):
            if token_id in end_ids:
                return ReasoningEnd(offset, token_id != self._reasoning_end_token_id)
        return ReasoningEnd(len(token_ids), False)

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        config = self.parser_engine_config
        wait_for_reasoning = config.wait_for_reasoning
        if wait_for_reasoning is None:
            wait_for_reasoning = config.initial_state is ParserState.REASONING
        start_transition = config.transitions.get((config.initial_state, "THINK_START"))
        start_opens_reasoning = (
            start_transition is not None
            and start_transition.next_state is ParserState.REASONING
        )
        end_ids = self._reasoning_end_token_ids
        start_id = self._reasoning_start_token_id
        boundary_ids = self._turn_boundary_token_ids
        for token_id in reversed(input_ids):
            if token_id in end_ids:
                return True
            if token_id == start_id:
                if start_opens_reasoning:
                    return False
                break
            if token_id in boundary_ids:
                break
        return not wait_for_reasoning

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        config = self.parser_engine_config
        wait_for_reasoning = config.wait_for_reasoning
        if wait_for_reasoning is None:
            wait_for_reasoning = config.initial_state is ParserState.REASONING
        if not wait_for_reasoning:
            return input_ids

        end_id = self._reasoning_end_token_id
        if end_id is not None:
            for i in range(len(input_ids) - 1, -1, -1):
                if input_ids[i] == end_id:
                    return input_ids[i + 1 :]

        end_ids = self._reasoning_end_token_ids
        if end_ids:
            turn_start = 0
            boundary_ids = self._turn_boundary_token_ids
            for i in range(len(input_ids) - 1, -1, -1):
                if input_ids[i] in boundary_ids:
                    turn_start = i + 1
                    break
            for i in range(turn_start, len(input_ids)):
                if input_ids[i] in end_ids:
                    return input_ids[i:]
        return input_ids

    def get_streaming_fallback_content(
        self,
        text: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> str | None:
        return None

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        """Return reasoning tokens observed by the parser engine so far."""
        return self._engine.reasoning_token_count

    # ── Single-pass parse helper ────────────────────────────────────────

    def _single_pass_parse(
        self,
        text: str,
        token_ids: Sequence[int],
        initial_state: ParserState | None = None,
    ) -> tuple[str | None, str | None, ExtractedToolCallInformation]:
        """Reset, feed, finish, and extract results in one pass.

        Must be called as a unit — ``_events_to_delta`` populates tool
        state that ``_build_extracted_result`` reads.
        """
        self._reset(initial_state=initial_state)
        events = self._feed(text, token_ids)
        events.extend(self._engine.finish())

        delta = self._events_to_delta(events, finished=True)
        tool_call_info = self._build_extracted_result()

        reasoning = delta.reasoning if delta else None
        if reasoning and self._strip_trailing_reasoning_ws:
            reasoning = reasoning.rstrip() or None

        content = delta.content if delta else None
        if content:
            content = self._strip_content_whitespace(
                content, tool_call_info.tools_called
            )

        return reasoning, content, tool_call_info

    # ── Non-streaming: parse ───────────────────────────────────────────

    def parse(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
        model_output_token_ids: Sequence[int] = (),
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        self._initialize_history_tool_call_cnt(request)
        self._check_skip_tool_parsing(request)
        reasoning, content, tool_call_info = self._single_pass_parse(
            model_output,
            model_output_token_ids,
        )

        tool_calls: list[FunctionCall] | None = None
        if tool_call_info.tools_called:
            tool_calls = [
                FunctionCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in tool_call_info.tool_calls
            ]

        return reasoning, content, tool_calls

    # ── Event-to-delta conversion ─────────────────────────────────────

    def _events_to_delta(
        self,
        events: list[SemanticEvent],
        finished: bool = False,
    ) -> DeltaMessage | None:
        if not events and not self._deferred_content:
            return None

        tool_call_deltas: list[DeltaToolCall] = []
        content_parts: list[str] = []
        reasoning_parts: list[str] = []

        carried_deferred = self._deferred_content
        seen_tool_event = False
        suppress = self._suppress_tool_calls
        for event in events:
            match event.type:
                case EventType.TEXT_CHUNK:
                    if seen_tool_event:
                        self._deferred_content += event.value
                    else:
                        content_parts.append(event.value)
                case EventType.REASONING_CHUNK:
                    reasoning_parts.append(event.value)
                case EventType.REASONING_END:
                    self._reasoning_ended = True
                case EventType.TOOL_CALL_START:
                    if not suppress:
                        seen_tool_event = True
                        self._ensure_slot(event.tool_index)
                case EventType.TOOL_NAME:
                    if not suppress:
                        seen_tool_event = True
                        self._handle_tool_name(event)
                case EventType.ARG_VALUE_CHUNK:
                    if not suppress:
                        seen_tool_event = True
                        self._handle_arg_chunk(event, tool_call_deltas)
                case EventType.TOOL_CALL_END:
                    if not suppress:
                        seen_tool_event = True
                        self._handle_tool_end(event, tool_call_deltas)
                case EventType.REASONING_START:
                    pass  # no delta-level effect

        if len(tool_call_deltas) > 1:
            tool_call_deltas = self._coalesce_tool_call_deltas(tool_call_deltas)

        if self._deferred_content and (not seen_tool_event or not tool_call_deltas):
            # Deferred content carried in from a previous delta precedes this
            # delta's content; content deferred during this call (text after
            # an unpromoted tool event) follows it.
            deferred_now = self._deferred_content[len(carried_deferred) :]
            if carried_deferred:
                content_parts.insert(0, carried_deferred)
            if deferred_now:
                content_parts.append(deferred_now)
            self._deferred_content = ""

        content_str = "".join(content_parts)

        if self._content_has_nonws:
            pass
        elif content_str:
            stripped = content_str.strip()
            if stripped:
                self._content_has_nonws = True
            elif self._tool_slots:
                if self._drop_ws_only_content_before_tools:
                    content_str = ""
            elif not finished:
                self._deferred_content = content_str
                content_str = ""

        content = content_str or None
        reasoning = "".join(reasoning_parts) or None

        if content or tool_call_deltas or reasoning:
            kwargs: dict[str, object] = {}
            if content is not None:
                kwargs["content"] = content
            if reasoning is not None:
                kwargs["reasoning"] = reasoning
            if tool_call_deltas:
                kwargs["tool_calls"] = tool_call_deltas
            return DeltaMessage(**kwargs)
        return None

    def _ensure_slot(self, idx: int) -> None:
        while len(self._tool_slots) <= idx:
            self._tool_slots.append(ToolCallSlot())

    def _ensure_tool_id(self, slot: ToolCallSlot, name: str) -> None:
        if not slot.id:
            state = self._stream_state
            slot.id = make_tool_call_id(
                id_type=state.tool_call_id_type,
                func_name=name,
                idx=state.history_tool_call_cnt,
            )
            state.history_tool_call_cnt += 1

    def _bind_tool(self, slot: ToolCallSlot, name: str) -> None:
        slot.name = name
        slot.name_sent = True
        slot.properties = find_tool_properties(self._tools, name)
        slot.string_keys = self._streamable_string_keys(slot.properties)
        self._ensure_tool_id(slot, name)

    def _handle_tool_name(self, event: SemanticEvent) -> None:
        idx = event.tool_index
        self._tool_slots[idx].name += event.value

    def _emit_name_delta(
        self,
        idx: int,
        deltas: list[DeltaToolCall],
        name: str | None,
    ) -> None:
        if name is None or not self._accept_tool_name(name):
            return
        slot = self._tool_slots[idx]
        self._bind_tool(slot, name)
        deltas.append(
            DeltaToolCall(
                index=idx,
                id=slot.id,
                type="function",
                function=DeltaFunctionCall(name=name),
            )
        )

    def _handle_arg_chunk(
        self,
        event: SemanticEvent,
        deltas: list[DeltaToolCall],
    ) -> None:
        idx = event.tool_index
        slot = self._tool_slots[idx]
        if event.value:
            slot.append_args(event.value)

        if not slot.name_sent:
            if slot.name:
                self._emit_name_delta(idx, deltas, slot.name)
            elif event.value:
                # Name not yet known — try to extract from accumulated args
                name = self._try_extract_name(idx)
                self._emit_name_delta(idx, deltas, name)
        elif event.value:
            # Name already sent — emit arg delta
            arg_delta = self._compute_arg_delta(idx, event.value)
            if arg_delta:
                deltas.append(
                    DeltaToolCall(
                        index=idx,
                        function=DeltaFunctionCall(arguments=arg_delta),
                    )
                )

    def _handle_tool_end(
        self,
        event: SemanticEvent,
        deltas: list[DeltaToolCall],
    ) -> None:
        idx = event.tool_index
        if idx >= len(self._tool_slots):
            return

        remaining = self._flush_arg_converter(idx)
        slot = self._tool_slots[idx]

        if not slot.name_sent:
            name = slot.name or self._try_extract_name(idx) or ""
            if self._accept_tool_name(name):
                self._bind_tool(slot, name)
                deltas.append(
                    DeltaToolCall(
                        index=idx,
                        id=slot.id,
                        type="function",
                        function=DeltaFunctionCall(
                            name=name,
                            arguments=remaining or "",
                        ),
                    )
                )
                remaining = None

        if remaining and slot.name_sent:
            deltas.append(
                DeltaToolCall(
                    index=idx,
                    function=DeltaFunctionCall(arguments=remaining),
                )
            )

    # ── Tool-call delta coalescing ──────────────────────────────────────

    @staticmethod
    def _coalesce_tool_call_deltas(
        deltas: list[DeltaToolCall],
    ) -> list[DeltaToolCall]:
        """Merge entries that share the same index into one per index."""
        merged: dict[int, DeltaToolCall] = {}
        for tc in deltas:
            existing = merged.get(tc.index)
            if existing is None:
                merged[tc.index] = tc
                continue
            if tc.id is not None and existing.id is None:
                existing.id = tc.id
            if tc.type is not None and existing.type is None:
                existing.type = tc.type
            if tc.function is not None:
                if existing.function is None:
                    existing.function = tc.function
                else:
                    if tc.function.name is not None and existing.function.name is None:
                        existing.function.name = tc.function.name
                    if tc.function.arguments is not None:
                        if existing.function.arguments is None:
                            existing.function.arguments = tc.function.arguments
                        else:
                            existing.function.arguments += tc.function.arguments
        if len(merged) == len(deltas):
            return deltas
        return list(merged.values())

    # ── Arg conversion helpers ─────────────────────────────────────────

    def _compute_arg_delta(self, idx: int, raw_delta: str) -> str | None:
        converter = self._arg_converter
        if converter is None:
            return raw_delta

        if not self._stream_arg_deltas:
            return None

        slot = self._tool_slots[idx]
        structural = self._arg_structural_chars
        if slot.extend_open_value is not None:
            diff = slot.extend_open_value(raw_delta)
            if diff is not None:
                if not diff or not slot.open_value_streams:
                    return None
                slot.append_streamed(diff)
                # Emitted outside the incremental state, which is resynced on
                # the next converted delta.
                slot.arg_cache = None
                return diff
            slot.extend_open_value = None
        elif structural is not None and structural.isdisjoint(raw_delta):
            return None

        try:
            current_json = converter(slot.args, True)
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.debug("arg converter failed (streaming): %s", slot.args[:80])
            return None

        if not current_json:
            return None

        cache = slot.arg_cache
        result = cache.delta(current_json) if cache is not None else None
        if result is None:
            cache = _StreamingArgs(current_json, slot.properties, slot.string_keys)
            safe_json, value_streams = cache.safe, cache.streams
            prev = slot.streamed_json
            if prev and not safe_json.startswith(prev):
                slot.arg_cache = None
                return None
            slot.arg_cache = cache
            if not safe_json:
                return None
            slot.streamed_json = safe_json
            delta = safe_json[len(prev) :]
        else:
            delta, value_streams = result
            if delta:
                slot.append_streamed(delta)

        if self._arg_value_extender is not None:
            # Deltas that only extend the open value skip the conversion above.
            slot.extend_open_value = self._arg_value_extender(slot.args)
            slot.open_value_streams = value_streams
        return delta or None

    def _flush_arg_converter(self, idx: int) -> str | None:
        converter = self._arg_converter
        if converter is None:
            return None

        slot = self._tool_slots[idx]
        try:
            final_json = converter(slot.args, False)
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.debug("arg converter failed (flush): %s", slot.args[:80])
            return None

        if final_json:
            final_json = self._fix_arg_types(final_json, slot.name)

        prev = slot.streamed_json
        if final_json and len(final_json) > len(prev):
            if prev and not final_json.startswith(prev):
                return None
            diff = final_json[len(prev) :]
            slot.streamed_json = final_json
            slot.arg_cache = None
            return diff
        return None

    _NAME_RE = re.compile(r'"name"\s*:\s*"([^"]*)"')

    def _try_extract_name(self, idx: int) -> str | None:
        m = self._NAME_RE.search(self._tool_slots[idx].args)
        if m:
            name = m.group(1)
            if name:
                return name
        return None

    # ── Build ExtractedToolCallInformation ─────────────────────────────

    def _build_extracted_result(
        self,
        *deltas: DeltaMessage | None,
    ) -> ExtractedToolCallInformation:
        content_parts: list[str] = []
        for delta in deltas:
            if delta is not None and delta.content:
                content_parts.append(delta.content)

        tool_calls: list[ToolCall] = []
        for idx, slot in enumerate(self._tool_slots):
            if not slot.name and not slot.args:
                continue

            name = slot.name.strip()
            raw_body = slot.args

            if not name and raw_body.strip():
                name, args_json = self._extract_name_and_args(raw_body)
            elif raw_body.strip():
                converter = self._arg_converter
                if converter is not None:
                    try:
                        args_json = converter(raw_body, False)
                    except (json.JSONDecodeError, ValueError, TypeError):
                        logger.debug(
                            "arg converter failed (extract): %s", raw_body[:80]
                        )
                        args_json = self._extract_args_json(raw_body, name)
                else:
                    args_json = self._extract_args_json(raw_body, name)
            else:
                args_json = "{}"

            if self._accept_tool_name(name):
                self._ensure_tool_id(slot, name)
                args_json = self._fix_arg_types(args_json, name)
                tool_calls.append(
                    ToolCall(
                        id=slot.id,
                        function=FunctionCall(name=name, arguments=args_json),
                    )
                )

        content_str = "".join(content_parts)
        content = self._strip_content_whitespace(content_str, len(tool_calls) > 0)

        return ExtractedToolCallInformation(
            tools_called=len(tool_calls) > 0,
            tool_calls=tool_calls,
            content=content,
        )

    @staticmethod
    def _extract_args_value(parsed: dict) -> str | None:
        for key in ("arguments", "parameters"):
            if key in parsed:
                val = parsed[key]
                if isinstance(val, str):
                    return val
                return json.dumps(val, ensure_ascii=False)
        return None

    def _extract_name_and_args(
        self,
        raw_body: str,
    ) -> tuple[str, str]:
        raw_body = raw_body.strip()
        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError:
            return "", raw_body

        if not isinstance(parsed, dict):
            return "", raw_body

        name = parsed.get("name", "")
        args = self._extract_args_value(parsed)
        if args is not None:
            return name, args

        without_name = {k: v for k, v in parsed.items() if k != "name"}
        return name, json.dumps(without_name, ensure_ascii=False)

    def _extract_args_json(self, raw_args: str, func_name: str) -> str:
        if not raw_args.strip():
            return "{}"
        _, args = self._extract_name_and_args(raw_args)
        return args
