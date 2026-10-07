# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Llama 3.x/4 JSON tool-call parser (llama3_json / llama4_json).

Format: ``[<|python_tag|>]{"name": "f", "parameters": {...}}; {"name": "g", ...}``
-- a bare JSON envelope with no tool-name state and no end marker. A ``{``
opens a call; it closes when the envelope's JSON balances
(``tool_call_ends_on_args_balance``); the name is the top-level ``"name"``;
``parameters``/``arguments`` is carved out as a verbatim prefix-stable
substring. A balanced ``{...}`` with no top-level ``"name"`` is prose JSON,
not a call; its text is restored as content in place.

This commit brings the parser up to full *format-shape* coverage: named
tool choice constrained to bare parameters (no envelope), phantom-JSON
retraction (prose that merely looks like a call), and envelope-aware name
resolution that never picks up a nested ``"name"`` key the way the engine's
default regex would. Truncation repair, schema-coercion safety, and
resumable scanning are not implemented yet -- a value cut off mid-stream is
still emitted verbatim, which later commits fix.
"""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine import ParserEngine, ToolCallSlot
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.json_scan import JsonWalker

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
# Matched by position, not this order: the first alias in the text wins.
_ARG_KEYS = ("arguments", "parameters")


def _scan_json_value(raw: str, start: int) -> int | None:
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
        if raw[i] in ",}]" or raw[i] in _WS:
            return i
    return None


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
    found: list[int] = []

    def on_outside(i: int, ch: str, depth: int) -> bool:
        if ch != ":" or depth != 1 or walker.last_key not in key_names:
            return True
        value_start = i + 1
        while value_start < len(raw) and raw[value_start] in _WS:
            value_start += 1
        found.append(-1 if value_start >= len(raw) else value_start)
        return False

    walker = JsonWalker(brackets="any", decode_keys=_decode_key, key_depth=1)
    walker.advance(raw, on_outside_char=on_outside)
    return found[0] if found else -1


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

    def on_outside(i: int, ch: str, depth: int) -> bool:
        if ch == ":" and depth == 1:
            if walker.last_key is not None:
                keys.add(walker.last_key)
                walker.clear_last_key()
        elif ch == ",":
            walker.clear_last_key()
        return True

    walker = JsonWalker(brackets="any", decode_keys=_decode_key, key_depth=1)
    walker.advance(raw, on_outside_char=on_outside)
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

    Needs a ``parameters``/``arguments`` key alongside ``name``: a bare
    ``{"name": "f"}`` and prose JSON merely containing a ``name`` field are
    both not a call.
    """
    name = _top_level_name(raw)
    if name is None:
        return None
    keys = _top_level_keys(raw)
    if "arguments" in keys or "parameters" in keys:
        return name
    return None


def _llama_bare_arg_converter(raw_args: str, partial: bool) -> str:
    """Convert a named choice's bare parameter object to arguments.

    No truncation repair yet: an unfinished value is cut back to the last
    *complete* value and the rest held, same limitation as the envelope
    converter below.
    """
    if partial:
        return raw_args
    end = _scan_json_value(raw_args, 0) if raw_args else None
    if end is None:
        return raw_args or "{}"
    return raw_args[:end]


def _llama_arg_converter(raw_args: str, partial: bool) -> str:
    """Carve the arguments value out of the envelope (verbatim,
    prefix-stable). No truncation repair yet -- see
    ``_llama_bare_arg_converter``.
    """
    span = _args_value_span(raw_args)
    if span is None:
        return "" if partial else "{}"
    return span if partial else (span or "{}")


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


@functools.cache
def llama_json_config(
    markers: tuple[str, ...] = (PYTHON_TAG, PYTHON_START, PYTHON_END),
) -> ParserEngineConfig:
    """Config for the Llama JSON format.

    *markers* is the subset of tool markers present in the tokenizer's
    vocab; only those become **text** terminals -- a marker the model
    can't emit as a token can only have been written as prose, and
    consuming it there deletes the client's text.
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
        token_id_terminals={
            "PYTHON_TAG": PYTHON_TAG,
            "PYTHON_START": PYTHON_START,
            "PYTHON_END": PYTHON_END,
        },
        transitions={
            (ParserState.CONTENT, "PYTHON_TAG"): Transition(ParserState.CONTENT, ()),
            (ParserState.CONTENT, "PYTHON_START"): Transition(ParserState.CONTENT, ()),
            (ParserState.CONTENT, "PYTHON_END"): Transition(ParserState.CONTENT, ()),
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


class LlamaJsonParser(ParserEngine):
    """Llama 3.x/4 JSON tool-call parser backed by the parser engine."""

    CONFIG_NAME = "llama_json"

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
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
        return _envelope_name(self._tool_slots[idx].args)

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
        text, and its slot is recycled so real calls stream with dense
        indices. After the first real call, further content is dropped.

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
