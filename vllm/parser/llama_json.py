# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Llama 3.x/4 JSON tool-call parser (llama3_json / llama4_json).

Format (tool_chat_template_llama3.1/3.2/llama4 JSON templates)::

    [<|python_tag|>]{"name": "f", "parameters": {...}}; {"name": "g", ...}

The payload is a bare JSON envelope with no tool-name state and no end
marker: a ``{`` opens a tool call, the call closes when the envelope's
JSON balances (``tool_call_ends_on_args_balance``), the name is the
envelope's top-level ``"name"`` value, and the ``parameters``/
``arguments`` value is carved out of the envelope as a verbatim
prefix-stable substring.  Separators between calls (``;``, newlines,
or nothing at all under the xgrammar "llama" structural tag) are
dropped like all text after the first completed call, matching the
legacy parser.  A balanced ``{...}`` that never produces a top-level
``"name"`` key was prose JSON, not a tool call; its text is restored
as content in place.
"""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine import ParserEngine, ToolCallSlot
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

PYTHON_TAG = "<|python_tag|>"
_WS = " \t\r\n"
_ARG_KEYS = ("arguments", "parameters")


def _scan_json_value(raw: str, start: int) -> int | None:
    """Return the end index (exclusive) of the JSON value starting at
    ``raw[start]``, or ``None`` while it is still unterminated.

    Handles objects, arrays, strings, and primitives (numbers and
    literals end at a top-level ``,``/``}``/``]`` or whitespace).
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


def _scan_top_level(raw: str, key_names: tuple[str, ...]) -> str | None:
    """Return the raw text span of the first top-level *key_names* value
    in a (possibly incomplete) JSON envelope.

    Verbatim substring (prefix-stable across growing input, required by
    the engine's argument-delta diffing), possibly an unterminated
    prefix; ``None`` when the value has not started.
    """
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
                    last_string = raw[string_start + 1 : i]
            continue
        if ch == '"':
            in_string = True
            string_start = i
        elif ch == ":" and depth == 1 and last_string in key_names:
            value_start = i + 1
            while value_start < len(raw) and raw[value_start] in _WS:
                value_start += 1
            if value_start >= len(raw):
                return None
            value_end = _scan_json_value(raw, value_start)
            if value_end is None:
                return raw[value_start:]
            return raw[value_start:value_end]
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
    return None


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
                    pending = raw[string_start + 1 : i]
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
    """Extract the completed top-level ``"name"`` value from a (possibly
    incomplete) envelope, or ``None``.

    Unlike the engine's regex-based name extraction, this never picks up
    a ``"name"`` key nested inside the parameters object, and only
    accepts a terminated string value (an unterminated span ending in an
    escaped quote must not be misread as complete).
    """
    span = _scan_top_level(raw, ("name",))
    if not span or not span.startswith('"'):
        return None
    if _scan_json_value(span, 0) != len(span):
        return None
    try:
        name = json.loads(span)
    except json.JSONDecodeError:
        return None
    return name or None


def _envelope_name(raw: str, closed: bool) -> str | None:
    """Classify *raw* as a tool-call envelope and return its name.

    A call must carry a ``parameters``/``arguments`` key alongside the
    top-level ``name`` (legacy raised KeyError otherwise), except the
    bare ``{"name": "f"}`` form, decidable only once *closed*.  Prose
    JSON that merely contains a ``name`` field (e.g. a user-data example
    object) is NOT a call — legacy returned it as content.
    """
    name = _top_level_name(raw)
    if name is None:
        return None
    keys = _top_level_keys(raw)
    if "arguments" in keys or "parameters" in keys:
        return name
    if closed and keys == {"name"}:
        return name
    return None


def _llama_arg_converter(raw_args: str, partial: bool) -> str:
    """Carve the arguments value out of the JSON envelope (verbatim,
    prefix-stable; see ``inkling._inkling_arg_converter`` for the full
    rationale)."""
    span = _args_value_span(raw_args)
    if span is None:
        return "" if partial else "{}"
    return span


@functools.cache
def llama_json_config() -> ParserEngineConfig:
    return ParserEngineConfig(
        name="llama_json",
        terminals={
            "PYTHON_TAG": PYTHON_TAG,
            "OPEN_BRACE": "{",
        },
        token_id_terminals={"PYTHON_TAG": PYTHON_TAG},
        transitions={
            # The tag alone opens nothing: ipython-style non-JSON after
            # <|python_tag|> stays content; the "{" starts the call.
            (ParserState.CONTENT, "PYTHON_TAG"): Transition(ParserState.CONTENT, ()),
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
        # Deliberately no vocab check for <|python_tag|>: Llama 4
        # tokenizers lack the token (legacy raised RuntimeError there);
        # token-id resolution silently skips unresolved tokens and the
        # text terminal covers both families.
        self._engine_to_dense: dict[int, int] = {}
        self._phantom_count: int = 0
        self._drop_content: bool = False
        self._held_ws: list[str] = []
        self._closed_dense: set[int] = set()
        kwargs.setdefault("parser_engine_config", llama_json_config())
        super().__init__(tokenizer, tools, **kwargs)

    def _reset(self, initial_state: ParserState | None = None) -> None:
        super()._reset(initial_state=initial_state)
        self._engine_to_dense.clear()
        self._phantom_count = 0
        self._drop_content = False
        self._held_ws.clear()
        self._closed_dense.clear()

    def _check_skip_tool_parsing(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> None:
        super()._check_skip_tool_parsing(request)
        # The base only suppresses for tool_choice="none" WITH tools, but
        # requests without tools default to tool_choice="none" too, and a
        # bare "{" in ordinary content would otherwise be parsed as a
        # tool call and eaten.
        if (
            not self._suppress_tool_calls
            and getattr(request, "tool_choice", None) == "none"
        ):
            self._suppress_tool_calls = True

    def finish_streaming(self) -> DeltaMessage | None:
        delta = super().finish_streaming()
        # The engine skips _events_to_delta entirely when it has nothing
        # buffered, leaving held whitespace unflushed.
        if self._held_ws and not self._drop_content:
            ws = "".join(self._held_ws)
            self._held_ws.clear()
            if delta is None:
                delta = DeltaMessage(content=ws)
            else:
                delta.content = (delta.content or "") + ws
        return delta

    def _try_extract_name(self, idx: int) -> str | None:
        # Top-level extraction only (the engine's regex would match a
        # "name" key nested inside the parameters object), gated on the
        # envelope classification: mid-stream a name is only emitted once
        # an args key proves this is a call; the bare {"name": ...} form
        # is accepted only at close.
        return _envelope_name(
            self._tool_slots[idx].args, closed=idx in self._closed_dense
        )

    def _slot_args(self, dense_idx: int) -> str:
        if dense_idx < len(self._tool_slots):
            return self._tool_slots[dense_idx].args
        return ""

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

        A balanced ``{...}`` with no top-level ``"name"`` was ordinary
        content: its events are replaced in place by a TEXT_CHUNK so the
        swallowed text is restored (legacy fell back to content here),
        and its slot index is recycled so real calls stream with dense
        indices.  After the first real call completes, all further
        content is dropped (legacy returned content=None with calls;
        leading prose is kept per the engine convention).

        Whitespace-only text before any non-whitespace content is held
        rather than forwarded: with a candidate call open, the engine
        would drop it as whitespace-before-tools even when the call turns
        out to be prose JSON, making streamed content chunk-dependent.
        The hold is flushed before the next content and discarded once a
        real call completes (the engine strips it in non-streaming too).
        """
        out: list[SemanticEvent] = []
        pending: dict[int, list[str]] = {}
        for event in events:
            if event.type == EventType.TEXT_CHUNK:
                if self._drop_content:
                    self._held_ws.clear()
                elif not self._content_has_nonws and not event.value.strip():
                    self._held_ws.append(event.value)
                else:
                    self._flush_held_ws(out)
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
                if not slot_named and _envelope_name(accumulated, closed=True) is None:
                    # Phantom: retract this batch's (remapped) events for
                    # the call and restore the full text as content.
                    out = [
                        e
                        for e in out
                        if e.type == EventType.TEXT_CHUNK or e.tool_index != dense_idx
                    ]
                    if dense_idx < len(self._tool_slots):
                        self._tool_slots[dense_idx] = ToolCallSlot()
                    self._phantom_count += 1
                    del self._engine_to_dense[engine_idx]
                    if accumulated and not self._drop_content:
                        self._flush_held_ws(out)
                        out.append(SemanticEvent(EventType.TEXT_CHUNK, accumulated))
                    continue
                self._drop_content = True
                self._held_ws.clear()
                self._closed_dense.add(dense_idx)
            out.append(SemanticEvent(event.type, event.value, dense_idx))
        if finished and not self._drop_content:
            self._flush_held_ws(out)
        return out

    def _events_to_delta(
        self,
        events: list[SemanticEvent],
        finished: bool = False,
    ) -> DeltaMessage | None:
        if self._suppress_tool_calls:
            # Bare-JSON tool markup is indistinguishable from ordinary
            # JSON content, so with tool_choice="none" (or no tools at
            # all) return it as content in place instead of dropping it
            # like marker-based formats do.
            events = [
                SemanticEvent(EventType.TEXT_CHUNK, e.value, e.tool_index)
                if e.value and e.type == EventType.ARG_VALUE_CHUNK
                else e
                for e in events
            ]
        else:
            events = self._filter_events(events, finished=finished)
        return super()._events_to_delta(events, finished=finished)
