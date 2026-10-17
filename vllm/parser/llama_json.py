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

The generic incomplete/resumable JSON scanning this format needs (finding a
top-level key's value in truncated text, repairing a truncated value into
closeable JSON, splicing schema-coerced values into a text span without
re-serialising) lives in :mod:`vllm.parser.json_scan` and
:mod:`vllm.parser.utils` -- none of it is Llama-specific, it's just the only
parser that currently needs it. What's here is the envelope shape itself
and the state machine that streams it.
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
from vllm.parser.json_scan import (
    WS,
    KeyScan,
    ValueScan,
    closeable_prefix,
    decode_key,
    scan_json_value,
    scan_top_level,
    top_level_keys,
)
from vllm.parser.utils import splice_types
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
# Matched by position, not this order: the first alias in the text wins.
_ARG_KEYS = ("arguments", "parameters")


def _name_from_span(span: str | None) -> str | None:
    """Validate a top-level ``"name"`` span and decode it."""
    if not span or not span.startswith('"'):
        return None
    if scan_json_value(span, 0) != len(span):
        return None
    try:
        name = json.loads(span)
    except json.JSONDecodeError:
        return None
    return name or None


def _args_value_span(raw: str) -> str | None:
    return scan_top_level(raw, _ARG_KEYS)


def _top_level_name(raw: str) -> str | None:
    """Completed top-level ``"name"`` value from a (possibly incomplete)
    envelope, or ``None``. Unlike the engine's regex extraction, never
    picks up a nested ``"name"``, and only accepts a terminated string
    (an unterminated span ending in an escaped quote isn't complete).
    """
    return _name_from_span(scan_top_level(raw, ("name",)))


def _name_if_envelope(name: str | None, raw: str) -> str | None:
    """*name* is a call's name only if a ``parameters``/``arguments`` key
    sits alongside it: a bare ``{"name": "f"}`` and prose JSON merely
    containing a ``name`` field are both not a call."""
    if name is None:
        return None
    keys = top_level_keys(raw)
    return name if ("arguments" in keys or "parameters" in keys) else None


def _envelope_name(raw: str) -> str | None:
    """Classify *raw* as a tool-call envelope and return its name."""
    return _name_if_envelope(_top_level_name(raw), raw)


def _llama_bare_arg_converter(raw_args: str, partial: bool) -> str:
    """Convert a named choice's bare parameter object to arguments."""
    end, closers = closeable_prefix(raw_args)
    if partial:
        return raw_args[:end]
    if end == 0:
        return "{}"
    return raw_args[:end] + closers


def _llama_arg_converter(raw_args: str, partial: bool) -> str:
    """Carve the arguments value out of the envelope (verbatim, prefix-
    stable; see ``inkling._inkling_arg_converter`` for the rationale), then
    hand off to ``_llama_bare_arg_converter`` to close it into valid JSON."""
    span = _args_value_span(raw_args)
    if span is None:
        return "" if partial else "{}"
    return _llama_bare_arg_converter(span, partial)


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
        self.key: KeyScan | None = None  # args-key scan
        # Where the envelope's top-level "name" is; same resumability need.
        self.name_key: KeyScan | None = None
        self.name_value: ValueScan | None = None
        self.value: ValueScan | None = None  # args value scan
        self.spliced: str = ""  # members already spliced/streamed
        self.cursor: int = 0
        self.member: ValueScan | None = None  # trailing member's scan


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

    def _slot_args(self, dense_idx: int) -> str:
        if dense_idx < len(self._tool_slots):
            return self._tool_slots[dense_idx].args
        return ""

    def _arg_scan(self, idx: int) -> _ArgScan:
        slot = self._tool_slots[idx]
        scan = self._arg_scans.get(idx)
        if scan is None or scan.slot is not slot:
            scan = self._arg_scans[idx] = _ArgScan(slot)
        return scan

    def _try_extract_name(self, idx: int) -> str | None:
        # Top-level only (the engine's regex would match a nested "name"),
        # gated on an args key proving this is actually a call.
        if self._named_bare_args_name is not None:
            return self._named_bare_args_name
        raw = self._tool_slots[idx].args
        span = self._resumable_name_span(idx, raw)
        return _name_if_envelope(_name_from_span(span), raw)

    def _resumable_name_span(self, idx: int, raw: str) -> str | None:
        """``scan_top_level(raw, ("name",))`` carried across feeds."""
        scan = self._arg_scan(idx)
        key = scan.name_key
        if key is None or not key.resumable(raw):
            key = scan.name_key = KeyScan()
            scan.name_value = None
        start = key.start(raw, ("name",))
        if start < 0:
            return None
        value = scan.name_value
        if value is None or not value.resumable(raw):
            value = scan.name_value = ValueScan(start)
        return raw[start : value.end(raw)]

    def _partial_args(self, scan: _ArgScan, raw: str) -> str:
        """``_llama_arg_converter(raw, True)`` with the envelope scan
        carried across feeds instead of restarted on every one."""
        value = scan.value
        if value is None:
            key = scan.key
            if key is None or not key.resumable(raw):
                key = scan.key = KeyScan()
            start = key.start(raw, _ARG_KEYS)
            if start < 0:
                return ""
            value = scan.value = ValueScan(start)
        elif not value.resumable(raw):
            value = scan.value = ValueScan(value.start)
        # A ``None`` end means the value is still open, i.e. runs to the
        # end of the envelope.
        span = raw[value.start : value.end(raw)]
        return span[: closeable_prefix(span)[0]]

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
        member: ValueScan | None = None
        if scan is not None and scan.cursor <= n:
            out.append(scan.spliced)
            cursor = scan.cursor
            i = cursor or 1
            member = scan.member
        tail_end = n
        while True:
            while i < n and (raw[i] in WS or raw[i] == ","):
                i += 1
            if i >= n or raw[i] == "}":
                break
            if raw[i] != '"':
                tail_end = i
                break
            key_end = scan_json_value(raw, i)
            if key_end is None:
                break
            key = decode_key(raw[i + 1 : key_end - 1])
            j = key_end
            while j < n and raw[j] in WS:
                j += 1
            if j >= n:
                break
            if raw[j] != ":":
                tail_end = j
                break
            j += 1
            while j < n and raw[j] in WS:
                j += 1
            if j >= n:
                break
            if member is None or member.start != j or not member.resumable(raw):
                member = ValueScan(j)
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
                else splice_types(raw, j, value_end, schema)
            )
            cursor = i = value_end
            member = None
        settled = "".join(out)
        if scan is not None:
            scan.spliced = settled
            scan.cursor = cursor
            scan.member = member
        return settled + raw[cursor:tail_end]

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
        return splice_types(args_json, 0, len(args_json), {"properties": properties})

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
