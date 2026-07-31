# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming parser engine that orchestrates token ID scanning,
incremental lexing, and state-machine-driven semantic event emission."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.incremental_lexer import (
    CONTENT_TERMINAL,
    IncrementalLexer,
    LexerShape,
    LexToken,
    TerminalDef,
)
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.engine.token_id_scanner import (
    DROP_TERMINAL,
    LexerInput,
    PreLexedTerminal,
    TextChunk,
    TokenIDScanner,
)


@dataclass(slots=True)
class _DropInfo:
    lexer_shape: LexerShape
    extra_token_ids: dict[int, str]


def _build_drop_info(
    config: ParserEngineConfig,
    tokenizer,
) -> _DropInfo | None:
    try:
        special_tokens: list[str] = list(tokenizer.all_special_tokens)
        special_ids: list[int] = list(tokenizer.all_special_ids)
    except (AttributeError, NotImplementedError):
        return None

    if not special_tokens:
        return None

    configured_texts = (
        set(config.token_id_terminals.values())
        | set(config.terminals.values())
        | config.preserve_tokens
    )

    extra_token_ids: dict[int, str] = {}
    drop_texts: set[str] = set()
    for text, tid in zip(special_tokens, special_ids):
        if text not in configured_texts:
            extra_token_ids[tid] = DROP_TERMINAL
            drop_texts.add(text)

    if not drop_texts:
        return None

    import regex as re

    drop_terminal_defs = [
        TerminalDef(
            name=DROP_TERMINAL,
            pattern=re.compile(re.escape(text)),
            is_literal=True,
            literal=text,
        )
        for text in drop_texts
    ]

    all_terminal_defs = list(config.terminal_defs) + drop_terminal_defs
    lexer_shape = LexerShape(all_terminal_defs)

    return _DropInfo(
        lexer_shape=lexer_shape,
        extra_token_ids=extra_token_ids,
    )


class StreamingParserEngine:
    """Consumes ``(delta_text, delta_token_ids)`` pairs and produces a
    stream of :class:`SemanticEvent` instances.

    This is the main entry point for streaming parsing.
    Create one per request (it is stateful).

    The pipeline is::

        delta_text + delta_token_ids
            → TokenIDScanner  (special token pre-lexing)
            → IncrementalLexer  (text → terminal tokens with prefix buffering)
            → State Machine  (terminal → semantic events)
            → list[SemanticEvent]

    Usage::

        engine = StreamingParserEngine(config, tokenizer)
        for each streaming delta:
            events = engine.feed(delta_text, delta_token_ids)
            # convert events to DeltaMessage
    """

    def __init__(
        self,
        config: ParserEngineConfig,
        tokenizer,
        initial_state: ParserState | None = None,
        vocab: dict[str, int] | None = None,
    ) -> None:
        self.config = config

        resolved_token_ids: dict[int, str] = {}
        if tokenizer is not None:
            if vocab is None:
                vocab = tokenizer.get_vocab()
            if config.token_id_terminals:
                for terminal_name, token_text in config.token_id_terminals.items():
                    tid = vocab.get(token_text)
                    if tid is not None:
                        resolved_token_ids[tid] = terminal_name

        drop_info: _DropInfo | None = None
        if tokenizer is not None:
            drop_info = _build_drop_info(config, tokenizer)

        lexer_shape = config.lexer_shape
        if drop_info is not None:
            resolved_token_ids.update(drop_info.extra_token_ids)
            lexer_shape = drop_info.lexer_shape

        self._resolved_token_ids = resolved_token_ids
        self._has_drops = drop_info is not None

        self._scanner = TokenIDScanner(
            resolved_token_ids,
            tokenizer,
        )

        self._token_id_terminal_names: frozenset[str] = frozenset(
            resolved_token_ids.values()
        )

        self._lexer = IncrementalLexer(lexer_shape, content_terminal=CONTENT_TERMINAL)

        self._tool_terminals: frozenset[str] = frozenset(
            terminal
            for (state, terminal), tr in config.transitions.items()
            if tr.next_state in self._TOOL_STATES or state in self._TOOL_STATES
        )

        self.skip_tool_parsing = False
        # Function names declared by the request, or None when unknown.
        # Consulted only by transitions with ``validate_tool_name``;
        # set per request by the owning ParserEngine, like
        # ``skip_tool_parsing`` it survives reset().
        self.allowed_tool_names: frozenset[str] | None = None
        # True when the request asked for tool_choice "none".  Recovery
        # transitions are skipped while set, so text that looks like a
        # recovered tool call stays plain content instead of being
        # consumed and then suppressed.  Set per request by the owning
        # ParserEngine; survives reset() like ``skip_tool_parsing``.
        self.suppress_tool_calls = False
        self.reset(initial_state=initial_state)

    def _reset_args_state(self) -> None:
        self._args_buffer: str = ""
        self._args_safe_end: int = 0
        self._args_brace_depth: int = 0
        self._args_in_string: bool = False
        self._args_escape_next: bool = False

    def reset(self, initial_state: ParserState | None = None) -> None:
        """Reset mutable state for reuse across requests.

        Preserves cached immutable structures (compiled terminals,
        resolved token IDs, lexer shape, token text cache) to avoid
        redundant initialization work.
        """
        self.state = (
            initial_state if initial_state is not None else self.config.initial_state
        )
        self.tool_index = -1
        self._ever_had_token_ids = False
        # DO NOT reset skip_tool_parsing here — callers set it before
        # calling methods that trigger reset() (e.g. extract_reasoning),
        # and clearing it silently breaks non-streaming tool-call-as-
        # implicit-reasoning-end (content returns None).
        self._scanner.reset()
        self._lexer.reset()
        self._message_header_buffer = ""
        self._reset_args_state()
        self._recovered_tool_call = False
        self._pending_between_text = ""
        self._hold_active = False
        self._held_events: list[SemanticEvent] = []
        self._held_raw: list[str] = []
        self._held_name: list[str] = []
        self._held_prior_state: ParserState = self.state
        self._held_prior_tool_index: int = -1

    def feed(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> list[SemanticEvent]:
        if delta_token_ids:
            self._ever_had_token_ids = True

        # Fast path: skip scanner and lexer when the delta is plain
        # content with no special tokens and no terminal-starting chars.
        if (
            delta_text
            and not self._lexer.buffer
            and not self._scanner._deferred_terminals
            and self._lexer._literal_first_chars.isdisjoint(delta_text)
        ):
            has_special = False
            for tid in delta_token_ids:
                if tid in self._resolved_token_ids:
                    has_special = True
                    break
            if not has_special:
                return self._emit_for_state(delta_text)

        scanner_items = self._scanner.scan(delta_text, delta_token_ids)

        if len(scanner_items) == 1 and isinstance(scanner_items[0], TextChunk):
            lex_tokens = self._lexer.feed(scanner_items[0].text)
            if len(lex_tokens) == 1 and lex_tokens[0].terminal == CONTENT_TERMINAL:
                text = lex_tokens[0].value
                return self._emit_for_state(text)
            return self._process_lex_tokens(lex_tokens)

        return self._process_scanner_items(scanner_items)

    def _process_scanner_items(
        self, items: Sequence[LexerInput]
    ) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        for item in items:
            if isinstance(item, PreLexedTerminal):
                events.extend(self._process_lex_tokens(self._lexer.flush()))
                events.extend(self._on_terminal(item.terminal, item.text))
            elif isinstance(item, TextChunk):
                events.extend(self._process_lex_tokens(self._lexer.feed(item.text)))
        return events

    def finish(self) -> list[SemanticEvent]:
        events = self._process_scanner_items(self._scanner.flush_pending())

        events.extend(self._process_lex_tokens(self._lexer.flush()))

        if self._hold_active:
            # Stream ended before the recovered tool name completed:
            # the held events never validated, so flush the raw text
            # as content in the pre-recovery state.
            events.extend(self._abort_hold("".join(self._held_raw)))

        if self._args_buffer:
            events.append(
                SemanticEvent(
                    EventType.ARG_VALUE_CHUNK,
                    value=self._args_buffer,
                    tool_index=self.tool_index,
                )
            )
            self._args_buffer = ""
            self._args_safe_end = 0

        if self.state in (
            ParserState.TOOL_PREAMBLE,
            ParserState.TOOL_ARGS,
            ParserState.TOOL_NAME,
            ParserState.TOOL_BETWEEN,
        ):
            if self.tool_index >= 0:
                events.append(
                    SemanticEvent(
                        EventType.TOOL_CALL_END,
                        tool_index=self.tool_index,
                    )
                )
            self.state = ParserState.CONTENT
        elif self.state == ParserState.REASONING:
            events.append(
                SemanticEvent(EventType.REASONING_END, tool_index=self.tool_index)
            )
            self.state = ParserState.CONTENT
        elif self.state == ParserState.MESSAGE_HEADER:
            if self._message_header_buffer:
                events.append(
                    SemanticEvent(
                        EventType.TEXT_CHUNK,
                        value=self._message_header_buffer,
                        tool_index=self.tool_index,
                    )
                )
                self._message_header_buffer = ""
            self.state = ParserState.CONTENT

        return events

    def parse_complete(self, text: str) -> list[SemanticEvent]:
        token_ids: list[int] = []
        events = self.feed(text, token_ids)
        events.extend(self.finish())
        return events

    def _process_lex_tokens(self, tokens: list[LexToken]) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        strict = self._token_id_terminal_names if self._ever_had_token_ids else None
        for tok in tokens:
            if tok.terminal == CONTENT_TERMINAL or (strict and tok.terminal in strict):
                events.extend(self._on_content(tok.value))
            else:
                events.extend(self._on_terminal(tok.terminal, tok.value))
        return events

    _TOOL_STATES = frozenset(
        {
            ParserState.TOOL_PREAMBLE,
            ParserState.TOOL_NAME,
            ParserState.TOOL_ARGS,
            ParserState.TOOL_BETWEEN,
        }
    )

    def _on_terminal(self, terminal: str, value: str) -> list[SemanticEvent]:
        key = (self.state, terminal)
        transition = self.config.transitions.get(key)

        if transition is None:
            if self._has_drops and terminal == DROP_TERMINAL:
                return []
            if self._hold_active and self.state == ParserState.TOOL_NAME:
                # A terminal with no meaning inside a held tool name,
                # for example a real tool call start token, ends the
                # hold: replay the held text as content, then handle
                # the terminal again in the restored state so it keeps
                # its normal meaning.
                events = self._abort_hold("".join(self._held_raw))
                events.extend(self._on_terminal(terminal, value))
                return events
            return self._emit_for_state(value)

        if self.skip_tool_parsing and terminal in self._tool_terminals:
            if self.state == ParserState.MESSAGE_HEADER:
                self.state = ParserState.CONTENT
                self._message_header_buffer = ""
                return [
                    SemanticEvent(
                        EventType.TEXT_CHUNK,
                        value=value,
                        tool_index=self.tool_index,
                    )
                ]
            if EventType.REASONING_END in transition.events:
                self.state = ParserState.CONTENT
                return [
                    SemanticEvent(
                        EventType.REASONING_END,
                        value=value,
                        tool_index=self.tool_index,
                    ),
                    SemanticEvent(
                        EventType.TEXT_CHUNK,
                        value=value,
                        tool_index=self.tool_index,
                    ),
                ]
            content_type = self.config.content_events.get(self.state)
            if content_type is not None:
                return [
                    SemanticEvent(content_type, value=value, tool_index=self.tool_index)
                ]
            return []

        if transition.skip_in_token_id_mode and self._ever_had_token_ids:
            return self._emit_for_state(value)

        return self._apply_transition(transition, value)

    def _emit_for_state(self, text: str) -> list[SemanticEvent]:
        if self._hold_active and self.state == ParserState.TOOL_NAME:
            candidate = "".join(self._held_name) + text
            if not self._can_grow_into_declared_name(candidate):
                # The held text can no longer become a declared tool
                # name, so holding longer would only stall streaming.
                # Release everything consumed so far as content.
                return self._abort_hold("".join(self._held_raw) + text)
            self._held_raw.append(text)
            self._held_name.append(text)
            self._held_events.append(
                SemanticEvent(
                    EventType.TOOL_NAME,
                    value=text,
                    tool_index=self.tool_index,
                )
            )
            return []
        if self.state == ParserState.MESSAGE_HEADER:
            self._message_header_buffer += text
            return []
        if self.state == ParserState.TOOL_ARGS:
            if self.config.tool_args_json:
                return self._feed_args_text(text)
            return [
                SemanticEvent(
                    EventType.ARG_VALUE_CHUNK,
                    value=text,
                    tool_index=self.tool_index,
                )
            ]
        content_type = self.config.content_events.get(self.state)
        if content_type is not None:
            return [SemanticEvent(content_type, value=text, tool_index=self.tool_index)]
        if self._recovered_tool_call and self.state == ParserState.TOOL_BETWEEN:
            # A response that lost its opening wrapper usually loses the
            # closing one too, so text after a recovered invoke is often
            # the rest of the answer rather than padding before the next
            # invoke.  Whitespace is held back because that is what
            # padding looks like; as soon as anything else shows up the
            # whole run is real output and goes out as content.
            self._pending_between_text += text
            if self._pending_between_text.strip():
                held = self._pending_between_text
                self._pending_between_text = ""
                return [
                    SemanticEvent(
                        EventType.TEXT_CHUNK,
                        value=held,
                        tool_index=self.tool_index,
                    )
                ]
        return []

    def _on_content(self, text: str) -> list[SemanticEvent]:
        if not text:
            return []
        return self._emit_for_state(text)

    def _apply_transition(
        self,
        transition: Transition,
        value: str,
    ) -> list[SemanticEvent]:
        if self._hold_active:
            return self._resolve_hold(transition, value)
        if transition.validate_tool_name:
            if self.suppress_tool_calls or self.allowed_tool_names is None:
                # Recovery could never be accepted for this request, so
                # the trigger text stays plain content and nothing is
                # buffered.
                return self._emit_for_state(value)
            return self._begin_hold(transition, value)
        return self._run_transition(transition, value)

    def _begin_hold(
        self,
        transition: Transition,
        value: str,
    ) -> list[SemanticEvent]:
        """Apply a ``validate_tool_name`` transition but hold its events.

        The events (and every TOOL_NAME chunk that follows) stay
        buffered until the name completes and validates, so a false
        positive can be undone without having emitted anything.
        """
        prior_state = self.state
        prior_tool_index = self.tool_index
        self._held_events = self._run_transition(transition, value)
        self._held_raw = [value]
        self._held_name = []
        self._held_prior_state = prior_state
        self._held_prior_tool_index = prior_tool_index
        self._hold_active = True
        self._recovered_tool_call = True
        return []

    def _resolve_hold(
        self,
        transition: Transition,
        value: str,
    ) -> list[SemanticEvent]:
        """End the hold window at the name-completing transition."""
        name = "".join(self._held_name)
        allowed = self.allowed_tool_names
        if allowed is not None and name in allowed:
            events = self._held_events
            self._clear_hold()
            events.extend(self._run_transition(transition, value))
            return events
        return self._abort_hold("".join(self._held_raw) + value)

    def _abort_hold(self, raw: str) -> list[SemanticEvent]:
        """Discard held events and re-emit the raw text as content."""
        self.state = self._held_prior_state
        self.tool_index = self._held_prior_tool_index
        self._recovered_tool_call = self._held_prior_state in self._TOOL_STATES
        self._clear_hold()
        return self._emit_for_state(raw)

    def _clear_hold(self) -> None:
        self._hold_active = False
        self._held_events = []
        self._held_raw = []
        self._held_name = []

    def _can_grow_into_declared_name(self, candidate: str) -> bool:
        """Return True when *candidate* is a prefix of a declared tool name.

        Consulted while a recovery hold is active.  Membership in the
        declared set is the only way a held name can validate, so once
        the text seen so far stops being a prefix of any declared name
        the caller aborts the hold.  This also bounds how much text a
        hold can buffer to the length of the longest declared name.
        """
        allowed = self.allowed_tool_names
        if allowed is None:
            return False
        return any(name.startswith(candidate) for name in allowed)

    def _run_transition(
        self,
        transition: Transition,
        value: str,
    ) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        previous_state = self.state
        message_header = ""

        if (
            self.state == ParserState.TOOL_ARGS
            and transition.next_state != ParserState.TOOL_ARGS
            and self._args_buffer
        ):
            events.append(
                SemanticEvent(
                    EventType.ARG_VALUE_CHUNK,
                    value=self._args_buffer,
                    tool_index=self.tool_index,
                )
            )
            self._args_buffer = ""

        # Whatever is still held between invokes is whitespace padding,
        # which the wrapped path drops too.
        self._pending_between_text = ""

        if previous_state == ParserState.MESSAGE_HEADER:
            message_header = self._message_header_buffer
            self._message_header_buffer = ""

        if transition.next_state not in self._TOOL_STATES:
            self._recovered_tool_call = False

        self.state = transition.next_state

        for event_type in transition.events:
            if event_type == EventType.TOOL_CALL_START:
                self.tool_index += 1
            event_value = (
                message_header
                if previous_state == ParserState.MESSAGE_HEADER
                and event_type == EventType.TEXT_CHUNK
                else value
            )
            if event_type == EventType.TEXT_CHUNK and not event_value:
                continue
            events.append(
                SemanticEvent(
                    event_type,
                    value=event_value,
                    tool_index=self.tool_index,
                )
            )

        if self.state == ParserState.TOOL_ARGS:
            self._args_brace_depth = 0
            self._args_in_string = False
            self._args_escape_next = False
            self._args_safe_end = 0

        return events

    def _feed_args_text(self, text: str) -> list[SemanticEvent]:
        """Feed text into the JSON argument streaming buffer.

        Streams argument characters incrementally while holding back
        closing braces/brackets that might change as more input arrives.
        """
        events: list[SemanticEvent] = []
        for ch in text:
            result = self._feed_args_char(ch)
            events.extend(result)
        return events

    def _feed_args_char(self, ch: str) -> list[SemanticEvent]:
        self._args_buffer += ch

        if self._args_escape_next:
            self._args_escape_next = False
            self._args_safe_end = len(self._args_buffer)
            return self._flush_safe_args()

        if self._args_in_string:
            if ch == "\\":
                self._args_escape_next = True
            elif ch == '"':
                self._args_in_string = False
            self._args_safe_end = len(self._args_buffer)
            return self._flush_safe_args()

        if ch == '"':
            self._args_in_string = True
            self._args_safe_end = len(self._args_buffer)
            return self._flush_safe_args()

        if ch in ("{", "["):
            self._args_brace_depth += 1
            self._args_safe_end = len(self._args_buffer)
            return self._flush_safe_args()

        if ch in ("}", "]"):
            if self._args_brace_depth > 0:
                self._args_brace_depth -= 1
            if self._args_brace_depth == 0:
                return []
            self._args_safe_end = len(self._args_buffer)
            return self._flush_safe_args()

        self._args_safe_end = len(self._args_buffer)
        return self._flush_safe_args()

    def _flush_safe_args(self) -> list[SemanticEvent]:
        """Emit buffered argument characters up to the safe-end watermark.

        Top-level closing braces are held back (safe_end not advanced)
        until confirmed safe by a subsequent character or finish().
        """
        if self._args_safe_end == 0:
            return []
        to_emit = self._args_buffer[: self._args_safe_end]
        self._args_buffer = self._args_buffer[self._args_safe_end :]
        self._args_safe_end = 0
        return [
            SemanticEvent(
                EventType.ARG_VALUE_CHUNK,
                value=to_emit,
                tool_index=self.tool_index,
            )
        ]
