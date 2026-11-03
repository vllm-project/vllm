# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Declarative configuration for model tool-call and reasoning formats.

Each model format is described by a :class:`ParserEngineConfig` that specifies:

* **terminals** – literal strings or regex patterns that delimit the format
  (e.g. ``<tool_call>``, ``</think>``).
* **token_id_terminals** – terminals that should be matched by token ID
  rather than (or in addition to) text.
* **transitions** – a state machine mapping
  ``(state, terminal) → (new_state, events_to_emit)`` that drives semantic
  event generation during streaming.
* **content_events** – what :class:`EventType` to emit for plain content
  (non-terminal text) in each state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property

from vllm.parser.engine.events import EventType

# Maps a raw argument delta to the JSON text it appends to the open trailing
# value, or returns None when the delta may change anything else.
ArgValueExtender = Callable[[str], str | None]

# Validates a complete non-streaming output. True means a valid tool-call
# candidate, False means a malformed candidate that must fall back to content,
# and None means the output is not a tool-call candidate.
CompleteToolCallValidator = Callable[[str], bool | None]


class ParserState(Enum):
    CONTENT = auto()
    REASONING = auto()
    MESSAGE_HEADER = auto()
    TOOL_PREAMBLE = auto()
    TOOL_NAME = auto()
    TOOL_ARGS = auto()
    TOOL_BETWEEN = auto()


class ToolCallBodyCollection(Enum):
    JSON_ARRAY = auto()
    SEMICOLON_SEQUENCE = auto()


@dataclass(frozen=True, slots=True)
class Transition:
    next_state: ParserState
    events: tuple[EventType, ...] = field(default_factory=tuple)
    skip_in_token_id_mode: bool = False


@dataclass(frozen=True)
class ParserEngineConfig:
    """Declarative description of a model's tool-call / reasoning format.

    The engine feeds terminals from the incremental lexer into the
    transition table and emits the corresponding semantic events.
    Content tokens (text between terminals) are classified by the
    current state via ``content_events``.
    """

    name: str

    # A terminal may carry several spellings; the first spelling is the canonical one.
    terminals: Mapping[str, str | tuple[str, ...]] = field(default_factory=dict)

    token_id_terminals: dict[str, str] = field(default_factory=dict)

    transitions: dict[tuple[ParserState, str], Transition] = field(
        default_factory=dict,
    )

    content_events: dict[ParserState, EventType] = field(
        default_factory=lambda: {
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
    )

    initial_state: ParserState = ParserState.CONTENT

    wait_for_reasoning: bool | None = None

    arg_converter: Callable[[str, bool], str] | None = None

    complete_tool_call_validator: CompleteToolCallValidator | None = None

    stream_arg_deltas: bool = True

    tool_args_json: bool = True

    # Framing for a tool region containing multiple top-level JSON objects.
    # The engine emits one tool call per object. JSON_ARRAY is used by Granite;
    # SEMICOLON_SEQUENCE handles Llama's ``{...}; {...}`` format.
    # ...
    tool_call_body_collection: ToolCallBodyCollection | None = None

    arg_structural_chars: frozenset[str] | None = None

    # Called with the raw argument text after each streaming conversion whose
    # prefix-safe JSON extends the JSON streamed so far; returns an extender
    # when the text ends in an open value that is the last value in the
    # converted JSON, else None. While live, the extender receives every delta
    # in place of ``arg_converter`` until it returns None, meaning the delta
    # closed the value. Returned text is streamed as is, so it must match what
    # ``arg_converter`` would serialise for those raw characters. Deltas are
    # lexer output: a terminal is never split across deltas, but the lexer
    # flushes a held partial terminal as text at the end of the stream or
    # before a terminal matched by token ID, so a truncated value may end in
    # a terminal prefix. Requires ``arg_converter`` and ``stream_arg_deltas``.
    arg_value_extender: Callable[[str], ArgValueExtender | None] | None = None

    # Special tokens exempt from auto-drop but not state-machine terminals.
    preserve_tokens: frozenset[str] = field(default_factory=frozenset)

    # Special tokens delimiting conversation turns in the prompt only
    # considers reasoning markers after the last boundary token.
    turn_boundary_tokens: frozenset[str] = field(default_factory=frozenset)

    # Prevents trailing-whitespace accumulation across multi-turn conversations.
    strip_trailing_reasoning_whitespace: bool = True

    # Drop content that is entirely whitespace when tool calls follow.
    drop_whitespace_only_content_before_tools: bool = True

    # .strip() content text when tool calls are present.
    strip_content_whitespace_with_tools: bool = True

    # Reject tool calls whose names are absent from the request tools.
    validate_tool_names: bool = False

    def __post_init__(self) -> None:
        if self.arg_value_extender is not None and (
            self.arg_converter is None or not self.stream_arg_deltas
        ):
            raise ValueError(
                f"{self.name}: arg_value_extender requires arg_converter "
                "and stream_arg_deltas=True"
            )

    def terminal_literal(self, name: str) -> str | None:
        """Canonical spelling of terminal *name*, or ``None`` if undeclared."""
        value = self.terminals.get(name)
        if isinstance(value, tuple):
            return value[0] if value else None
        return value

    @property
    def terminal_literals(self) -> frozenset[str]:
        """Every declared spelling across all terminals."""
        return frozenset(
            lit
            for value in self.terminals.values()
            for lit in ((value,) if isinstance(value, str) else value)
        )

    @cached_property
    def terminal_defs(self):
        from vllm.parser.engine.incremental_lexer import terminals_from_literals

        return terminals_from_literals(self.terminals)

    @cached_property
    def lexer_shape(self):
        from vllm.parser.engine.incremental_lexer import LexerShape

        return LexerShape(self.terminal_defs)
