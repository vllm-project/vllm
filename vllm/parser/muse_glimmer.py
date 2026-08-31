# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseGlimmer parser: channel-scoped messages parsed by a single state machine.

MuseGlimmer writes every assistant turn as a sequence of channel-scoped
messages (the HuggingFace ``MUSE_GLIMMER_RESPONSE_SCHEMA`` contract)::

    <|start|>assistant to=self<|message|>...reasoning...<|eom|>
    <|start|>assistant to=<tool>.<fn><|message|>...ATEM tool call...<|eom|>
    <|start|>assistant to=user<|message|>...final answer...<|eot|>

The prompt ends with ``<|start|>assistant``, so generation begins inside a
message header (``initial_state=MESSAGE_HEADER``) and the first emitted text
is typically `` to=self<|message|>``. A bare ``<|message|>`` header (no
recipient) opens untagged visible content.

Channel headers carry a dynamic recipient, so the tool-channel header is a
regex terminal (``to=<name><|message|>``) rather than a literal. Requiring
the complete header is what keeps prose ``to=`` inside reasoning from being
misread as a channel switch, while a bare tool header following an
unterminated reasoning block (a known model defect: no ``<|eom|>`` before
the header) still ends reasoning and routes the tool call.

Channel scoping falls out of the state machine: ATEM markup echoed inside a
``to=self`` or ``to=user`` body has no transition and remains body text, so
it can never parse as a real tool call.

Framing markers are not guaranteed to be single vocab tokens across every
checkpoint's tokenizer, so the grammar is text-based (no token-id
terminals), like the Inkling parser.

Only a real tool channel emits ``REASONING_END``: a ``to=user`` answer is
surfaced by the reasoning pass itself and must not flip the phase machine
over to the tool parser (matching the legacy parser contract).

Usage: ``--reasoning-parser muse_glimmer``
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

TURN_START = "<|start|>"
MESSAGE = "<|message|>"
EOM = "<|eom|>"
EOT = "<|eot|>"
REASONING_HEADER = "to=self<|message|>"
USER_HEADER = "to=user<|message|>"
TOOL_HEADER_PATTERN = r"to=[A-Za-z0-9_.\-]+<\|message\|>"


@functools.cache
def muse_glimmer_config() -> ParserEngineConfig:
    # Terminal labels: THINK_START/THINK_END are what the engine keys its
    # reasoning plumbing on (``_has_reasoning``, ``reasoning_start_str``), so
    # the reasoning header and ``<|eom|>`` carry those labels; the transition
    # table carries the actual semantics. HEADER_TOOL also matches the self/
    # user headers, but the lexer prefers the longer/equal literal match.
    H = ParserState.MESSAGE_HEADER
    R = ParserState.REASONING
    C = ParserState.CONTENT
    T = ParserState.TOOL_PREAMBLE
    transitions: dict[tuple[ParserState, str], Transition] = {}

    # Between messages / inside a header: route on the completed header.
    transitions[(H, "THINK_START")] = Transition(R, ())
    transitions[(H, "USER_START")] = Transition(C, ())
    transitions[(H, "HEADER_TOOL")] = Transition(T, (EventType.REASONING_END,))
    # Bare header, no recipient: untagged visible content.
    transitions[(H, "MSG")] = Transition(C, ())

    # A reasoning body normally closes at <|eom|>; more blocks may follow, so
    # no REASONING_END is emitted here. The bare-header defect transitions
    # exit reasoning directly when the model skips the <|eom|>.
    transitions[(R, "THINK_START")] = Transition(R, ())
    transitions[(R, "USER_START")] = Transition(C, ())
    transitions[(R, "HEADER_TOOL")] = Transition(T, (EventType.REASONING_END,))

    # Defensive re-entries from a content body.
    transitions[(C, "THINK_START")] = Transition(R, ())
    transitions[(C, "USER_START")] = Transition(C, ())
    transitions[(C, "HEADER_TOOL")] = Transition(T, (EventType.REASONING_END,))
    transitions[(C, "MSG")] = Transition(C, ())

    # NOTE: no (T, MSG) or (T, THINK_START) transitions: a terminal becomes a
    # "tool terminal" for the skip_tool_parsing gate as soon as any transition
    # touches a tool state, and that gate must not hijack the bare-header and
    # reasoning-open transitions in the reasoning pass.
    transitions[(T, "HEADER_TOOL")] = Transition(T, ())

    # Message terminators and turn openers all return to the header state;
    # any text between messages is header metadata (e.g. "assistant"), not
    # assistant output, and is dropped with the header buffer.
    for state in (H, R, C, T):
        transitions[(state, "THINK_END")] = Transition(H, ())
        transitions[(state, "EOT")] = Transition(H, ())
        transitions[(state, "TURN_START")] = Transition(H, ())

    return ParserEngineConfig(
        name="muse_glimmer",
        initial_state=H,
        terminals={
            "THINK_START": REASONING_HEADER,
            "USER_START": USER_HEADER,
            "MSG": MESSAGE,
            "TURN_START": TURN_START,
            "THINK_END": EOM,
            "EOT": EOT,
        },
        regex_terminals={
            "HEADER_TOOL": TOOL_HEADER_PATTERN,
        },
        token_id_terminals={},
        transitions=transitions,
        tool_args_json=False,
        strip_trailing_reasoning_whitespace=False,
        strip_content_whitespace_with_tools=False,
    )


class MuseGlimmerParser(ParserEngine):
    CONFIG_NAME = "muse_glimmer"

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("parser_engine_config", muse_glimmer_config())
        super().__init__(tokenizer, tools, **kwargs)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # Content-id slicing is unreliable for multi-token framing markers;
        # the serving path splits on text via extract_reasoning() instead.
        return []
