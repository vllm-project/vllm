# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K2 parser.

Handles <think>…</think> reasoning plus structured tool calls in a single
state machine::

    <think>
    ...reasoning...</think>
    <|tool_calls_section_begin|>
    <|tool_call_begin|>functions.name:0<|tool_call_argument_begin|>{"k":"v"}<|tool_call_end|>
    <|tool_calls_section_end|>

Reasoning may also end implicitly when <|tool_calls_section_begin|> appears.
Tool arguments are standard JSON, so no custom arg_converter is needed.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import regex as re

from vllm.entrypoints.openai.engine.protocol import DeltaFunctionCall, DeltaToolCall
from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine import ParserEngine
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

# Parses Kimi K2 tool call IDs of the form [prefix.]name[:index].
# Examples: "functions.get_weather:0" -> "get_weather", "search:1" -> "search"
_KIMI_FUNC_NAME_RE = re.compile(r"^(?:[^.]+\.)?(.+?)(?::\d+)?$")

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_SECTION_START = "<|tool_calls_section_begin|>"
TOOL_SECTION_END = "<|tool_calls_section_end|>"
TOOL_CALL_START = "<|tool_call_begin|>"
TOOL_ARG_START = "<|tool_call_argument_begin|>"
TOOL_CALL_END = "<|tool_call_end|>"

# Tokens that must not leak into response content as plain text.
_KIMI_K2_DROP_TOKENS: frozenset[str] = frozenset(
    {
        THINK_START,
        THINK_END,
        TOOL_SECTION_START,
        TOOL_SECTION_END,
        TOOL_CALL_START,
        TOOL_ARG_START,
        TOOL_CALL_END,
    }
)


@functools.cache
def kimi_k2_config(thinking: bool = True) -> ParserEngineConfig:
    """Return the ParserEngineConfig for Kimi K2.

    Args:
        thinking: When True (default) the model starts in REASONING state and
            emits reasoning content before tool calls or plain content.
            When False the model starts in CONTENT state (thinking disabled via
            chat_template_kwargs).
    """
    terminals: dict[str, str] = {
        "TOOL_SECTION_START": TOOL_SECTION_START,
        "TOOL_CALL_START": TOOL_CALL_START,
        "TOOL_ARG_START": TOOL_ARG_START,
        "TOOL_CALL_END": TOOL_CALL_END,
        "TOOL_SECTION_END": TOOL_SECTION_END,
    }
    token_id_terminals: dict[str, str] = {
        "TOOL_SECTION_START": TOOL_SECTION_START,
        "TOOL_CALL_START": TOOL_CALL_START,
        "TOOL_CALL_END": TOOL_CALL_END,
    }

    transitions: dict[tuple[ParserState, str], Transition] = {
        # <|tool_calls_section_begin|> in CONTENT: absorbed (marks section start)
        (ParserState.CONTENT, "TOOL_SECTION_START"): Transition(
            ParserState.CONTENT, ()
        ),
        # Individual tool call
        (ParserState.CONTENT, "TOOL_CALL_START"): Transition(
            ParserState.TOOL_NAME, (EventType.TOOL_CALL_START,)
        ),
        (ParserState.TOOL_NAME, "TOOL_ARG_START"): Transition(
            ParserState.TOOL_ARGS, ()
        ),
        (ParserState.TOOL_ARGS, "TOOL_CALL_END"): Transition(
            ParserState.CONTENT, (EventType.TOOL_CALL_END,)
        ),
        # Absorb stray end tokens in CONTENT (e.g. between tool calls)
        (ParserState.CONTENT, "TOOL_CALL_END"): Transition(ParserState.CONTENT, ()),
        (ParserState.CONTENT, "TOOL_SECTION_END"): Transition(ParserState.CONTENT, ()),
    }

    if thinking:
        terminals["THINK_START"] = THINK_START
        terminals["THINK_END"] = THINK_END
        token_id_terminals["THINK_START"] = THINK_START
        token_id_terminals["THINK_END"] = THINK_END

        transitions.update(
            {
                # Absorb <think> when already in REASONING (model starts here)
                (ParserState.REASONING, "THINK_START"): Transition(
                    ParserState.REASONING, ()
                ),
                # Explicit reasoning end
                (ParserState.REASONING, "THINK_END"): Transition(
                    ParserState.CONTENT, (EventType.REASONING_END,)
                ),
                # Implicit reasoning end via tool section
                (ParserState.REASONING, "TOOL_SECTION_START"): Transition(
                    ParserState.CONTENT, (EventType.REASONING_END,)
                ),
            }
        )

    used_tokens = set(terminals.values())
    drop_tokens = _KIMI_K2_DROP_TOKENS - used_tokens

    return ParserEngineConfig(
        name="kimi_k2",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals=terminals,
        token_id_terminals=token_id_terminals,
        transitions=transitions,
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
        tool_args_json=True,
        drop_tokens=frozenset(drop_tokens),
    )


class KimiK2Parser(ParserEngine):
    """Kimi K2 parser: ``<think>`` reasoning + structured tool calls.

    - Starts in REASONING state by default (thinking enabled)
    - <|tool_calls_section_begin|> implicitly ends reasoning
    - Tool arguments are standard JSON (no custom converter needed)
    """

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self._thinking_enabled = bool(chat_kwargs.get("thinking", True))
        super().__init__(
            tokenizer,
            tools,
            parser_engine_config=kimi_k2_config(self._thinking_enabled),
            **kwargs,
        )
        vocab = self.vocab
        self._tool_section_start_token_id: int | None = vocab.get(TOOL_SECTION_START)

    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Keep special tokens visible when thinking or tools are active."""
        request = super().adjust_request(request)
        chat_template_kwargs = getattr(request, "chat_template_kwargs", None) or {}
        thinking = chat_template_kwargs.get("thinking", True)
        has_tools = bool(getattr(request, "tools", None))
        tools_active = has_tools and request.tool_choice != "none"
        if not thinking and not tools_active:
            request.skip_special_tokens = True
        return request

    def _handle_tool_name(self, event: SemanticEvent) -> None:
        idx = event.tool_index
        slot = self._tool_slots[idx]
        # Accumulate raw tool call ID in slot.id so _ensure_tool_id won't
        # overwrite it with a generated one later.
        slot.id += event.value
        raw_id = slot.id.strip()
        slot.id = raw_id  # Keep stored ID clean (no trailing whitespace).
        # Parse clean function name: strip "prefix." and ":index" parts.
        m = _KIMI_FUNC_NAME_RE.match(raw_id)
        slot.name = m.group(1) if m and m.group(1) else raw_id

    def _handle_arg_chunk(
        self,
        event: SemanticEvent,
        deltas: list[DeltaToolCall],
    ) -> None:
        idx = event.tool_index
        was_name_sent = (
            self._tool_slots[idx].name_sent if idx < len(self._tool_slots) else True
        )
        super()._handle_arg_chunk(event, deltas)
        # The parent emits the name delta but skips the arg delta for the chunk
        # that triggers name emission. Emit the arg here to avoid losing it.
        if not was_name_sent and idx < len(self._tool_slots):
            slot = self._tool_slots[idx]
            if slot.name_sent and event.value:
                arg_delta = self._compute_arg_delta(idx, event.value)
                if arg_delta:
                    deltas.append(
                        DeltaToolCall(
                            index=idx,
                            function=DeltaFunctionCall(arguments=arg_delta),
                        )
                    )

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        start_id = self._reasoning_start_token_id
        end_id = self._reasoning_end_token_id
        tool_section_id = self._tool_section_start_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            tid = input_ids[i]
            if start_id is not None and tid == start_id:
                return False
            if end_id is not None and tid == end_id:
                return True
            if tool_section_id is not None and tid == tool_section_id:
                return True
        return not self._thinking_enabled
