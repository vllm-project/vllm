# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K2 parser for reasoning and tool calls.

Kimi K2 tool call format::

    <|tool_calls_section_begin|>
    <|tool_call_begin|>functions.get_weather:0
    <|tool_call_argument_begin|>{"city": "Tokyo"}<|tool_call_end|>
    <|tool_calls_section_end|>

The header before ``<|tool_call_argument_begin|>`` is Kimi's native tool
call id. The function name is the final component before ``:N``.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import regex as re

from vllm.entrypoints.openai.engine.protocol import DeltaFunctionCall, DeltaToolCall
from vllm.parser.engine.events import EventType
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

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_SECTION_START = "<|tool_calls_section_begin|>"
TOOL_SECTION_END = "<|tool_calls_section_end|>"
TOOL_CALL_START = "<|tool_call_begin|>"
TOOL_CALL_END = "<|tool_call_end|>"
TOOL_ARG_START = "<|tool_call_argument_begin|>"

_TOOL_ID_RE = re.compile(r"(?P<id>.+:\d+)")


@functools.cache
def kimi_k2_config(thinking: bool = True) -> ParserEngineConfig:
    reasoning_terminals = (
        {
            "THINK_START": THINK_START,
            "THINK_END": THINK_END,
        }
        if thinking
        else {}
    )
    reasoning_transitions = (
        {
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
        }
        if thinking
        else {}
    )

    return ParserEngineConfig(
        name="kimi_k2",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals={
            **reasoning_terminals,
            "TOOL_SECTION_START": TOOL_SECTION_START,
            "TOOL_SECTION_END": TOOL_SECTION_END,
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
            "ARG_START": TOOL_ARG_START,
        },
        token_id_terminals={
            **reasoning_terminals,
            "TOOL_SECTION_START": TOOL_SECTION_START,
            "TOOL_SECTION_END": TOOL_SECTION_END,
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
            "ARG_START": TOOL_ARG_START,
        },
        transitions={
            **reasoning_transitions,
            (ParserState.REASONING, "TOOL_SECTION_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "TOOL_SECTION_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
            (ParserState.TOOL_PREAMBLE, "TOOL_START"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_NAME, "ARG_START"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_SECTION_END"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_START"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            # Keep the parser in a tool state after the section closes so
            # trailing model text after native tool calls is suppressed.
            (ParserState.TOOL_PREAMBLE, "TOOL_SECTION_END"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_SECTION_END"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
        },
        stream_arg_deltas=True,
        tool_args_json=True,
        strip_trailing_reasoning_whitespace=True,
        drop_whitespace_only_content_before_tools=True,
        strip_content_whitespace_with_tools=False,
        validate_tool_names=False,
    )


class KimiK2Parser(ParserEngine):
    """Kimi K2 parser backed by the declarative parser engine."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self.thinking_enabled = bool(chat_kwargs.get("thinking", True))
        kwargs.setdefault(
            "parser_engine_config",
            kimi_k2_config(thinking=self.thinking_enabled),
        )
        super().__init__(tokenizer, tools, **kwargs)

        vocab = self.vocab
        self._start_token_id = vocab.get(THINK_START)
        self._end_token_id = vocab.get(THINK_END)
        self._tool_section_start_token_id = vocab.get(TOOL_SECTION_START)

    @staticmethod
    def _extract_tool_id_and_name(header: str | None) -> tuple[str | None, str | None]:
        if header is None:
            return None, None
        match = _TOOL_ID_RE.match(header.strip())
        if not match:
            return None, None

        tool_id = match.group("id").strip()
        tool_name = tool_id.split(":")[0].split(".")[-1]
        return tool_id, tool_name

    def _emit_name_delta(
        self,
        idx: int,
        deltas: list[DeltaToolCall],
        name: str | None,
    ) -> None:
        tool_id, tool_name = self._extract_tool_id_and_name(name)
        if not tool_name:
            if 0 <= idx < len(self._tool_slots):
                self._tool_slots[idx].name = ""
            return

        slot = self._tool_slots[idx]
        slot.id = tool_id or ""
        super()._emit_name_delta(idx, deltas, tool_name)

    def _handle_tool_end(self, event, deltas) -> None:
        idx = event.tool_index
        if 0 <= idx < len(self._tool_slots) and not self._tool_slots[idx].name_sent:
            tool_id, tool_name = self._extract_tool_id_and_name(
                self._tool_slots[idx].name
            )
            if tool_name:
                self._tool_slots[idx].id = tool_id or ""
                self._tool_slots[idx].name = tool_name
        super()._handle_tool_end(event, deltas)

    def _handle_arg_chunk(self, event, deltas) -> None:
        idx = event.tool_index
        name_sent_before = (
            0 <= idx < len(self._tool_slots) and self._tool_slots[idx].name_sent
        )
        super()._handle_arg_chunk(event, deltas)
        if (
            event.value
            and not name_sent_before
            and 0 <= idx < len(self._tool_slots)
            and self._tool_slots[idx].name_sent
        ):
            deltas.append(
                DeltaToolCall(
                    index=idx,
                    function=DeltaFunctionCall(arguments=event.value),
                )
            )

    def _extract_args_json(self, raw_args: str, func_name: str) -> str:
        return raw_args.strip() or "{}"

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        if not self.thinking_enabled:
            return True

        start_id = self._start_token_id
        end_id = self._end_token_id
        tool_section_id = self._tool_section_start_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            token_id = input_ids[i]
            if start_id is not None and token_id == start_id:
                return False
            if end_id is not None and token_id == end_id:
                return True
            if tool_section_id is not None and token_id == tool_section_id:
                return True
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self.thinking_enabled:
            return input_ids

        end_id = self._end_token_id
        if end_id is not None and end_id in input_ids:
            end_idx = len(input_ids) - 1 - input_ids[::-1].index(end_id)
            return input_ids[end_idx + 1 :]

        tool_section_id = self._tool_section_start_token_id
        if tool_section_id is not None and tool_section_id in input_ids:
            section_idx = len(input_ids) - 1 - input_ids[::-1].index(tool_section_id)
            return input_ids[section_idx:]

        return []

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            return None, model_output

        reasoning, content = super().extract_reasoning(model_output, request)
        if model_output.startswith(THINK_START + THINK_END):
            reasoning = ""
        return reasoning, content

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        if not self.thinking_enabled:
            return 0
        return super().count_reasoning_tokens(token_ids)
