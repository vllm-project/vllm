# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 parser for reasoning markers."""

from __future__ import annotations

import functools
from collections.abc import Iterable, Sequence
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

THINK_START = "<mm:think>"
THINK_END = "</mm:think>"


@functools.cache
def minimax_m3_config(thinking: bool = False) -> ParserEngineConfig:
    return ParserEngineConfig(
        name="minimax_m3",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals={
            "THINK_START": THINK_START,
            "THINK_END": THINK_END,
        },
        transitions={
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
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
        },
    )


class MiniMaxM3Parser(ParserEngine):
    """MiniMax M3 parser backed by the declarative parser engine."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self._initial_in_reasoning = chat_kwargs.get("thinking_mode") == "enabled"
        kwargs.setdefault(
            "parser_engine_config",
            minimax_m3_config(thinking=self._initial_in_reasoning),
        )
        super().__init__(tokenizer, tools, **kwargs)
        self._start_token_ids = self._encode_marker(THINK_START)
        self._end_token_ids = self._encode_marker(THINK_END)

    def _encode_marker(self, marker: str) -> tuple[int, ...]:
        try:
            token_ids = self.model_tokenizer.encode(marker, add_special_tokens=False)
        except TypeError:
            token_ids = self.model_tokenizer.encode(marker)
        return tuple(token_ids)

    @staticmethod
    def _contains_token_sequence(
        token_ids: Sequence[int], marker_ids: Sequence[int]
    ) -> bool:
        if not marker_ids or len(marker_ids) > len(token_ids):
            return False
        marker_len = len(marker_ids)
        return any(
            tuple(token_ids[i : i + marker_len]) == tuple(marker_ids)
            for i in range(len(token_ids) - marker_len + 1)
        )

    @staticmethod
    def _rfind_token_sequence(
        token_ids: Sequence[int], marker_ids: Sequence[int]
    ) -> int:
        if not marker_ids or len(marker_ids) > len(token_ids):
            return -1
        marker_len = len(marker_ids)
        for i in range(len(token_ids) - marker_len, -1, -1):
            if tuple(token_ids[i : i + marker_len]) == tuple(marker_ids):
                return i
        return -1

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        start_index = self._rfind_token_sequence(input_ids, self._start_token_ids)
        end_index = self._rfind_token_sequence(input_ids, self._end_token_ids)
        if end_index < 0:
            return False
        if start_index < 0:
            return True
        return end_index > start_index

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if self.reasoning_ended:
            return True
        if self._engine._lexer.buffer:
            return False
        if self._initial_in_reasoning:
            return False
        if self._engine.state == ParserState.CONTENT:
            return bool(input_ids)
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        end_index = self._rfind_token_sequence(input_ids, self._end_token_ids)
        if end_index >= 0:
            return input_ids[end_index + len(self._end_token_ids) :]

        has_start = self._contains_token_sequence(input_ids, self._start_token_ids)
        if self._initial_in_reasoning and not has_start:
            return []

        if not has_start:
            return input_ids
        return []

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        count = 0
        depth = 1 if self._initial_in_reasoning else 0
        i = 0
        while i < len(token_ids):
            if tuple(token_ids[i : i + len(self._start_token_ids)]) == (
                self._start_token_ids
            ):
                depth += 1
                i += len(self._start_token_ids)
                continue
            if tuple(token_ids[i : i + len(self._end_token_ids)]) == (
                self._end_token_ids
            ):
                if depth > 0:
                    depth -= 1
                i += len(self._end_token_ids)
                continue
            if depth > 0:
                count += 1
            i += 1
        return count
