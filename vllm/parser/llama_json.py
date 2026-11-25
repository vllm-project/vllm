# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    ToolCallBodyCollection,
    Transition,
)
from vllm.parser.utils import extract_json_member_object

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

TOOL_START = "<|python_tag|>"
LLAMA_OUTPUT_END_TOKENS = (
    "<|eom_id|>",  # end of message
    "<|eot_id|>",  # end of turn
    "<|end_of_text|>",  # end of text
)


def _has_only_output_end_token(text: str, position: int) -> bool:
    return any(
        text.startswith(token, position) and not text[position + len(token) :].strip()
        for token in LLAMA_OUTPUT_END_TOKENS
    )


def _validate_complete_tool_output(text: str) -> bool | None:
    """Validate complete Llama tool-call output"""
    if text.startswith(TOOL_START):
        text = text[len(TOOL_START) :]
    elif not text.startswith("{"):
        return None

    decoder = json.JSONDecoder()
    position = 0
    found = False
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            return found
        try:
            value, end = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            return False
        if (
            not isinstance(value, dict)
            or "name" not in value
            or ("arguments" not in value and "parameters" not in value)
        ):
            return False
        found = True
        position = end
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            return True
        if _has_only_output_end_token(text, position):
            return True
        if text[position] != ";":
            return False
        position += 1
    return found


def _llama_arg_converter(raw_args: str, partial: bool) -> str:
    args = extract_json_member_object(raw_args, "arguments")
    if args is None:
        args = extract_json_member_object(raw_args, "parameters")
    return args if args is not None else ("" if partial else "{}")


@functools.cache
def llama_json_config() -> ParserEngineConfig:
    return ParserEngineConfig(
        name="llama_json",
        initial_state=ParserState.CONTENT,
        wait_for_reasoning=False,
        terminals={
            "TOOL_START": TOOL_START,
        },
        token_id_terminals={
            "TOOL_START": TOOL_START,
        },
        transitions={
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
        },
        arg_converter=_llama_arg_converter,
        complete_tool_call_validator=_validate_complete_tool_output,
        stream_arg_deltas=False,
        tool_args_json=True,
        tool_call_body_collection=ToolCallBodyCollection.SEMICOLON_SEQUENCE,
    )


class LlamaJsonParser(ParserEngine):
    CONFIG_NAME = "llama_json"

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        kwargs.pop("chat_template_kwargs", None)
        kwargs.setdefault("parser_engine_config", llama_json_config())
        super().__init__(tokenizer, tools, **kwargs)
        self._tool_start_token_id = self.vocab.get(TOOL_START)
        # Some Llama templates omit TOOL_START and begin directly with JSON.
        self._at_output_start = True

    def _reset(self, initial_state: ParserState | None = None) -> None:
        super()._reset(initial_state)
        self._at_output_start = True

    def _preprocess_feed(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> tuple[str, Sequence[int]]:
        if not delta_text or not self._at_output_start:
            return delta_text, delta_token_ids

        self._at_output_start = False
        if self._suppress_tool_calls or not delta_text.startswith("{"):
            return delta_text, delta_token_ids

        delta_text = TOOL_START + delta_text
        if not delta_token_ids:
            return delta_text, delta_token_ids
        if self._tool_start_token_id is None:
            return delta_text, []

        return delta_text, [self._tool_start_token_id, *delta_token_ids]
