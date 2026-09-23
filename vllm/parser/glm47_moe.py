# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-4.7 parser for reasoning and tool calls.

GLM-4.7 uses XML-like tool calls::

    <tool_call>func_name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>

The function name can be followed directly by the first ``<arg_key>`` tag,
and tool calls may have no arguments.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

if TYPE_CHECKING:
    from vllm.entrypoints.generate.base.protocol import ExtractedToolCallInformation
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
ARG_KEY_START = "<arg_key>"
ARG_KEY_END = "</arg_key>"
ARG_VALUE_START = "<arg_value>"
ARG_VALUE_END = "</arg_value>"

# Special tokens that delimit conversation turns in a rendered GLM prompt.
# Reasoning markers belonging to earlier turns must not be mistaken for the
# state of the turn currently being generated.
GLM_TURN_BOUNDARIES = frozenset(
    ("<|system|>", "<|user|>", "<|assistant|>", "<|observation|>")
)

_ARG_RE = re.compile(
    r"<arg_key>(?P<key>.*?)</arg_key>\s*"
    r"<arg_value>(?P<value>.*?)</arg_value>",
    re.DOTALL,
)
_PARTIAL_ARG_RE = re.compile(
    r"<arg_key>(?P<key>.*?)</arg_key>\s*"
    r"<arg_value>(?P<value>.*)$",
    re.DOTALL,
)


def _glm47_arg_converter(raw_args: str, partial: bool) -> str:
    params: dict[str, object] = {}

    for match in _ARG_RE.finditer(raw_args):
        params[match.group("key").strip()] = match.group("value")

    if partial:
        remaining = _ARG_RE.sub("", raw_args)
        match = _PARTIAL_ARG_RE.search(remaining)
        if match:
            key = match.group("key").strip()
            if key:
                params[key] = match.group("value")

    return json.dumps(params, ensure_ascii=False)


@functools.cache
def glm47_moe_config(thinking: bool = True) -> ParserEngineConfig:
    arg_tag_transitions = {
        (ParserState.TOOL_ARGS, terminal): Transition(
            ParserState.TOOL_ARGS,
            (EventType.ARG_VALUE_CHUNK,),
        )
        for terminal in (
            "ARG_KEY_START",
            "ARG_KEY_END",
            "ARG_VALUE_START",
            "ARG_VALUE_END",
        )
    }

    reasoning_terminals = (
        {
            "THINK_START": THINK_START,
            "THINK_END": THINK_END,
        }
        if thinking
        else {}
    )
    reasoning_token_id_terminals = (
        {
            "THINK_START": THINK_START,
            "THINK_END": THINK_END,
        }
        if thinking
        else {}
    )
    reasoning_transitions = (
        {
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
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
        name="glm47_moe",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals={
            **reasoning_terminals,
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
            "ARG_KEY_START": ARG_KEY_START,
            "ARG_KEY_END": ARG_KEY_END,
            "ARG_VALUE_START": ARG_VALUE_START,
            "ARG_VALUE_END": ARG_VALUE_END,
        },
        token_id_terminals={
            **reasoning_token_id_terminals,
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
        },
        transitions={
            **reasoning_transitions,
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_NAME,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_NAME, "ARG_KEY_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.ARG_VALUE_CHUNK,),
            ),
            (ParserState.TOOL_NAME, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            **arg_tag_transitions,
        },
        turn_boundary_tokens=GLM_TURN_BOUNDARIES,
        arg_converter=_glm47_arg_converter,
        stream_arg_deltas=True,
        tool_args_json=False,
        validate_tool_names=True,
    )


@functools.cache
def _glm47_moe_thinking_off_config() -> ParserEngineConfig:
    # Keep the reasoning terminals so a template that ignores
    # ``enable_thinking`` and opens ``<think>`` anyway (e.g. GLM-5.3) can
    # still be parsed.
    return dataclasses.replace(
        glm47_moe_config(thinking=True), initial_state=ParserState.CONTENT
    )


class Glm47MoeParser(ParserEngine):
    """GLM-4.7 parser backed by the declarative parser engine."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = chat_kwargs.get("thinking", None)
        enable_thinking = chat_kwargs.get("enable_thinking", None)
        self.thinking_enabled = (
            True
            if thinking is None and enable_thinking is None
            else bool(thinking) or bool(enable_thinking)
        )
        kwargs.setdefault(
            "parser_engine_config",
            glm47_moe_config()
            if self.thinking_enabled
            else _glm47_moe_thinking_off_config(),
        )
        super().__init__(tokenizer, tools, **kwargs)

    def adjust_initial_state_from_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        if self.thinking_enabled:
            return
        for token_id in reversed(prompt_token_ids):
            if token_id == self._reasoning_start_token_id:
                self._engine.reset(initial_state=ParserState.REASONING)
                self._streaming_initialized = True
                return
            if (
                token_id == self._reasoning_end_token_id
                or token_id in self._turn_boundary_token_ids
            ):
                return

    def _output_closes_open_reasoning(self, model_output: str) -> bool:
        """Whether the output ends a ``<think>`` that the prompt opened.

        The non-streaming path has no prompt ids, so this detects a template
        that ignored ``enable_thinking=False`` from the output alone.
        """
        if self.thinking_enabled:
            return False
        end = model_output.find(THINK_END)
        return end >= 0 and THINK_START not in model_output[:end]

    def _single_pass_parse(
        self,
        text: str,
        token_ids: Sequence[int],
        initial_state: ParserState | None = None,
    ) -> tuple[str | None, str | None, ExtractedToolCallInformation]:
        if initial_state is None and self._output_closes_open_reasoning(text):
            initial_state = ParserState.REASONING
        return super()._single_pass_parse(text, token_ids, initial_state)

    def _emit_name_delta(self, idx: int, deltas, name: str | None) -> None:
        if name is not None:
            name = name.strip()
        super()._emit_name_delta(idx, deltas, name)

    def _handle_tool_end(self, event, deltas) -> None:
        idx = event.tool_index
        if 0 <= idx < len(self._tool_slots):
            self._tool_slots[idx].name = self._tool_slots[idx].name.strip()
        super()._handle_tool_end(event, deltas)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if self._output_closes_open_reasoning(model_output):
            # Re-open the prompt's ``<think>`` so the engine enters REASONING.
            model_output = THINK_START + model_output
        elif not self.thinking_enabled:
            return None, model_output
        return super().extract_reasoning(model_output, request)
