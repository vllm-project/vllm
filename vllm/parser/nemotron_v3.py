# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron V3 parser.

The Nemotron 3 Super model uses the same tool call and reasoning
format as Qwen3 (``<think>``/``</think>`` + ``<tool_call>`` XML).
This config reuses :func:`qwen3_config` with a distinct name.

When ``enable_thinking=False`` or ``force_nonempty_content=True`` and
content is empty, reasoning and content are swapped.

The stock chat template frames each parameter value as ``\\n{value}\\n``,
so one leading and one trailing newline are trimmed as markup. That makes a
value's own edge newline unrecoverable. :class:`NemotronV3CompactParser` is
for a compact template that renders ``<parameter=NAME>{value}</parameter>``
and keeps values verbatim.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING

from vllm.parser.mimo import _mimo_arg_converter
from vllm.parser.qwen3 import (
    CHATML_TURN_BOUNDARIES,
    Qwen3Parser,
    _param_value_extender,
    qwen3_config,
)

if TYPE_CHECKING:
    from vllm.entrypoints.generate.base.protocol import DeltaMessage
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.parser.engine.parser_engine import SemanticEvent
    from vllm.parser.engine.parser_engine_config import ParserEngineConfig
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool


@functools.cache
def nemotron_v3_config(thinking: bool = True) -> ParserEngineConfig:
    return dataclasses.replace(
        qwen3_config(thinking=thinking, turn_boundary_tokens=CHATML_TURN_BOUNDARIES),
        name="nemotron_v3",
        strip_trailing_reasoning_whitespace=True,
    )


@functools.cache
def nemotron_v3_compact_config(thinking: bool = True) -> ParserEngineConfig:
    return dataclasses.replace(
        nemotron_v3_config(thinking=thinking),
        name="nemotron_v3_compact",
        arg_converter=_mimo_arg_converter,
        arg_value_extender=functools.partial(
            _param_value_extender, trim_newlines=False
        ),
    )


class NemotronV3Parser(Qwen3Parser):
    """Nemotron V3 parser: same format as Qwen3, with Nemotron-specific
    behavior: when ``enable_thinking=False`` or
    ``force_nonempty_content=True`` and content is empty, swaps
    reasoning and content.
    """

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = chat_kwargs.get("enable_thinking", True)
        super().__init__(
            tokenizer,
            tools,
            parser_engine_config=self._engine_config(thinking),
            **kwargs,
        )
        self._streamed_reasoning: list[str] = []

    @staticmethod
    def _engine_config(thinking: bool) -> ParserEngineConfig:
        return nemotron_v3_config(thinking=thinking)

    def _reset(self, initial_state=None) -> None:
        super()._reset(initial_state=initial_state)
        self._streamed_reasoning = []

    def _events_to_delta(
        self,
        events: list[SemanticEvent],
        finished: bool = False,
    ) -> DeltaMessage | None:
        delta = super()._events_to_delta(events, finished=finished)
        if delta is not None and delta.reasoning is not None:
            self._streamed_reasoning.append(delta.reasoning)
        return delta

    @staticmethod
    def _should_force_content(
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> bool:
        chat_template_kwargs = getattr(request, "chat_template_kwargs", None)
        return bool(
            chat_template_kwargs
            and (
                chat_template_kwargs.get("enable_thinking") is False
                or chat_template_kwargs.get("force_nonempty_content") is True
            )
        )

    def get_streaming_fallback_content(
        self,
        text: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> str | None:
        if not self._should_force_content(request):
            return None
        return "".join(self._streamed_reasoning) or None

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        reasoning, content = super().extract_reasoning(model_output, request)

        if self._should_force_content(request) and (
            content is None or not content.strip()
        ):
            reasoning, content = content, reasoning

        return reasoning, content


class NemotronV3CompactParser(NemotronV3Parser):
    """Nemotron V3 parser for the compact template, where parameter values
    are rendered verbatim and edge newlines are part of the value.
    """

    @staticmethod
    def _engine_config(thinking: bool) -> ParserEngineConfig:
        return nemotron_v3_compact_config(thinking=thinking)
