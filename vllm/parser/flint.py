# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flint parser for reasoning and tool calls.

Flint uses the GLM-4.7 XML tool-call format, but its generation prompt ends at
``<|assistant|>`` without opening ``<think>``: the model decides on each turn
whether to reason, so a reply without ``<think>`` is content. The chat template
has no thinking switch, so ``thinking`` / ``enable_thinking`` have no effect.
"""

from __future__ import annotations

import functools
from dataclasses import replace
from typing import TYPE_CHECKING

from vllm.parser.engine.adapters import make_adapters
from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import ParserEngineConfig, ParserState
from vllm.parser.glm47_moe import Glm47MoeParser, glm47_moe_config

if TYPE_CHECKING:
    from vllm.entrypoints.generate.base.protocol import DeltaMessage
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool


@functools.cache
def flint_config() -> ParserEngineConfig:
    return replace(
        glm47_moe_config(),
        name="flint",
        initial_state=ParserState.CONTENT,
        # Structured outputs still start after </think> or at <tool_call>, so
        # a reply may open <think> when the request has tools.
        wait_for_reasoning=True,
    )


class FlintParser(Glm47MoeParser):
    """Flint parser backed by the GLM-4.7 parser engine."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        self.thinking_enabled = True
        kwargs.setdefault("parser_engine_config", flint_config())
        ParserEngine.__init__(self, tokenizer, tools, **kwargs)

    def _events_to_delta(
        self,
        events: list[SemanticEvent],
        finished: bool = False,
    ) -> DeltaMessage | None:
        if not self._reasoning_ended:
            # A reply that starts with text or a tool call has no reasoning.
            # Ending it here lets a delegating parser hand the stream to the
            # tool parser.
            for event in events:
                if event.type is EventType.REASONING_START:
                    break
                if event.type is EventType.TOOL_CALL_START or (
                    event.type is EventType.TEXT_CHUNK and event.value.strip()
                ):
                    self._reasoning_ended = True
                    break
        return super()._events_to_delta(events, finished)


(
    FlintParserReasoningAdapter,
    FlintParserToolAdapter,
) = make_adapters(FlintParser)
