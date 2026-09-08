# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Granite 4.2 thinking parser.

Granite 4.2 uses the same tool call and reasoning format as Nemotron V3
(``<think>``/``</think>`` + ``<tool_call>`` XML).  This config reuses
:class:`NemotronV3Parser` with two Granite-specific behaviors:

1. Strip leading newlines from content — the Granite chat template writes
   ``\\n</think>\\n``, so the first content character is always ``\\n``.
2. Inherit the Nemotron V3 reasoning-to-content swap for
   ``enable_thinking=False`` / ``force_nonempty_content=True``.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING

from vllm.parser.nemotron_v3 import NemotronV3Parser, nemotron_v3_config
from vllm.parser.qwen3 import Qwen3Parser

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
def granite_thinking_config(thinking: bool = True) -> ParserEngineConfig:
    return dataclasses.replace(
        nemotron_v3_config(thinking=thinking),
        name="granite_thinking_parser",
    )


class GraniteThinkingParser(NemotronV3Parser):
    """Granite 4.2 parser: same format as Nemotron V3, with
    leading-newline stripping on content after ``</think>``.

    The ``reasoning_ended`` property is overridden to delay the
    reasoning-to-content transition reported to
    :class:`~vllm.parser.abstract_parser.DelegatingParser` until the
    template's leading newline has been consumed.  Without this delay,
    ``DelegatingParser.parse_delta`` passes newline tokens through a
    raw pass-through path that bypasses the engine's
    ``_events_to_delta`` (and therefore our stripping logic).
    """

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = chat_kwargs.get("enable_thinking", True)
        kwargs.pop("parser_engine_config", None)
        Qwen3Parser.__init__(
            self,
            tokenizer,
            tools,
            parser_engine_config=granite_thinking_config(thinking=thinking),
            **kwargs,
        )
        self._streamed_reasoning: list[str] = []
        self._content_started = False

    @property
    def reasoning_ended(self) -> bool:
        # Internal engine logic uses _reasoning_ended directly; this
        # property is only read by DelegatingParser (via the adapter's
        # has_engine_confirmed_reasoning_end) to decide when to stop
        # routing deltas through extract_reasoning_streaming.
        if not self._reasoning_ended:
            return False
        return self._content_started

    def _reset(self, initial_state=None) -> None:
        super()._reset(initial_state=initial_state)
        self._content_started = False

    def _events_to_delta(
        self,
        events: list[SemanticEvent],
        finished: bool = False,
    ) -> DeltaMessage | None:
        delta = super()._events_to_delta(events, finished=finished)
        if (
            delta is not None
            and delta.content is not None
            and not self._content_started
        ):
            stripped = delta.content.lstrip("\n")
            if not stripped:
                delta.content = None
                if delta.reasoning is None and not delta.tool_calls:
                    return None
            else:
                self._content_started = True
                delta.content = stripped
        return delta

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        reasoning, content = Qwen3Parser.extract_reasoning(self, model_output, request)

        if content is not None:
            content = content.lstrip("\n") or None

        if self._should_force_content(request) and (
            content is None or not content.strip()
        ):
            reasoning, content = content, reasoning

        return reasoning, content
