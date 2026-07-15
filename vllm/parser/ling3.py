# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ling3 parser for reasoning and tool calls.

Ling3 uses the same XML tool-call format as GLM-4.7, but its chat template
defaults thinking off.  When thinking is enabled, Ling3 follows the GLM-style
``<think>`` / ``</think>`` reasoning format and treats ``<tool_call>`` as an
implicit reasoning terminator.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.engine.adapters import make_adapters
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.glm47_moe import (
    THINK_END,
    THINK_START,
    Glm47MoeParser,
    glm47_moe_config,
)

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool


class Ling3Parser(Glm47MoeParser):
    """Ling3 parser backed by the GLM XML parser engine."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = chat_kwargs.get("thinking", None)
        enable_thinking = chat_kwargs.get("enable_thinking", None)
        self.thinking_enabled = bool(thinking) or bool(enable_thinking)
        parser_config = replace(
            glm47_moe_config(thinking=self.thinking_enabled),
            name="ling3",
        )
        kwargs.setdefault(
            "parser_engine_config",
            parser_config,
        )
        ParserEngine.__init__(self, tokenizer, tools, **kwargs)

    @property
    def reasoning_start_str(self) -> str:
        return THINK_START

    @property
    def reasoning_end_str(self) -> str:
        return THINK_END

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            return None, model_output

        reasoning, content = super().extract_reasoning(model_output, request)
        if reasoning and not content and "<tool_call>" not in model_output:
            return None, reasoning
        return reasoning, content


(
    Ling3ParserReasoningAdapter,
    Ling3ParserToolAdapter,
) = make_adapters(Ling3Parser)
