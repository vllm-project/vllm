# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-3.5 / Step-3.7 parser for tool calls and reasoning.

Both models use the Qwen3 XML tool-call grammar and ``<think>`` reasoning
unchanged. Two things differ from :class:`Qwen3Parser`:

* The chat template always prefills ``<think>\\n`` and has no switch to turn
  thinking off, so ``enable_thinking`` is ignored.
* The model closes reasoning with ``\\n</think>\\n``, so trailing reasoning
  whitespace is stripped.
"""

from __future__ import annotations

import functools
from dataclasses import replace
from typing import TYPE_CHECKING

from vllm.parser.engine.parser_engine_config import ParserEngineConfig
from vllm.parser.qwen3 import CHATML_TURN_BOUNDARIES, Qwen3Parser, qwen3_config

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool


@functools.cache
def step3p5_config() -> ParserEngineConfig:
    return replace(
        qwen3_config(
            thinking=True,
            name="step3p5",
            turn_boundary_tokens=CHATML_TURN_BOUNDARIES,
        ),
        strip_trailing_reasoning_whitespace=True,
    )


class Step3p5Parser(Qwen3Parser):
    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("parser_engine_config", step3p5_config())
        super().__init__(tokenizer, tools, **kwargs)
        self.thinking_enabled = True
