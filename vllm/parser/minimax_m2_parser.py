# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
MiniMax M2 Parser - A unified parser for MiniMax M2 models.

This parser combines the existing MiniMaxM2ReasoningParser and
MinimaxM2ToolParser into a single unified interface by delegating
to those implementations.
"""

from vllm.logger import init_logger
from vllm.parser.abstract_parser import DelegatingParser
from vllm.reasoning.minimax_m2_reasoning_parser import MiniMaxM2ReasoningParser
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.minimax_m2_tool_parser import MinimaxM2ToolParser

logger = init_logger(__name__)


class MiniMaxM2Parser(DelegatingParser):
    """
    Unified parser for MiniMax M2 models that handles both reasoning
    extraction and tool call parsing.

    This parser delegates to the existing implementations:
    - MiniMaxM2ReasoningParser for reasoning extraction
    - MinimaxM2ToolParser for tool call parsing

    MiniMax M2 models have two special behaviors:
    1. Reasoning: They don't generate <think> start token, only </think> end
       token. All content before </think> is reasoning, content after is the
       actual response.
    2. Tool Calls: They use <minimax:tool_call>...</minimax:tool_call> tags
       with <invoke name="...">...</invoke> and <parameter name="...">...</parameter>
       syntax.
    """

    # Class-level parser classes for compatibility
    reasoning_parser_cls = MiniMaxM2ReasoningParser
    tool_parser_cls = MinimaxM2ToolParser

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        # Initialize the underlying parsers
        self._reasoning_parser = MiniMaxM2ReasoningParser(tokenizer)
        self._tool_parser = MinimaxM2ToolParser(tokenizer)

        logger.debug(
            "vLLM Successfully initialized parser %s!", self.__class__.__name__
        )
