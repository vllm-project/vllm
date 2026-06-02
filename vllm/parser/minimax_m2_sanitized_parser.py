# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.parser.abstract_parser import DelegatingParser
from vllm.reasoning.minimax_m2_sanitized_reasoning_parser import (
    MiniMaxM2SanitizedReasoningParser,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.minimax_m2_sanitized_tool_parser import (
    MinimaxM2SanitizedToolParser,
)

logger = init_logger(__name__)


class MiniMaxM2SanitizedParser(DelegatingParser):
    """Unified MiniMax M2 parser with conservative path normalization."""

    reasoning_parser_cls = MiniMaxM2SanitizedReasoningParser
    tool_parser_cls = MinimaxM2SanitizedToolParser

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self._reasoning_parser = MiniMaxM2SanitizedReasoningParser(tokenizer)
        self._tool_parser = MinimaxM2SanitizedToolParser(tokenizer)
        logger.debug("Successfully initialized parser %s!", self.__class__.__name__)
