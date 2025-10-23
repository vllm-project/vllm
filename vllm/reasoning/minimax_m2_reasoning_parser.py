# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser

@ReasoningParserManager.register_module("minimax_m2")
class MiniMaxM2ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for MiniMax M2 model.
    """

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"
