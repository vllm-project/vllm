# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser


class SeedOSSReasoningParser(DeepSeekR1ReasoningParser):
    """
    Reasoning parser for SeedOSS model.

    The SeedOSS model uses <seed:think>...</seed:think> tokens to denote
    reasoning content text.

    Like DeepSeek R1, SeedOSS may omit the start token (the chat template can
    open the reasoning block), so it reuses DeepSeek R1's streaming logic,
    which treats leading text as reasoning. The base parser would emit that
    text as content, disagreeing with non-streaming extraction.
    """

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<seed:think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</seed:think>"
