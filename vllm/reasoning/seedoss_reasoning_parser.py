# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from collections.abc import Sequence

from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser


class SeedOSSReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for SeedOSS model.

    The SeedOSS model uses <seed:think>...</seed:think> tokens to
    denote reasoning content text. This parser extracts
    the reasoning content from the model output.
    Similar to DeepSeek R1, it supports cases
    where the model doesn't generate the start token.
    """

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<seed:think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</seed:think>"

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == self.start_token_id:
                return False
            if input_ids[i] == self.end_token_id:
                return True
        # No reasoning tokens found — thinking was never started,
        # treat as already ended so structured output is applied.
        return True
