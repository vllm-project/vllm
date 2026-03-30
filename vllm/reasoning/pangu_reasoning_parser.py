# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import DeltaMessage
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser


class PanguReasoningParser(DeepSeekR1ReasoningParser):
    """
    Reasoning parser for Pangu model.

    The Pangu model uses <think>...</think> tokens to denote reasoning
    text. This parser extracts the reasoning content from the model output.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.delta_token_ids = []

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>" if self.vocab.get("<think>") else "[unused16]"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>" if self.vocab.get("</think>") else "[unused17]"

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return (
            self.end_token_id in input_ids and self.end_token_id in self.delta_token_ids
        )

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        self.delta_token_ids = delta_token_ids

        ret = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

        if (
            ret is not None
            and self.start_token_id in delta_token_ids
            and self.end_token_id not in delta_token_ids
        ):
            # start token in delta with extra tokens
            delta_text = delta_text.split(self.start_token)[-1]
            return DeltaMessage(reasoning=delta_text)

        return ret
