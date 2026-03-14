# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
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

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        # Like R1, SeedOSS may not emit the start token (it's in the chat
        # template).  When neither previous nor delta contains the start
        # token, treat text as reasoning unless the end token has been seen.
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
            and self.start_token_id not in previous_token_ids
            and self.start_token_id not in delta_token_ids
        ):
            if self.end_token_id in delta_token_ids:
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning,
                    content=content if content else None,
                )
            elif self.end_token_id in previous_token_ids:
                return DeltaMessage(content=delta_text)
            else:
                return DeltaMessage(reasoning=delta_text)

        return ret
