# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.v1.token_buffer import rfind_tokens


class DeepSeekR1ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for DeepSeek R1 model.

    The DeepSeek R1 model uses <think>...</think> tokens to denote reasoning
    text. This parser extracts the reasoning content from the model output.
    """

    _prompt_opens_reasoning = False

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def set_prompt_token_ids(self, prompt_token_ids: Sequence[int]) -> None:
        last_marker = rfind_tokens(
            prompt_token_ids, (self.start_token_id, self.end_token_id)
        )
        self._prompt_opens_reasoning = (
            last_marker >= 0 and prompt_token_ids[last_marker] == self.start_token_id
        )

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        if not self._prompt_opens_reasoning:
            return super().count_reasoning_tokens(token_ids)

        # Count tokens before the first generated reasoning tag.
        first_marker = len(token_ids)
        for marker_id in (self.end_token_id, self.start_token_id):
            try:
                first_marker = token_ids.index(marker_id, 0, first_marker)
            except ValueError:
                continue
        return first_marker + super().count_reasoning_tokens(token_ids[first_marker:])

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
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
                # end token in delta with more tokens,
                # extract reasoning content and content
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning,
                    content=content if content else None,
                )
            elif self.end_token_id in previous_token_ids:
                # end token in previous, thinking content ends
                return DeltaMessage(content=delta_text)
            else:
                # no end token in previous or delta, reasoning content continues
                return DeltaMessage(reasoning=delta_text)

        return ret
