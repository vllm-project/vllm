# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser


class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for the Qwen3 model.

    The Qwen3 model uses <think>...</think> tokens to denote reasoning text
    within its output. The model provides a strict switch to disable reasoning
    output via the 'enable_thinking=False' parameter. This parser extracts the
    reasoning content enclosed by <think> and </think> tokens from the model's
    output.
    """

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[list[str], str | None]:
        """
        Extract reasoning content from the model output.

        Qwen3 has stricter requirements - it needs both start and end tokens
        to be present, unlike other models that work with just the end token.

        For text <think>abc</think>xyz:
        - 'abc' goes to reasoning
        - 'xyz' goes to content

        The extraction is recursive - after finding the first reasoning block,
        it continues to look for additional reasoning blocks in the remaining
        content.

        Returns:
            tuple[list[str], Optional[str]]: list of reasoning content strings
            and final content
        """
        reasoning_list: list[str] = []
        remaining = model_output

        while True:
            # Check if both start and end tokens are present
            if self.start_token not in remaining or self.end_token not in remaining:
                # No complete reasoning block - rest is content
                final_content = remaining if remaining else None
                return reasoning_list, final_content

            # Extract reasoning block
            parts = remaining.partition(self.start_token)
            after_start = parts[2]

            # Get the reasoning content
            reasoning, _, after_end = after_start.partition(self.end_token)
            if reasoning:
                reasoning_list.append(reasoning)

            # Check if there's more content to process
            if not after_end:
                return reasoning_list, None

            # Check for more reasoning blocks
            if self.start_token in after_end and self.end_token in after_end:
                remaining = after_end
                continue
            else:
                # No more complete reasoning blocks
                return reasoning_list, after_end if after_end else None
