# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ResponsesRequest
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

    def extract_reasoning_content(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.

        Qwen3 has stricter requirements - it needs both start and end tokens
        to be present, unlike other models that work with just the end token.

        For text <think>abc</think>xyz:
        - 'abc' goes to reasoning_content
        - 'xyz' goes to content

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        # Check if the model output contains both <think> and </think> tokens.
        if self.start_token not in model_output or self.end_token not in model_output:
            return None, model_output

        # Check if the <think> is present in the model output, remove it
        # if it is present.
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        # Check if the model output contains the </think> tokens.
        # If the end token is not found, return the model output as is.
        if self.end_token not in model_output:
            return None, model_output

        # Extract reasoning content from the model output.
        reasoning_content, _, content = model_output.partition(self.end_token)

        final_content = content or None
        return reasoning_content, final_content
