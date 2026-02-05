# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 
# PATCHED: Fix for Qwen3-Thinking models where <think> is added as prompt prefix
# The model output only contains </think>, not the opening <think> tag.
# Original parser required both tags, this fix handles prefix format.

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
    
    Note: Qwen3-Thinking models add <think> as a prompt prefix, so the model
    output may only contain </think> without the opening tag. This parser
    handles both cases.
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
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.

        Handles two formats:
        1. Full format: <think>reasoning</think>content
        2. Prefix format: reasoning</think>content (when <think> is in prompt)

        For text <think>abc</think>xyz or abc</think>xyz:
        - 'abc' goes to reasoning
        - 'xyz' goes to content

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """
        # If no end token, no reasoning to extract
        if self.end_token not in model_output:
            return None, model_output

        # Remove start token if present (handles full format)
        if self.start_token in model_output:
            model_output_parts = model_output.partition(self.start_token)
            model_output = (
                model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
            )

        # Extract reasoning content (everything before </think>)
        reasoning, _, content = model_output.partition(self.end_token)
        
        # Strip leading/trailing whitespace from reasoning
        reasoning = reasoning.strip() if reasoning else None
        
        # Content may have leading newlines from template format
        final_content = content.lstrip('\n') if content else None
        
        return reasoning, final_content
