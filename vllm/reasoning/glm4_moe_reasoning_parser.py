# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class Glm4MoeReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for GLM-4 MoE models.

    Unlike DeepSeek R1, GLM-4 injects <think> via the chat template rather
    than generating it.  When the model output lacks </think>, the entire
    output is treated as *content* (not reasoning), because the absence of
    the end tag means the model chose not to reason.
    """

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        if self.end_token not in model_output:
            # No closing tag — model didn't produce reasoning.
            # Return the full original output as content.
            return None, model_output

        # Normal case: <think>reasoning</think>content
        parts = model_output.partition(self.start_token)
        after_start = parts[2] if parts[1] else parts[0]
        reasoning, _, content = after_start.partition(self.end_token)
        return reasoning or None, content or None
