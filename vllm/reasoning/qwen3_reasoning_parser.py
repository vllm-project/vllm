# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser


class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for the Qwen3 model.

    The Qwen3 model uses <think>...</think> tokens to denote reasoning text
    within its output.

    Newer Qwen3 thinking models (e.g., Qwen3-4B-Thinking-2507) support only
    thinking mode, meaning ``enable_thinking=True`` is always in effect. Their
    default chat template automatically appends ``<think>`` to the assistant
    turn prefix inside the prompt. As a result, the model's generated output
    typically starts directly with the reasoning text and ends with
    ``</think>content`` — the opening ``<think>`` tag does NOT appear in the
    generated tokens.

    This parser handles both cases:
      1. Full tags in output: ``<think>reasoning</think>content``
      2. Only closing tag in output (opening tag in prompt):
         ``reasoning</think>content``
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

        Handles two cases:
        1. Both <think> and </think> in output: text between them is reasoning.
        2. Only </think> in output (when <think> was added by the chat
           template as part of the prompt): text before </think> is reasoning.

        If neither token is present, returns (None, model_output).

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        # If no </think> in output, there is no reasoning to extract.
        if self.end_token not in model_output:
            return None, model_output

        # Remove <think> start token if present in the model output.
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        # Extract reasoning content from the model output.
        reasoning, _, content = model_output.partition(self.end_token)

        final_content = content or None
        return reasoning, final_content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Extract reasoning content from a streaming delta.

        Extends the base implementation to handle the case where the <think>
        token was added by the chat template as part of the prompt and is not
        present in the generated token IDs. In this case, all content before
        </think> is treated as reasoning content.
        """
        ret = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        # If <think> was not found in either previous or delta token IDs,
        # it means <think> was part of the prompt (added by the chat
        # template). We need to treat content before </think> as reasoning.
        if (
            ret is not None
            and self.start_token_id not in previous_token_ids
            and self.start_token_id not in delta_token_ids
        ):
            if self.end_token_id in delta_token_ids:
                # </think> found in delta: split into reasoning and content
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning,
                    content=content if content else None,
                )
            elif self.end_token_id in previous_token_ids:
                # </think> was in a previous chunk: reasoning is done,
                # this is content
                return DeltaMessage(content=delta_text)
            else:
                # No </think> yet: reasoning content continues
                return DeltaMessage(reasoning=delta_text)

        return ret
