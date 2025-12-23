# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser

logger = init_logger(__name__)


class Glm4MoeModelReasoningParser(ReasoningParser):
    """
    Reasoning parser for the Glm4MoeModel model.

    The Glm4MoeModel model uses <think>...</think> tokens to denote reasoning
    text within its output. The model provides a strict switch to disable
    reasoning output via the 'enable_thinking=False' parameter. This parser
    extracts the reasoning content enclosed by <think> and </think> tokens
    from the model's output.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"
        self.assistant_token = "<|assistant|>"

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

        self.think_start_token_id = self.vocab.get(self.think_start_token)
        self.think_end_token_id = self.vocab.get(self.think_end_token)
        self.assistant_token_id = self.vocab.get(self.assistant_token)
        if (
            self.think_start_token_id is None
            or self.think_end_token_id is None
            or self.assistant_token_id is None
        ):
            raise RuntimeError(
                "Glm4MoeModel reasoning parser could not locate "
                "think start/end or assistant tokens in the tokenizer!"
            )

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        """
        GLM's chat template has <think></think> tokens after every
        <|assistant|> token. Thus, we need to check if </think> is
        after the most recent <|assistant|> token (if present).
        """
        for token_id in input_ids[::-1]:
            if token_id == self.think_end_token_id:
                return True
            elif token_id == self.assistant_token_id:
                return False
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract the content after the end tokens
        """
        if self.think_end_token_id not in input_ids[:-1]:
            return []
        else:
            return input_ids[input_ids.index(self.think_end_token_id) + 1 :]

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
        Extract reasoning content from a delta message.
        Handles streaming output where previous + delta = current.
        Uses token IDs for faster processing.
        For text <think>abc</think>xyz:
        - 'abc' goes to reasoning
        - 'xyz' goes to content

        Note: When <think> is added by chat template (not generated),
        we assume reasoning mode until </think> is seen.
        """
        # Skip single special tokens
        if len(delta_token_ids) == 1 and (
            delta_token_ids[0] in [self.think_start_token_id, self.think_end_token_id]
        ):
            return None

        # Check if we've already seen </think> in previous tokens
        # If so, we're in content mode
        if self.think_end_token_id in previous_token_ids:
            return DeltaMessage(content=delta_text)

        # Check if </think> is in the current delta
        if self.think_end_token_id in delta_token_ids:
            # Transition from reasoning to content
            end_index = delta_text.find(self.think_end_token)
            reasoning = delta_text[:end_index]
            content = delta_text[end_index + len(self.think_end_token) :]
            return DeltaMessage(
                reasoning=reasoning if reasoning else None,
                content=content if content else None,
            )

        # Check if <think> is in previous or current delta
        if self.think_start_token_id in previous_token_ids:
            # <think> seen, still in reasoning mode
            return DeltaMessage(reasoning=delta_text)
        elif self.think_start_token_id in delta_token_ids:
            # <think> in delta, extract content after it
            start_index = delta_text.find(self.think_start_token)
            reasoning = delta_text[start_index + len(self.think_start_token):]
            return DeltaMessage(reasoning=reasoning if reasoning else None)

        # No <think> or </think> seen yet
        # Assume reasoning mode if chat template added <think> to prompt
        # (content before </think> is reasoning)
        # Check if </think> will appear later by looking at current_text
        if self.think_end_token in current_text or self.think_end_token not in previous_text:
            # We're likely still before </think>, treat as reasoning
            return DeltaMessage(reasoning=delta_text)
        else:
            # No thinking tokens at all, just content
            return DeltaMessage(content=delta_text)

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.

        For text <think>abc</think>xyz:
        - 'abc' goes to reasoning
        - 'xyz' goes to content

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        # Check if </think> is present - if not, no reasoning to extract
        if self.think_end_token not in model_output:
            return None, model_output

        # Check if <think> is present in model output
        if self.think_start_token in model_output:
            # Both tokens present - extract between them
            model_output_parts = model_output.partition(self.think_start_token)
            model_output = (
                model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
            )

        # Extract reasoning content (everything before </think>)
        # and content (everything after </think>)
        reasoning, _, content = model_output.partition(self.think_end_token)

        final_content = content or None
        return reasoning, final_content
