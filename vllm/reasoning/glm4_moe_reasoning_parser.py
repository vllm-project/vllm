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
        """
        # Skip single special tokens
        if len(delta_token_ids) == 1 and (
            delta_token_ids[0] in [self.think_start_token_id, self.think_end_token_id]
        ):
            return None

        if self.think_start_token_id in previous_token_ids:
            if self.think_end_token_id in delta_token_ids:
                # <think> in previous, </think> in delta,
                # extract reasoning content
                end_index = delta_text.find(self.think_end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.think_end_token) :]
                return DeltaMessage(
                    reasoning=reasoning,
                    content=content if content else None,
                )
            elif self.think_end_token_id in previous_token_ids:
                # <think> in previous, </think> in previous,
                # reasoning content continues
                return DeltaMessage(content=delta_text)
            else:
                # <think> in previous, no </think> in previous or delta,
                # reasoning content continues
                return DeltaMessage(reasoning=delta_text)
        elif self.think_start_token_id in delta_token_ids:
            if self.think_end_token_id in delta_token_ids:
                # <think> in delta, </think> in delta, extract reasoning content
                start_index = delta_text.find(self.think_start_token)
                end_index = delta_text.find(self.think_end_token)
                reasoning = delta_text[
                    start_index + len(self.think_start_token) : end_index
                ]
                content = delta_text[end_index + len(self.think_end_token) :]
                return DeltaMessage(
                    reasoning=reasoning,
                    content=content if content else None,
                )
            else:
                # <think> in delta, no </think> in delta,
                # reasoning content continues
                return DeltaMessage(reasoning=delta_text)
        else:
            # thinking is disabled, just content
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

        # Check if the model output contains the <think> and </think> tokens.
        if (
            self.think_start_token not in model_output
            or self.think_end_token not in model_output
        ):
            return None, model_output
        # Check if the <think> is present in the model output, remove it
        # if it is present.
        model_output_parts = model_output.partition(self.think_start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )
        # Check if the model output contains the </think> tokens.
        # If the end token is not found, return the model output as is.
        if self.think_end_token not in model_output:
            return None, model_output

        # Extract reasoning content from the model output.
        reasoning, _, content = model_output.partition(self.think_end_token)

        final_content = content or None
        return reasoning, final_content
