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


class Glm47ReasoningParser(Glm4MoeModelReasoningParser):
    """
    Reasoning parser for GLM-4.7 models.

    GLM-4.7 chat templates include the <think> token at the end of the prompt
    (after <|assistant|>), so the model output starts directly with thinking
    content but ends with </think>. This parser handles this case by prepending
    <think> when needed.

    Use this parser with: --reasoning-parser glm47
    """

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
        Handles the case where <think> is injected by the template.
        """
        # Skip single start token to avoid issues when templates inject it
        if (
            len(delta_token_ids) == 1
            and delta_token_ids[0] == self.think_start_token_id
        ):
            return None

        # At the very start, if we don't see <think>, treat output as reasoning
        # since GLM-4.7 templates inject <think> in the prompt
        if (
            len(previous_token_ids) == 0
            and self.think_start_token_id not in delta_token_ids
        ):
            # No <think> in first delta, this is reasoning content
            if self.think_end_token_id in delta_token_ids:
                # </think> in first delta - extract and split
                end_index = delta_text.find(self.think_end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.think_end_token) :]
                return DeltaMessage(
                    reasoning=reasoning,
                    content=content if content else None,
                )
            return DeltaMessage(reasoning=delta_text)

        # For subsequent tokens, use parent behavior but with adjustment
        # If we've been streaming reasoning (no <think> seen but also no
        # </think> yet), continue as reasoning
        if (
            self.think_start_token_id not in previous_token_ids
            and self.think_end_token_id not in previous_token_ids
        ):
            # We're in "implicit reasoning mode" (template injected <think>)
            if self.think_end_token_id in delta_token_ids:
                end_index = delta_text.find(self.think_end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.think_end_token) :]
                return DeltaMessage(
                    reasoning=reasoning,
                    content=content if content else None,
                )
            return DeltaMessage(reasoning=delta_text)

        # Fall back to parent implementation for other cases
        return super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.
        Handles the case where <think> is in the template but not in output.
        """
        # If we have </think> but no <think>, prepend <think>
        # This handles GLM-4.7 templates that inject <think> in the prompt
        if (
            self.think_end_token in model_output
            and self.think_start_token not in model_output
        ):
            model_output = self.think_start_token + model_output

        # Now use parent implementation
        return super().extract_reasoning(model_output, request)
