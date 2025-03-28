# SPDX-License-Identifier: Apache-2.0

import re
from collections.abc import Sequence
from typing import Optional, Union

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser, ReasoningParserManager

logger = init_logger(__name__)


@ReasoningParserManager.register_module("deepseek_r1")
class DeepSeekR1ReasoningParser(ReasoningParser):
    """
    Reasoning parser for DeepSeek R1 model.

    The DeepSeek R1 model uses <think>...</think> tokens to denote reasoning
    text. This parser extracts the reasoning content from the model output.
    """

    start_token_id: int
    end_token_id: int

    start_token: str = "<think>"
    end_token: str = "</think>"

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)

        self.reasoning_regex = re.compile(
            rf"{self.start_token}(.*?){self.end_token}", re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction.")

        self.start_token_id = self.vocab.get(self.start_token)
        self.end_token_id = self.vocab.get(self.end_token)
        if self.start_token_id is None or self.end_token_id is None:
            raise RuntimeError(
                "DeepSeek R1 reasoning parser could not locate think start/end "
                "tokens in the tokenizer!")

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return self.end_token_id in input_ids

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract the content after the end tokens
        """
        if self.end_token_id not in input_ids[:-1]:
            return []
        else:
            return input_ids[input_ids.index(self.end_token_id) + 1:]

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        """
        Extract reasoning content from a delta message.
        Handles streaming output where previous + delta = current.
        Uses token IDs for faster processing.
        For text <think>abc</think>xyz:
        - 'abc' goes to reasoning_content
        - 'xyz' goes to content
        """
        # Skip single special tokens
        if len(delta_token_ids) == 1 and (delta_token_ids[0] in [
                self.start_token_id, self.end_token_id
        ]):
            return None

        # Check if <think> is present in previous or delta.
        # Keep compatibility with models that don't generate <think> tokens.
        if self.start_token_id in previous_token_ids:
            if self.end_token_id in delta_token_ids:
                # <think> in previous, </think> in delta,
                # extract reasoning content
                end_index = delta_text.find(self.end_token)
                reasoning_content = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token):]
                return DeltaMessage(
                    reasoning_content=reasoning_content,
                    content=content if content else None,
                )
            elif self.end_token_id in previous_token_ids:
                # <think> in previous, </think> in previous,
                # reasoning content continues
                return DeltaMessage(content=delta_text)
            else:
                # <think> in previous, no </think> in previous or delta,
                # reasoning content continues
                return DeltaMessage(reasoning_content=delta_text)
        elif self.start_token_id in delta_token_ids:
            if self.end_token_id in delta_token_ids:
                # <think> in delta, </think> in delta, extract reasoning content
                start_index = delta_text.find(self.start_token)
                end_index = delta_text.find(self.end_token)
                reasoning_content = delta_text[start_index +
                                               len(self.start_token):end_index]
                content = delta_text[end_index + len(self.end_token):]
                return DeltaMessage(
                    reasoning_content=reasoning_content,
                    content=content if content else None,
                )
            else:
                # <think> in delta, no </think> in delta,
                # reasoning content continues
                return DeltaMessage(reasoning_content=delta_text)
        else:
            # No <think> in previous or delta, also need to check for </think>.
            # Because the model may have generated </think> without <think>
            # Ref https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f
            if self.end_token_id in delta_token_ids:
                # </think> in delta with more tokens,
                # extract reasoning content and content
                end_index = delta_text.find(self.end_token)
                reasoning_content = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token):]
                return DeltaMessage(
                    reasoning_content=reasoning_content,
                    content=content if content else None,
                )
            elif self.end_token_id in previous_token_ids:
                # </think> in previous, thinking content ends
                return DeltaMessage(content=delta_text)
            else:
                # no </think> in previous or delta, reasoning content continues
                return DeltaMessage(reasoning_content=delta_text)

    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        # DeepSeek R1 doesn't generate <think> now.
        # Thus we assume the reasoning content is always at the start.
        # Ref https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f
        if self.end_token not in model_output:
            return model_output, None
        else:
            # Add a start token if it's missing to keep compatibility.
            if self.start_token not in model_output:
                model_output = f"{self.start_token}{model_output}"
            # Use a regex to find the reasoning content
            reasoning_content = self.reasoning_regex.findall(model_output)[0]

            end_index = len(
                f"{self.start_token}{reasoning_content}{self.end_token}")
            final_output = model_output[end_index:]

            if len(final_output) == 0:
                return reasoning_content, None

            return reasoning_content, final_output
