# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Optional, Union

import regex as re
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser, ReasoningParserManager

logger = init_logger(__name__)


@ReasoningParserManager.register_module("step3")
class Step3ReasoningParser(ReasoningParser):
    """
    Reasoning parser for Step3 model.

    The Step3 model uses </think> token to denote the end of reasoning 
    text. This parser extracts all content before </think> as reasoning content.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        self.think_end_token = "</think>"

        self.reasoning_regex = re.compile(rf"(.*?){self.think_end_token}",
                                          re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction.")

        self.think_end_token_id = self.vocab.get(self.think_end_token)
        if self.think_end_token_id is None:
            raise RuntimeError(
                "Step3 reasoning parser could not locate think end "
                "token in the tokenizer!")

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
        For text "abc</think>xyz":
        - 'abc' goes to reasoning_content
        - 'xyz' goes to content
        """
        # Skip single special token
        if len(delta_token_ids
               ) == 1 and delta_token_ids[0] == self.think_end_token_id:
            return None

        if self.think_end_token_id in delta_token_ids:
            # </think> in delta, extract reasoning content and remaining content
            end_index = delta_text.find(self.think_end_token)
            reasoning_content = delta_text[:end_index]
            content = delta_text[end_index + len(self.think_end_token):]
            return DeltaMessage(reasoning_content=reasoning_content,
                                content=content if content else None)
        elif self.think_end_token_id in previous_token_ids:
            # </think> already seen in previous text, everything is content
            return DeltaMessage(content=delta_text)
        else:
            # No </think> seen yet, everything is reasoning
            return DeltaMessage(reasoning_content=delta_text)

    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:

        # Check if the model output contains the </think> token
        if self.think_end_token not in model_output:
            # If no </think> token, everything is reasoning content
            return model_output, None
        else:
            # Find the first occurrence of </think>
            end_index = model_output.find(self.think_end_token)
            reasoning_content = model_output[:end_index]

            # Content after </think> token
            content = model_output[end_index + len(self.think_end_token):]

            if len(content) == 0:
                content = None

            return reasoning_content, content

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return self.think_end_token_id in input_ids

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self.think_end_token_id not in input_ids[:-1]:
            return []
        else:
            return input_ids[input_ids.index(self.think_end_token_id) + 1:]
