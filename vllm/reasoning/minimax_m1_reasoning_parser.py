# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Optional, Union

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser, ReasoningParserManager

logger = init_logger(__name__)


@ReasoningParserManager.register_module("minimax_m1")
class MiniMaxM1ReasoningParser(ReasoningParser):
    """
    Reasoning parser for MiniMax M1 model.

    The MiniMax M1 model uses <think>...</think> tokens to denote reasoning
    text. This parser extracts the reasoning content from the model output.
    think start: "<think>": [60, 37959, 62]
    think ends: "</think>": [1579, 37959, 62]
    """

    start_token: str = "<think>\n"
    end_token: str = "\n</think>\n"

    current_state: str
    matching_pos: int

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction.")
        
        self.current_state = "idle"
        self.matching_pos = 0

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return self.current_state == "response"

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract the content after the end tokens
        """
        # TODO
        return [] 

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:

        content = ""
        reasoning_content = ""
            
        for c in delta_text:
            if self.current_state == "idle":
                if c == self.start_token[self.matching_pos]:
                    self.matching_pos += 1
                    if self.matching_pos == len(self.start_token):
                        self.current_state = "reasoning"
                        self.matching_pos = 0
                else:
                    self.current_state = "response"
                    content += self.start_token[:self.matching_pos] + c
                    self.matching_pos = 0
            elif self.current_state == "reasoning":
                if c == self.end_token[self.matching_pos]:
                    self.matching_pos += 1
                    if self.matching_pos == len(self.end_token):
                        self.current_state = "response"
                        self.matching_pos = 0
                else:
                    if c == self.end_token[0]:
                        self.matching_pos = 1
                    else:
                        if self.matching_pos > 0:
                            reasoning_content += self.end_token[:self.matching_pos]
                        reasoning_content += c
                        self.matching_pos = 0
            else:
                content += c
        
        if len(content) > 0:
            return DeltaMessage(content=content)
        elif len(reasoning_content) > 0:
            return DeltaMessage(reasoning_content=reasoning_content)
        else:
            return None

    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content from the model output.

        For text <think>abc</think>xyz:
        - 'abc' goes to reasoning_content
        - 'xyz' goes to content

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        reasoning_content = None
        content = None

        if model_output.startswith(self.start_token):
            end_idx = model_output.find(self.end_token)
            if end_idx != -1:
                reasoning_content = model_output[len(self.start_token):end_idx]
                content = model_output[end_idx + len(self.end_token):]
                if len(content) == 0:
                    content = None
            else:
                content = model_output

        else:
            content = model_output

        return reasoning_content, content
