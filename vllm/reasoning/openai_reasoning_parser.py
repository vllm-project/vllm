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


@ReasoningParserManager.register_module("openai")
class OpenAIReasoningParser(ReasoningParser):
    """
    Reasoning parser for OpenAI model.

    The OpenAI model uses harmony to extract reasoning content and this parser
    is only used for detecting the end of the reasoning content.
    """

    # <|start|>assistant<|channel|>final<|message|>
    # token_ids: list[int] = [200006, 173781, 200005, 17196, 200008]

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        self.reasoning_end_token_ids = self.model_tokenizer.encode(
            "<|start|>assistant<|channel|>final<|message|>")
        print("reasoning_end_token_ids", self.reasoning_end_token_ids)

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        def find_last_index(lst, value):
            return len(lst) - 1 - list(reversed(lst)).index(value)
        last_start = find_last_index(input_ids, self.reasoning_end_token_ids[0])
        if last_start is None:
            return False
        for i in range(5):
            if last_start + i >= len(input_ids):
                return False
            if input_ids[last_start + i] != self.reasoning_end_token_ids[i]:
                return False
        return True

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        raise RuntimeError(
            "OpenAI model uses harmony to extract reasoning content. This "
            "function should not be called.")

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        raise RuntimeError(
            "OpenAI model uses harmony to extract reasoning content. This "
            "function should not be called.")

    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        raise RuntimeError(
            "OpenAI model uses harmony to extract reasoning content. This "
            "function should not be called.")