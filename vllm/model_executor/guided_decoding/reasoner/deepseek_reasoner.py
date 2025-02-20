# SPDX-License-Identifier: Apache-2.0
from transformers import PreTrainedTokenizer

from vllm.model_executor.guided_decoding.reasoner.reasoner import Reasoner


class DeepSeekReasoner(Reasoner):
    _instance = None
    _start_token_id = None
    _end_token_id = None

    def __new__(cls, tokenizer: PreTrainedTokenizer):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

        # Initialize token IDs only once
        if self.__class__._start_token_id is None:
            self.__class__._start_token_id = tokenizer.encode(
                "<think>", add_special_tokens=False)[0]
            self.__class__._end_token_id = tokenizer.encode(
                "</think>", add_special_tokens=False)[0]

        # Use class variables
        self.start_token_id = self.__class__._start_token_id
        self.end_token_id = self.__class__._end_token_id

    def get_start_token_id(self) -> int:
        return self.start_token_id

    def get_end_token_id(self) -> int:
        return self.end_token_id
