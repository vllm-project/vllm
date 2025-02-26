# SPDX-License-Identifier: Apache-2.0
from threading import Lock

from transformers import PreTrainedTokenizer

from vllm.model_executor.guided_decoding.reasoner.reasoner import Reasoner


class DeepSeekReasoner(Reasoner):
    """
    Reasoner for DeepSeek.

    This class is a singleton and should be instantiated with the tokenizer
    to ensure that the start and end token IDs are initialized only once.
    """
    _instance = None
    _start_token_id = None
    _end_token_id = None
    _lock = Lock()

    def __new__(cls, tokenizer: PreTrainedTokenizer):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                # Initialize token IDs in __new__
                cls._start_token_id = tokenizer.encode(
                    "<think>", add_special_tokens=False)[0]
                cls._end_token_id = tokenizer.encode(
                    "</think>", add_special_tokens=False)[0]
        return cls._instance

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        # Use class variables to avoid reinitializing the token IDs
        self.start_token_id = self.__class__._start_token_id
        self.end_token_id = self.__class__._end_token_id

    def get_start_token_id(self) -> int:
        return self.start_token_id

    def get_end_token_id(self) -> int:
        return self.end_token_id
