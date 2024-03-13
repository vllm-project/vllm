from abc import ABC, abstractmethod
from typing import List, Optional

from transformers import PreTrainedTokenizer

from vllm.lora.request import LoRARequest
from vllm.utils import LRUCache
from vllm.transformers_utils.tokenizer import get_tokenizer


class BaseTokenizerGroup(ABC):

    def __init__(self, tokenizer_id: str, enable_lora: bool, max_num_seqs: int,
                 max_input_length: Optional[int], **tokenizer_config):
        self.tokenizer_id = tokenizer_id
        self.tokenizer_config = tokenizer_config
        self.enable_lora = enable_lora
        self.max_input_length = max_input_length
        self.tokenizer = get_tokenizer(self.tokenizer_id, **tokenizer_config)
        if enable_lora:
            self.lora_tokenizers = LRUCache(capacity=max_num_seqs)
        else:
            self.lora_tokenizers = None

    def get_max_input_len(self,
                          lora_request: Optional[LoRARequest] = None
                          ) -> Optional[int]:
        """Get the maximum input length for the LoRA request."""
        return self.max_input_length

    def ping(self):
        """Check if the tokenizer group is alive."""
        return True

    @abstractmethod
    def encode(self, prompt: str, request_id: Optional[str],
               lora_request: Optional[LoRARequest]) -> List[int]:
        """Encode a prompt using the tokenizer group."""
        pass

    async def encode_async(self, prompt: str, request_id: Optional[str],
                           lora_request: Optional[LoRARequest]) -> List[int]:
        """Encode a prompt using the tokenizer group."""
        return self.encode(prompt=prompt,
                           request_id=request_id,
                           lora_request=lora_request)

    @abstractmethod
    def get_lora_tokenizer(
            self,
            lora_request: Optional[LoRARequest]) -> "PreTrainedTokenizer":
        ...
        """Get a tokenizer for a LoRA request."""

    async def get_lora_tokenizer_async(
            self,
            lora_request: Optional[LoRARequest]) -> "PreTrainedTokenizer":
        """Get a tokenizer for a LoRA request."""
        return self.get_lora_tokenizer(lora_request)
