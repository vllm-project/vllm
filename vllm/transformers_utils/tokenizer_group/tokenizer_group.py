from typing import List, Optional

from transformers import PreTrainedTokenizer

from vllm.lora.request import LoRARequest
from vllm.transformers_utils.tokenizer import (get_lora_tokenizer,
                                               get_lora_tokenizer_async)
from vllm.transformers_utils.tokenizer_group.base_tokenizer_group import (
    BaseTokenizerGroup)
from vllm.utils import LRUCache
from vllm.transformers_utils.tokenizer import get_tokenizer


class TokenizerGroup(BaseTokenizerGroup):
    """A group of tokenizers that can be used for LoRA adapters."""

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

    def ping(self) -> bool:
        """Check if the tokenizer group is alive."""
        return True

    def get_max_input_len(self,
                          lora_request: Optional[LoRARequest] = None
                          ) -> Optional[int]:
        """Get the maximum input length for the LoRA request."""
        return self.max_input_length

    def encode(self,
               prompt: str,
               request_id: Optional[str] = None,
               lora_request: Optional[LoRARequest] = None) -> List[int]:
        tokenizer = self.get_lora_tokenizer(lora_request)
        return tokenizer.encode(prompt)

    async def encode_async(
            self,
            prompt: str,
            request_id: Optional[str] = None,
            lora_request: Optional[LoRARequest] = None) -> List[int]:
        tokenizer = await self.get_lora_tokenizer_async(lora_request)
        return tokenizer.encode(prompt)

    def get_lora_tokenizer(
            self,
            lora_request: Optional[LoRARequest] = None
    ) -> "PreTrainedTokenizer":
        if not lora_request or not self.enable_lora:
            return self.tokenizer
        if lora_request.lora_int_id not in self.lora_tokenizers:
            tokenizer = (get_lora_tokenizer(
                lora_request, **self.tokenizer_config) or self.tokenizer)
            self.lora_tokenizers.put(lora_request.lora_int_id, tokenizer)
            return tokenizer
        else:
            return self.lora_tokenizers.get(lora_request.lora_int_id)

    async def get_lora_tokenizer_async(
            self,
            lora_request: Optional[LoRARequest] = None
    ) -> "PreTrainedTokenizer":
        if not lora_request or not self.enable_lora:
            return self.tokenizer
        if lora_request.lora_int_id not in self.lora_tokenizers:
            tokenizer = (await get_lora_tokenizer_async(
                lora_request, **self.tokenizer_config) or self.tokenizer)
            self.lora_tokenizers.put(lora_request.lora_int_id, tokenizer)
            return tokenizer
        else:
            return self.lora_tokenizers.get(lora_request.lora_int_id)
