from typing import List, Optional

from vllm.config import TokenizerPoolConfig
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.tokenizer import (AnyTokenizer,
                                               get_lora_tokenizer,
                                               get_lora_tokenizer_async,
                                               get_tokenizer)
from vllm.utils import LRUCache

from .base_tokenizer_group import BaseTokenizerGroup


class TokenizerGroup(BaseTokenizerGroup):
    """A group of tokenizers that can be used for LoRA adapters."""

    def __init__(self, tokenizer_id: str, enable_lora: bool, max_num_seqs: int,
                 max_input_length: Optional[int], **tokenizer_config):
        self.tokenizer_id = tokenizer_id
        self.tokenizer_config = tokenizer_config
        self.enable_lora = enable_lora
        self.max_input_length = max_input_length
        self.tokenizer = get_tokenizer(self.tokenizer_id, **tokenizer_config)
        self.lora_tokenizers = LRUCache[AnyTokenizer](
            capacity=max_num_seqs if enable_lora else 0)

    @classmethod
    def from_config(cls, tokenizer_pool_config: Optional[TokenizerPoolConfig],
                    **init_kwargs) -> "TokenizerGroup":
        return cls(**init_kwargs)

    def ping(self) -> bool:
        """Check if the tokenizer group is alive."""
        return True

    def get_max_input_len(self,
                          lora_request: Optional[LoRARequest] = None
                          ) -> Optional[int]:
        """Get the maximum input length for the LoRA request."""
        return self.max_input_length

    def _raise_if_input_too_long(self,
                                 encoded_tokens: List[int],
                                 lora_request: Optional[LoRARequest] = None):
        input_length = len(encoded_tokens)
        if lora_request:
            max_input_length = (lora_request.long_lora_max_len
                                or self.max_input_length)
        else:
            max_input_length = self.max_input_length
        if max_input_length is not None and input_length > max_input_length:
            raise ValueError("Input too long.", input_length, max_input_length)

    def encode(self,
               prompt: str,
               request_id: Optional[str] = None,
               lora_request: Optional[LoRARequest] = None) -> List[int]:
        tokenizer = self.get_lora_tokenizer(lora_request)
        ret = tokenizer.encode(prompt)
        self._raise_if_input_too_long(ret, lora_request)
        return ret

    async def encode_async(
            self,
            prompt: str,
            request_id: Optional[str] = None,
            lora_request: Optional[LoRARequest] = None) -> List[int]:
        tokenizer = await self.get_lora_tokenizer_async(lora_request)
        ret = tokenizer.encode(prompt)
        self._raise_if_input_too_long(ret, lora_request)
        return ret

    def get_lora_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        if not lora_request or not self.enable_lora:
            return self.tokenizer
        if lora_request.lora_int_id not in self.lora_tokenizers:
            tokenizer = (get_lora_tokenizer(
                lora_request, **self.tokenizer_config) or self.tokenizer)
            self.lora_tokenizers.put(lora_request.lora_int_id, tokenizer)
            return tokenizer
        else:
            return self.lora_tokenizers[lora_request.lora_int_id]

    async def get_lora_tokenizer_async(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        if not lora_request or not self.enable_lora:
            return self.tokenizer
        if lora_request.lora_int_id not in self.lora_tokenizers:
            tokenizer = (await get_lora_tokenizer_async(
                lora_request, **self.tokenizer_config) or self.tokenizer)
            self.lora_tokenizers.put(lora_request.lora_int_id, tokenizer)
            return tokenizer
        else:
            return self.lora_tokenizers[lora_request.lora_int_id]
