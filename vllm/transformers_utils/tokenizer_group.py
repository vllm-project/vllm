# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

from vllm.config import LoRAConfig, ModelConfig, SchedulerConfig
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.tokenizer import (AnyTokenizer, encode_tokens,
                                               get_lora_tokenizer,
                                               get_lora_tokenizer_async,
                                               get_tokenizer)
from vllm.utils import LRUCache


class TokenizerGroup:
    """A group of tokenizers that can be used for LoRA adapters."""

    def __init__(self, tokenizer_id: str, enable_lora: bool, max_num_seqs: int,
                 max_input_length: Optional[int], **tokenizer_config):
        self.tokenizer_id = tokenizer_id
        self.tokenizer_config = tokenizer_config
        self.enable_lora = enable_lora
        self.max_input_length = max_input_length
        self.tokenizer = get_tokenizer(self.tokenizer_id, **tokenizer_config)
        max_loras = tokenizer_config.get("max_loras", 0)
        self.lora_tokenizers = LRUCache[int, AnyTokenizer](
            capacity=max(max_loras, max_num_seqs) if enable_lora else 0)

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
               max_length: Optional[int] = None,
               truncation: Optional[bool] = None,
               lora_request: Optional[LoRARequest] = None,
               add_special_tokens: Optional[bool] = None) -> List[int]:

        tokenizer = self.get_lora_tokenizer(lora_request)
        ret = encode_tokens(tokenizer,
                            prompt,
                            max_length=max_length,
                            truncation=truncation,
                            add_special_tokens=add_special_tokens)
        self._raise_if_input_too_long(ret, lora_request)
        return ret

    async def encode_async(
            self,
            prompt: str,
            max_length: Optional[int] = None,
            truncation: Optional[bool] = None,
            lora_request: Optional[LoRARequest] = None,
            add_special_tokens: Optional[bool] = None) -> List[int]:
        tokenizer = await self.get_lora_tokenizer_async(lora_request)
        ret = encode_tokens(tokenizer,
                            prompt,
                            max_length=max_length,
                            truncation=truncation,
                            add_special_tokens=add_special_tokens)
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


def init_tokenizer_from_configs(model_config: ModelConfig,
                                scheduler_config: SchedulerConfig,
                                lora_config: Optional[LoRAConfig]):
    return TokenizerGroup(
        tokenizer_id=model_config.tokenizer,
        enable_lora=bool(lora_config),
        max_num_seqs=scheduler_config.max_num_seqs,
        max_loras=lora_config.max_loras if lora_config else 0,
        max_input_length=None,
        tokenizer_mode=model_config.tokenizer_mode,
        trust_remote_code=model_config.trust_remote_code,
        revision=model_config.tokenizer_revision,
        truncation_side=model_config.truncation_side)
