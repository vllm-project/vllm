from dataclasses import dataclass
import math
from typing import Dict, List, Protocol, runtime_checkable
import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.utils import get_dtype_size

# TODO: add comment for
# layer_name:
# layer_cnt:
# group_id:


# worker -> scheduler about the model architecture
# TODO(Chen): find a better name
@dataclass
class KVCacheSpecBase:
    block_size: int

    @property
    def key(self) -> str:
        raise NotImplementedError

    @property
    def page_size_bytes(self) -> int:
        raise NotImplementedError

    def bytes_for_tokens(self, num_tokens: int) -> int:
        raise NotImplementedError


@dataclass
class FullAttentionSpec(KVCacheSpecBase):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype

    @property
    def key(self) -> str:
        return f"full_attention_{self.block_size}_{self.bytes_for_tokens(1)}"

    @property
    def page_size_bytes(self) -> int:
        return  2 * self.block_size * self.num_kv_heads * self.head_size \
                * get_dtype_size(self.dtype)

    def bytes_for_tokens(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.block_size) * self.page_size_bytes


@dataclass
class SlidingWindowSpec(KVCacheSpecBase):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    sliding_window: int

    @property
    def key(self) -> str:
        return f"sliding_window_{self.sliding_window}_{self.block_size}_{self.bytes_for_tokens(1)}"

    @property
    def page_size_bytes(self) -> int:
        return  2 * self.block_size * self.num_kv_heads * self.head_size \
                * get_dtype_size(self.dtype)

    def bytes_for_tokens(self, num_tokens: int) -> int:
        num_tokens = min(num_tokens, self.sliding_window)
        return math.ceil(num_tokens / self.block_size) * self.page_size_bytes


KVCacheSpec = Dict[str, KVCacheSpecBase]


# scheduler -> worker about the kv cache configuration
@dataclass
class KVCacheTensor:
    size: int  # in bytes


@dataclass
class KVCacheTensorShareBuffer(KVCacheTensor):
    start_bias: int


@dataclass
class KVCacheTensorSeperate(KVCacheTensor):
    pass


@dataclass
class KVCacheConfig:
    buffer_size: int  # -1 if do not need a global buffer
    tensors: Dict[str, KVCacheTensor]  # layer_name -> KVCacheTensor
    groups: List[List[str]]  # group_id -> [layer_name in the group]
    kv_cache_spec: KVCacheSpec
