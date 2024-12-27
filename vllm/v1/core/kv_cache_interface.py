from dataclasses import dataclass
import math
from typing import Dict, List, Protocol, runtime_checkable
import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.utils import get_dtype_size

# TODO: add comment for
# layer_id:
# layer_cnt:
# group_id:


# worker -> scheduler about the model architecture
# TODO(Chen): find a better name
@dataclass
class LayerCache:

    @property
    def key(self) -> str:
        raise NotImplementedError

    @property
    def page_size(self) -> int:
        raise NotImplementedError

    def memory_size(self, num_tokens: int) -> int:
        raise NotImplementedError


@dataclass
class SelfAttentionCache(LayerCache):
    block_size: int
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype

    @property
    def key(self) -> str:
        return "self_attention"

    @property
    def page_size(self) -> int:
        return  2 * self.block_size * self.num_kv_heads * self.head_size \
                * get_dtype_size(self.dtype)

    def memory_size(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.block_size) * self.page_size


@dataclass
class SlidingWindowCache(LayerCache):
    block_size: int
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    sliding_window: int

    @property
    def key(self) -> str:
        return f"sliding_window_{self.sliding_window}"

    @property
    def page_size(self) -> int:
        return  2 * self.block_size * self.num_kv_heads * self.head_size \
                * get_dtype_size(self.dtype)

    def memory_size(self, num_tokens: int) -> int:
        num_tokens = min(num_tokens, self.sliding_window)
        return math.ceil(num_tokens / self.block_size) * self.page_size


@dataclass
class LayerConfig:
    # TODO (Chen): remove layer_cnt and use layer_id to index kv_cache in models
    layer_id_mapping: Dict[str, int]  # layer_cnt -> layer_id
    layers: Dict[str, LayerCache]


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
    tensors: Dict[int, KVCacheTensor]  # layer_cnt -> KVCacheTensor
    block_table_sharing: Dict[str, List[str]]  # group_id -> List[layer_id]
    layer_config: LayerConfig
