import math
from dataclasses import dataclass
from typing import Dict, List

import torch

from vllm.logger import init_logger
from vllm.utils import get_dtype_size

logger = init_logger(__name__)


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


KVCacheSpec = Dict[str, KVCacheSpecBase]


@dataclass
class KVCacheTensor:
    size: int  # in bytes


@dataclass
class KVCacheConfig:
    # layer_name -> the kv_cache tensor configuration for the layer
    tensors: Dict[str, KVCacheTensor]

    # [group_id][layer_name in the group]. One group containing all
    # layer_names if the Spec for kv_cache of all layers are the same
    groups: List[List[str]]

    # the KVCacheSpec of the model
    kv_cache_spec: KVCacheSpec
