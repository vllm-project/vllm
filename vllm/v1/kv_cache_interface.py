# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.utils import cdiv, get_dtype_size

logger = init_logger(__name__)


@dataclass
class KVCacheSpecBase:
    """
    A base class for specifying the KV cache format of one layer.
    """

    # number of tokens in a block
    block_size: int

    @property
    def type_id(self) -> str:
        """
        The type identifier of this KV cache.
        Return different strings for layers with different KV cache type (e.g., 
        different number of tokens like full attention vs sliding window 
        attention, different KV cache size per token like layers with different 
        number of heads)

        Returns:
            The type identifier of this KV cache.
        """
        raise NotImplementedError

    @property
    def page_size_bytes(self) -> int:
        """
        The size of a page with `block_size` tokens in bytes.

        Returns:
            The page size
        """
        raise NotImplementedError

    def bytes_for_tokens(self, num_tokens: int) -> int:
        """
        The KV cache size for `num_tokens` tokens in bytes. Returns the real
        memory size after padding `num_tokens` to full blocks.

        Returns:
            The KV cache size
        """
        raise NotImplementedError


@dataclass
class FullAttentionSpec(KVCacheSpecBase):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype

    @property
    def type_id(self) -> str:
        return f"full_attention_{self.block_size}_{self.page_size_bytes}"

    @property
    def page_size_bytes(self) -> int:
        return  2 * self.block_size * self.num_kv_heads * self.head_size \
                * get_dtype_size(self.dtype)

    def bytes_for_tokens(self, num_tokens: int) -> int:
        return cdiv(num_tokens, self.block_size) * self.page_size_bytes


KVCacheSpec = dict[str, KVCacheSpecBase]


@dataclass
class KVCacheTensor:
    """
    A dataclass for specifying how the workers should initialize the KV cache
    for a layer. Only contains the size of KV cache for that layer for now. Will
    be extended to support multiple layers sharing the same memory pool.
    """
    size: int  # The size of KV cache Tensor in bytes


@dataclass
class KVCacheConfig:
    """
    The KV cache configuration of a model.
    """
    """The number of KV cache blocks"""
    num_blocks: int
    """layer_name -> how to initialize KV cache for that layer"""
    tensors: dict[str, KVCacheTensor]
    """
    A list of kv-cache groups. Each group includes a set of layers with
    the same kv-cache spec, and the total page_size of layers inside a group
    is same across all groups (as the KVCacheManager only supports allocating
    pages of the same size). For example:
    1. A model only uses full attention: one group with all layers in the model.
    2. (not implemented yet) A model with the same number of full attention
    layers and sliding window attention layers: two groups, one for full
    attention layers and one for sliding window attention layers.
    3. (not implemented yet) A model with 2 full attention layers and 4 sliding 
    window attention layers: three groups, (full * 2), (sw * 2), (sw * 2).
    """
    groups: list[list[str]]
    """the KVCacheSpec of the model"""
    kv_cache_spec: KVCacheSpec
