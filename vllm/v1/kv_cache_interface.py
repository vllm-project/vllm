# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.utils import cdiv, get_dtype_size

logger = init_logger(__name__)


@dataclass
class KVCacheSpec:
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
class FullAttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    use_mla: bool

    @property
    def type_id(self) -> str:
        return f"full_attention_{self.block_size}_{self.page_size_bytes}"

    @property
    def page_size_bytes(self) -> int:
        # For MLA we only store a single latent vector
        coef = 1 if self.use_mla else 2
        return coef * self.block_size * self.num_kv_heads * self.head_size \
                * get_dtype_size(self.dtype)

    def bytes_for_tokens(self, num_tokens: int) -> int:
        return cdiv(num_tokens, self.block_size) * self.page_size_bytes


@dataclass
class KVCacheTensor:
    """
    A dataclass for specifying how the workers should initialize the KV cache
    for a layer. Only contains the size of KV cache for that layer for now. Will
    be extended to support multiple layers sharing the same memory pool.
    """
    size: int  # The size of KV cache Tensor in bytes


@dataclass
class VirtualLayer:
    """
    A dataclass for specifying a virtual layer, which represents multiple layers
    that can share the same block_table.
    """
    # The names of layers represented by this virtual layer
    layer_names: list[str]
    # The KV cache spec of this virtual layer
    kv_cache_spec: KVCacheSpec


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
    The virtual_layers of the model.
    The layers in the models are repeated with some patterns, e.g., a model
    with 10 full attention layers and 20 sliding window attention layers can be
    regarded as repeating the pattern (1 * full, 2 * sw) 10 times. And we regard
    this pattern as virtual layers (3 virtual layers in this case, each
    representing 10 layers).
    The KVCacheManager allocates the blocks for each virtual layer, and the
    model runner applies the block table of the virtual layer to all layers 
    represented by it.
    For example:
    1. A model only uses full attention. There is only one virtual layer, 
    and the block table is shared by all layers.
    2. (WIP) A model with 10 full attention layers and 20 sliding window 
    attention. There are 3 virtual layers (1 * full, 2 * sw), and the block 
    table of each virtual layer is shared by 10 layers of the same type.
    """
    virtual_layers: list[VirtualLayer]
