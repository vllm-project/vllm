# SPDX-License-Identifier: Apache-2.0

import copy
from dataclasses import dataclass
from typing import Optional

import torch
from typing_extensions import Self

from vllm.config import VllmConfig
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

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        """
        The maximum possible memory usage of this KV cache in bytes.

        Returns:
            The KV cache size in bytes
        """
        raise NotImplementedError

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of KVCacheSpec objects into a single KVCacheSpec object.
        """
        assert all(spec.type_id == specs[0].type_id for spec in specs[1:]), (
            "All layers in the same KV cache group must share the same "
            "type_id.")
        return copy.deepcopy(specs[0])


@dataclass
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    use_mla: bool

    @property
    def page_size_bytes(self) -> int:
        # For MLA we only store a single latent vector
        coef = 1 if self.use_mla else 2
        return coef * self.block_size * self.num_kv_heads * self.head_size \
                * get_dtype_size(self.dtype)


@dataclass
class FullAttentionSpec(AttentionSpec):
    sliding_window: Optional[int] = None
    """
    When hybrid allocator is disabled and the model contains both full 
    attention layers and sliding window attention layers, sliding 
    window attention are regarded as full attention in KV cache manager 
    (blocks are allocated for all tokens), while computed as sliding window 
    attention in model runner.
    In this case, we use FullAttentionSpec and record the sliding window size.
    Default to None for not using sliding window attention.
    """

    @property
    def type_id(self) -> str:
        return f"full_attention_{self.block_size}_{self.page_size_bytes}"

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of FullAttentionSpec objects into a single 
        FullAttentionSpec object.
        """
        merged_spec = super().merge(specs)
        sliding_window = set(spec.sliding_window for spec in specs
                             if spec.sliding_window is not None)
        if len(sliding_window) == 0:
            merged_spec.sliding_window = None
        elif len(sliding_window) == 1:
            merged_spec.sliding_window = sliding_window.pop()
        else:
            raise ValueError(
                "All sliding window layers in the same KV cache group "
                "must have the same window size.")
        return merged_spec


@dataclass
class SlidingWindowSpec(AttentionSpec):
    sliding_window: int

    def __post_init__(self):
        assert not self.use_mla, "MLA is not supported for sliding window"

    @property
    def type_id(self) -> str:
        return f"sliding_window_{self.sliding_window}_{self.block_size}_{self.page_size_bytes}"  # noqa

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens)

        # During chunked prefill, we allocate KV cache for the last
        # `self.sliding_window-1` computed tokens plus the newly scheduled
        # tokens. And we won't allocate KV cache for more than `max_model_len`
        # tokens.
        num_tokens = min(self.sliding_window - 1 + max_num_batched_tokens,
                         max_model_len)

        # +1 here because the sliding window may not start from the beginning
        # of the block. For example, if the block size is 4 and num_token
        # is 4, we need two blocks [XXCD] [EF] to store the sliding
        # window [CDEF] of 6 tokens.
        return (cdiv(num_tokens, self.block_size) + 1) * self.page_size_bytes


@dataclass
class KVCacheTensor:
    """
    A dataclass for specifying how the workers should initialize the KV cache
    for a layer. Only contains the size of KV cache for that layer for now. Will
    be extended to support multiple layers sharing the same memory pool.
    """
    size: int  # The size of KV cache Tensor in bytes


@dataclass
class KVCacheGroupSpec:
    """
    Represents a group of model layers that share the same KV cache block table.
    These layers are regarded as one layer in the KV cache manager.
    """
    # The names of model layers in this group
    layer_names: list[str]
    # The KV cache spec of this manager layer
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
    The kv cache groups of the model.
    The layers in the models are repeated with some patterns, e.g., a model
    with 10 full attention layers and 20 sliding window attention layers can be
    regarded as repeating the pattern (1 * full, 2 * sw) 10 times. 
    The KVCacheManager allocates different block tables for each of the 3 layers
    in the pattern, and repeats each of them 10 times to generate the 
    block_table for the 30 layers in the model.
    Therefore, we can group the layers in the model into 3 groups, each of which
    contains 10 layers in the model.
    The KVCacheManager allocates the block_table for each group based on its
    kv_cache spec, and the model runner applies the block table to each layer 
    in the group.
    For example:
    1. A model only uses full attention. The pattern is 
    (num_hidden_layers * full), so there is only one group and the block table 
    is shared by all layers.
    2. (WIP) A model with 10 full attention layers and 20 sliding window 
    attention layers. There are 3 layers in the pattern (1 * full, 2 * sw), so 
    there are 3 groups, each of which represents 10 layers in the model.
    """
    kv_cache_groups: list[KVCacheGroupSpec]
