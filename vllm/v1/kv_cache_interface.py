# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from dataclasses import dataclass
from math import prod
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
    attention_chunk_size: Optional[int] = None
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
    def merge_window_sizes(cls, window_sizes: set[int]) -> Optional[int]:
        if len(window_sizes) == 0:
            return None
        elif len(window_sizes) == 1:
            return window_sizes.pop()
        else:
            raise ValueError(
                "All attention layers in the same KV cache group must have the "
                "same window size.")

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of FullAttentionSpec objects into a single 
        FullAttentionSpec object.
        """
        merged_spec = super().merge(specs)
        sliding_window = set(spec.sliding_window for spec in specs
                             if spec.sliding_window is not None)
        attention_chunk_size = set(spec.attention_chunk_size for spec in specs
                                   if spec.attention_chunk_size is not None)

        merged_spec.sliding_window = cls.merge_window_sizes(sliding_window)
        merged_spec.attention_chunk_size = (
            cls.merge_window_sizes(attention_chunk_size))
        assert (
            (merged_spec.sliding_window is not None) +
            (merged_spec.attention_chunk_size is not None) <= 1
        ), ("Model with both sliding window layers and chunked local attention "
            "layers is not supported.")
        return merged_spec


@dataclass
class ChunkedLocalAttentionSpec(AttentionSpec):
    attention_chunk_size: int

    @property
    def type_id(self) -> str:
        return (
            f"local_attention_{self.attention_chunk_size}_{self.block_size}_{self.page_size_bytes}"
        )  # noqa

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens)

        # During chunked prefill, we allocate KV cache for at most
        # `self.attention_chunk_size` computed tokens plus the newly scheduled
        # tokens. And we won't allocate KV cache for more than `max_model_len`
        # tokens.
        num_tokens = min(self.attention_chunk_size + max_num_batched_tokens,
                         max_model_len)

        return cdiv(num_tokens, self.block_size) * self.page_size_bytes


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
class MambaSpec(KVCacheSpec):
    shapes: tuple[tuple[int, ...], ...]
    dtype: torch.dtype
    page_size_padded: Optional[int] = None
    mamba_type: str = "mamba2"

    def __post_init__(self):
        self.num_elements = sum(prod(shape) for shape in self.shapes)

    @property
    def type_id(self) -> str:
        return f"mamba_{self.shapes}_{self.dtype}_{self.mamba_type}"

    @property
    def page_size_bytes(self) -> int:
        page_size = self.num_elements * get_dtype_size(self.dtype)
        if self.page_size_padded is not None:
            assert self.page_size_padded >= page_size
            return self.page_size_padded
        return page_size

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # We allocate 1 block for each request now, so max_memory_usage_bytes is
        # the same as page_size_bytes.
        # Need to update this when supporting prefix caching.
        return self.page_size_bytes


@dataclass
class ShortConvSpec(MambaSpec):

    @property
    def type_id(self) -> str:
        return f"short_conv_{self.shapes}_{self.dtype}"


@dataclass
class KVCacheTensor:
    """
    A class for specifying how the workers should initialize the KV cache.
    """
    size: int  # size of the KV cache tensor in bytes
    shared_by: list[str]  # layer names that share the same KV cache tensor


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
    """How should model runner initialize the KV cache tensors for each layer"""
    kv_cache_tensors: list[KVCacheTensor]
    """
    The kv cache groups of the model.
    For models with only one type of attention, there is only one group that
    contains all layers.
    For models with multiple types of attention, there will be multiple groups,
    see `_get_kv_cache_config_uniform_page_size` for more details.
    """
    kv_cache_groups: list[KVCacheGroupSpec]
