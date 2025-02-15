# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Union, cast, overload

import torch

from vllm.logger import init_logger
from vllm.utils import cdiv, get_dtype_size

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock, ReqKVCacheBlocks

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
    num_heads: int
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


@dataclass
class SlidingWindowSpec(KVCacheSpec):
    num_heads: int
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    sliding_window: int

    @property
    def type_id(self) -> str:
        return f"sliding_window_{self.sliding_window}_{self.block_size}_{self.page_size_bytes}"  # noqa

    @property
    def page_size_bytes(self) -> int:
        return  2 * self.block_size * self.num_kv_heads * self.head_size \
                * get_dtype_size(self.dtype)

    def bytes_for_tokens(self, num_tokens: int) -> int:
        num_tokens = min(num_tokens, self.sliding_window)
        return cdiv(num_tokens, self.block_size) * self.page_size_bytes


@dataclass
class KVCacheTensorBase:
    """
    A dataclass for specifying how the workers should initialize the KV cache
    for a layer.
    """
    pass


@dataclass
class KVCacheNewTensor(KVCacheTensorBase):
    """
    Initialize the KV cache with a tensor of `size` bytes.
    """
    size: int  # The size of KV cache Tensor in bytes


@dataclass
class KVCacheReuseTensor(KVCacheTensorBase):
    """
    Reuse the KV cache tensor of `layer_name` for the current layer.
    """
    reused_layer_name: str


@dataclass
class KVCacheGroup:
    """
    A dataclass for specifying the KV cache group of a model.
    """
    # The names of layers in this group
    layer_names: List[str]
    # The KV cache spec of this group
    kv_cache_spec: KVCacheSpec


@dataclass
class KVCacheConfig:
    """
    The KV cache configuration of a model.
    """
    """The number of KV cache blocks"""
    num_blocks: int
    """layer_name -> how to initialize KV cache for that layer"""
    tensors: Dict[str, KVCacheTensorBase]
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
    groups: List[KVCacheGroup]


@dataclass
class GroupedBlockIDs:
    # A list of block IDs for each group of KV cache blocks
    _block_ids: List[List[int]]

    def __init__(self, block_ids: List[List[int]]):
        self._block_ids = block_ids

    @classmethod
    def from_kv_cache_blocks(cls, kv_cache_blocks: "ReqKVCacheBlocks"):
        return cls(
            block_ids=[[blk.block_id for blk in kv_cache_blocks_one_group]
                       for kv_cache_blocks_one_group in kv_cache_blocks])

    def extend(self, new_block_ids: "GroupedBlockIDs") -> None:
        for i, block_ids in enumerate(new_block_ids._block_ids):
            self._block_ids[i].extend(block_ids)

    def __add__(self, other: "GroupedBlockIDs") -> "GroupedBlockIDs":
        return GroupedBlockIDs(block_ids=[
            a + b for a, b in zip(self._block_ids, other._block_ids)
        ])

    def get_group(self, group_idx: int) -> List[int]:
        return self._block_ids[group_idx]


MayGroupedBlockIDs = Union[GroupedBlockIDs, List[int]]
MayGroupedInt = Union[int, List[int]]


class BlockIDGenerator:
    num_kv_cache_groups: int

    @overload
    @classmethod
    def generate(cls, kv_cache_blocks: List[KVCacheBlock]) -> List[int]:
        ...

    @overload
    @classmethod
    def generate(cls,
                 kv_cache_blocks: List[List[KVCacheBlock]]) -> GroupedBlockIDs:
        ...

    @classmethod
    def generate(
        cls, kv_cache_blocks: Union[List["KVCacheBlock"],
                                    List[List["KVCacheBlock"]]]
    ) -> MayGroupedBlockIDs:
        if cls.num_kv_cache_groups == 1:
            kv_cache_blocks = cast(List["KVCacheBlock"], kv_cache_blocks)
            return [blk.block_id for blk in kv_cache_blocks]
        else:
            kv_cache_blocks = cast(List[List["KVCacheBlock"]], kv_cache_blocks)
            return GroupedBlockIDs.from_kv_cache_blocks(kv_cache_blocks)
