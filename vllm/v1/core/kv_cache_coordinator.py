# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Callable, Optional

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager, get_manager_for_kv_cache_spec)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.request import Request


class KVCacheCoordinator(ABC):
    """
    Coordinate the KV cache of different KV cache groups.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        caching_hash_fn: Callable,
        enable_kv_cache_events: bool,
    ):
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len

        self.block_pool = BlockPool(kv_cache_config.num_blocks, enable_caching,
                                    enable_kv_cache_events)

        # Needs special handling for find_longest_cache_hit if eagle is enabled
        self.use_eagle = use_eagle
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                block_pool=self.block_pool,
                kv_cache_group_id=i,
                caching_hash_fn=caching_hash_fn,
            ) for i, kv_cache_group in enumerate(
                self.kv_cache_config.kv_cache_groups))

    def get_num_blocks_to_allocate(
            self, request_id: str, num_tokens: int,
            new_computed_blocks: tuple[list[KVCacheBlock], ...]) -> int:
        """
        Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including 
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.

        Returns:
            The number of blocks.
        """
        num_blocks_to_allocate = 0
        for i, manager in enumerate(self.single_type_managers):
            num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                request_id, num_tokens, new_computed_blocks[i])
        return num_blocks_to_allocate

    def save_new_computed_blocks(
            self, request_id: str,
            new_computed_blocks: tuple[list[KVCacheBlock], ...]) -> None:
        """
        Add the new computed blocks to the request.

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
        """
        for i, manager in enumerate(self.single_type_managers):
            manager.save_new_computed_blocks(request_id,
                                             new_computed_blocks[i])

    def allocate_new_blocks(self, request_id: str,
                            num_tokens: int) -> tuple[list[KVCacheBlock], ...]:
        """
        Allocate new blocks for the request to give it at least `num_tokens` 
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including 
                tokens that are already allocated).

        Returns:
            The new allocated blocks.
        """
        return tuple(
            manager.allocate_new_blocks(request_id, num_tokens)
            for manager in self.single_type_managers)

    def cache_blocks(self, request: Request, block_hashes: list[BlockHash],
                     num_computed_tokens: int) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            block_hashes: The block hashes of the request.
            num_tokens: The total number of tokens that need to be cached 
                (including tokens that are already cached).
        """
        for manager in self.single_type_managers:
            manager.cache_blocks(request, block_hashes, num_computed_tokens)

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        for manager in self.single_type_managers:
            manager.free(request_id)

    def get_num_common_prefix_blocks(self, request_id: str,
                                     num_running_requests: int) -> list[int]:
        """
        Get the number of common prefix blocks for a request.

        Args:
            request_id: The request ID.
            block_hashes: The block hashes of the request.

        Returns:
            The number of common prefix blocks.
        """
        num_blocks_per_group = [
            manager.get_num_common_prefix_blocks(request_id,
                                                 num_running_requests)
            for manager in self.single_type_managers
        ]
        return num_blocks_per_group

    def remove_skipped_blocks(self, request_id: str,
                              num_computed_tokens: int) -> None:
        """
        Remove the blocks that are no longer needed from `blocks` and replace 
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            num_computed_tokens: The number of tokens that have been computed.
        """
        for manager in self.single_type_managers:
            manager.remove_skipped_blocks(request_id, num_computed_tokens)

    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        """
        Get the blocks for the request.
        """
        return tuple(
            manager.req_to_blocks.get(request_id) or []
            for manager in self.single_type_managers)

    @abstractmethod
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        pass


class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for models with only one KV cache group. This is the
    case for models with only one KV cache type, e.g., all attention layers use
    full attention or all attention layers use sliding window attention.
    """

    def __init__(self, kv_cache_config: KVCacheConfig, max_model_len: int,
                 use_eagle: bool, enable_caching: bool,
                 caching_hash_fn: Callable, enable_kv_cache_events: bool):
        super().__init__(kv_cache_config, max_model_len, use_eagle,
                         enable_caching, caching_hash_fn,
                         enable_kv_cache_events)
        self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[
            0].kv_cache_spec
        self.block_size = self.kv_cache_spec.block_size
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "UnitaryKVCacheCoordinator assumes only one kv cache group")

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        hit_blocks = self.single_type_managers[0].find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.block_pool,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=self.use_eagle,
        )
        return hit_blocks, len(hit_blocks[0]) * self.block_size


class HybridKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for hybrid models with multiple KV cache types, and
    thus multiple kv cache groups.
    To simplify `find_longest_cache_hit`, it only supports the combination of 
    two types of KV cache groups, and one of them must be full attention.
    May extend to more general cases in the future.
    """

    def __init__(self, kv_cache_config: KVCacheConfig, max_model_len: int,
                 use_eagle: bool, enable_caching: bool,
                 caching_hash_fn: Callable, enable_kv_cache_events: bool):
        super().__init__(kv_cache_config, max_model_len, use_eagle,
                         enable_caching, caching_hash_fn,
                         enable_kv_cache_events)
        self.verify_and_split_kv_cache_groups()

    def verify_and_split_kv_cache_groups(self) -> None:
        """
        Verifies that the model has exactly two types of KV cache groups, and 
        one of them is full attention. Then, split the kv cache groups into full
        attention groups and other groups.
        """
        full_attention_type_id: Optional[str] = None
        other_type_id: Optional[str] = None
        self.full_attention_group_ids: list[int] = []
        self.other_group_ids: list[int] = []
        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            if isinstance(g.kv_cache_spec, FullAttentionSpec):
                if full_attention_type_id is None:
                    full_attention_type_id = g.kv_cache_spec.type_id
                else:
                    assert full_attention_type_id == g.kv_cache_spec.type_id, (
                        "HybridKVCacheCoordinator assumes exactly one type of "
                        "full attention groups now.")
                self.full_attention_group_ids.append(i)
            else:
                if other_type_id is None:
                    other_type_id = g.kv_cache_spec.type_id
                else:
                    assert other_type_id == g.kv_cache_spec.type_id, (
                        "HybridKVCacheCoordinator assumes "
                        "exactly one other type of groups now.")
                self.other_group_ids.append(i)

        assert full_attention_type_id is not None, (
            "HybridKVCacheCoordinator assumes exactly one type of full "
            "attention groups now.")
        assert other_type_id is not None, (
            "HybridKVCacheCoordinator assumes exactly one type of other "
            "groups now.")

        self.full_attention_manager_cls = FullAttentionManager
        self.other_attention_cls = self.single_type_managers[
            self.other_group_ids[0]].__class__

        self.full_attention_spec = self.kv_cache_config.kv_cache_groups[
            self.full_attention_group_ids[0]].kv_cache_spec
        self.other_spec = self.kv_cache_config.kv_cache_groups[
            self.other_group_ids[0]].kv_cache_spec

        self.full_attention_block_size = self.full_attention_spec.block_size
        self.other_block_size = self.other_spec.block_size
        assert self.other_block_size % self.full_attention_block_size == 0, (
            "KVCacheCoordinator assumes the block_size of full attention "
            "layers is divisible by other layers now.")

        if max(self.full_attention_group_ids) < min(self.other_group_ids):
            self.full_attn_first = True
        elif max(self.other_group_ids) < min(self.full_attention_group_ids):
            self.full_attn_first = False
        else:
            raise ValueError(
                "HybridKVCacheCoordinator assumes the full "
                "attention group ids and other attention group ids "
                "do not interleave, either full attention group ids "
                "are before other attention group ids or vice versa."
                "This is for simplifying merging hit_blocks_full_attn and "
                "hit_blocks_other_attn to hit_blocks.")

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        Find the longest cache hit for the request.

        Args:
            block_hashes: The block hashes of the request.
            max_cache_hit_length: The maximum length of the cache hit.

        Returns:
            A tuple containing:
                - A list of the cache hit blocks for each single type manager.
                - The number of tokens of the longest cache hit.
        """
        # First, find the longest cache hit for full attention.
        hit_blocks_full_attn = (
            self.full_attention_manager_cls.find_longest_cache_hit(
                block_hashes=block_hashes,
                max_length=max_cache_hit_length,
                kv_cache_group_ids=self.full_attention_group_ids,
                block_pool=self.block_pool,
                kv_cache_spec=self.full_attention_spec,
                use_eagle=self.use_eagle,
            ))
        hit_length = len(
            hit_blocks_full_attn[0]) * self.full_attention_block_size

        # Next, find the cache hit for the other attention WITHIN
        # the cache hit of full attention.
        hit_blocks_other_attn = (
            self.other_attention_cls.find_longest_cache_hit(
                block_hashes=block_hashes,
                max_length=hit_length,
                kv_cache_group_ids=self.other_group_ids,
                block_pool=self.block_pool,
                kv_cache_spec=self.other_spec,
                use_eagle=self.use_eagle,
            ))
        hit_length = len(hit_blocks_other_attn[0]) * self.other_block_size

        # NOTE: the prefix cache hit length must be a multiple of block_size as
        # we don't support partial block cache hit yet. The cache hit length
        # of other attention is ensured to be a multiple of the block size of
        # full attention layers in current implementation, because hit_length is
        # a multiple of other attention's block size, and other attention's
        # block size is a multiple of full attention's block size (verified in
        # `verify_and_split_kv_cache_groups`).
        assert hit_length % self.full_attention_block_size == 0

        # Truncate the full attention cache hit to the length of the
        # cache hit of the other attention.
        for group_hit_blocks in hit_blocks_full_attn:
            del group_hit_blocks[hit_length // self.full_attention_block_size:]

        # Merge the hit blocks of full attention and other attention.
        if self.full_attn_first:
            hit_blocks = hit_blocks_full_attn + hit_blocks_other_attn
        else:
            hit_blocks = hit_blocks_other_attn + hit_blocks_full_attn
        return hit_blocks, hit_length


def get_kv_cache_coordinator(
        kv_cache_config: KVCacheConfig, max_model_len: int, use_eagle: bool,
        enable_caching: bool, caching_hash_fn: Callable,
        enable_kv_cache_events: bool) -> KVCacheCoordinator:
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinator(kv_cache_config, max_model_len,
                                         use_eagle, enable_caching,
                                         caching_hash_fn,
                                         enable_kv_cache_events)
    return HybridKVCacheCoordinator(kv_cache_config, max_model_len, use_eagle,
                                    enable_caching, caching_hash_fn,
                                    enable_kv_cache_events)
