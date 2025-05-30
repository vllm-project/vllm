# SPDX-License-Identifier: Apache-2.0
from abc import abstractmethod
from collections import defaultdict
from typing import Callable

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashType, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    SingleTypeKVCacheManager, get_manager_for_kv_cache_spec)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.request import Request


class KVCacheCoordinator:
    """
    Coordinator the KV cache of different KV cache groups.
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
        self.single_type_managers: list[SingleTypeKVCacheManager] = []
        for i in range(len(self.kv_cache_config.kv_cache_groups)):
            kv_cache_spec = self.kv_cache_config.kv_cache_groups[
                i].kv_cache_spec
            self.single_type_managers.append(
                get_manager_for_kv_cache_spec(
                    kv_cache_spec=kv_cache_spec,
                    block_pool=self.block_pool,
                    use_eagle=use_eagle,
                    kv_cache_group_id=i,
                    caching_hash_fn=caching_hash_fn,
                ))

    def get_num_blocks_to_allocate(
            self, request_id: str, num_tokens: int,
            new_computed_blocks: list[list[KVCacheBlock]]) -> int:
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
            new_computed_blocks: list[list[KVCacheBlock]]) -> None:
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
                            num_tokens: int) -> list[list[KVCacheBlock]]:
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
        new_blocks = []
        for manager in self.single_type_managers:
            new_blocks.append(
                manager.allocate_new_blocks(request_id, num_tokens))
        return new_blocks

    def cache_blocks(self, request: Request,
                     block_hashes: dict[int, list[BlockHashType]],
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
            manager.cache_blocks(request, block_hashes[manager.block_size],
                                 num_computed_tokens)

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        for manager in self.single_type_managers:
            manager.free(request_id)

    def get_num_common_prefix_blocks(
        self,
        request_id: str,
        num_running_requests: int,
    ) -> list[int]:
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

    def get_blocks(self, request_id: str) -> list[list[KVCacheBlock]]:
        """
        Get the blocks for the request.
        """
        return [
            manager.req_to_blocks[request_id]
            for manager in self.single_type_managers
        ]

    @abstractmethod
    def find_longest_cache_hit(
            self, request_id: str,
            block_hashes_dict: dict[int, list[BlockHashType]],
            max_cache_hit_length: int) -> tuple[list[list[KVCacheBlock]], int]:
        pass


class UnifiedKVCacheCoordinator(KVCacheCoordinator):

    def __init__(self, kv_cache_config: KVCacheConfig, max_model_len: int,
                 use_eagle: bool, enable_caching: bool,
                 caching_hash_fn: Callable, enable_kv_cache_events: bool):
        super().__init__(kv_cache_config, max_model_len, use_eagle,
                         enable_caching, caching_hash_fn,
                         enable_kv_cache_events)
        self.block_size = self.kv_cache_config.kv_cache_groups[
            0].kv_cache_spec.block_size
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "UnifiedKVCacheCoordinator assumes only one kv cache group")

    def find_longest_cache_hit(
            self, request_id: str,
            block_hashes_dict: dict[int, list[BlockHashType]],
            max_cache_hit_length: int) -> tuple[list[list[KVCacheBlock]], int]:
        hit_blocks = self.single_type_managers[0].find_longest_cache_hit(
            block_hashes_dict[self.block_size], max_cache_hit_length, [0])
        return hit_blocks, len(hit_blocks[0]) * self.block_size


class HybridKVCacheCoordinator(KVCacheCoordinator):

    def __init__(self, kv_cache_config: KVCacheConfig, max_model_len: int,
                 use_eagle: bool, enable_caching: bool,
                 caching_hash_fn: Callable, enable_kv_cache_events: bool):
        super().__init__(kv_cache_config, max_model_len, use_eagle,
                         enable_caching, caching_hash_fn,
                         enable_kv_cache_events)
        self.initialize_group_ids()

    def initialize_group_ids(self) -> None:
        """
        For simplicity, find_longest_cache_hit makes some assumptions on the
        model architecture instead of provides a general solution. This function
        checks if the assumptions hold.
        NOTE(Chen): Please open an issue to discuss if you need other cases.

        TODO: add more notes
        """
        groups_by_type_id: dict[str, list[int]] = defaultdict(list)
        full_attention_type_ids: set[str] = set()
        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            groups_by_type_id[g.kv_cache_spec.type_id].append(i)
            if isinstance(g.kv_cache_spec, FullAttentionSpec):
                full_attention_type_ids.add(g.kv_cache_spec.type_id)

        assert len(full_attention_type_ids) == 1, (
            "find_longest_cache_hit assumes hybrid models have exactly "
            "one type of full attention groups now")
        assert len(groups_by_type_id) == 2, (
            "find_longest_cache_hit assumes hybrid models have exactly "
            "one other type of groups except full attention now")

        self.full_attention_group_ids = groups_by_type_id[next(
            iter(full_attention_type_ids))]
        self.other_group_ids = groups_by_type_id[next(
            iter(groups_by_type_id.keys() - full_attention_type_ids))]

        self.full_attention_block_size = self.kv_cache_config.kv_cache_groups[
            self.full_attention_group_ids[0]].kv_cache_spec.block_size
        self.other_block_size = self.kv_cache_config.kv_cache_groups[
            self.other_group_ids[0]].kv_cache_spec.block_size
        if self.other_block_size % self.full_attention_block_size != 0:
            raise NotImplementedError(
                "KVCacheCoordinator assumes the block_size of the full "
                "attention layer is divisible by other layers now.")

    def find_longest_cache_hit(
        self,
        request_id: str,
        block_hashes_dict: dict[int, list[BlockHashType]],
        max_cache_hit_length: int,
    ) -> tuple[list[list[KVCacheBlock]], int]:
        """
        Find the longest cache hit for the request.

        Args:
            request_id: The request ID.
            block_hashes_dict: The block hashes of the request.
            max_cache_hit_length: The maximum length of the cache hit.

        Returns:
            A tuple containing:
                - A list of the cache hit blocks for each single type manager.
                - The number of tokens of the longest cache hit.
        """
        # For simplicity, we assume the first manager is for full
        # attention layers, and the block_size of full attention layers
        # is divisible by other attention layers. This has been verified
        # in verify_support_find_longest_cache_hit().

        # First, find the longest cache hit for full attention.
        hit_blocks_full_attn = self.single_type_managers[
            0].find_longest_cache_hit(
                block_hashes_dict[self.full_attention_block_size],
                max_length=max_cache_hit_length,
                kv_cache_group_ids=self.full_attention_group_ids)
        hit_length = len(
            hit_blocks_full_attn[0]) * self.full_attention_block_size

        # Next, find the cache hit for the other attention WITHIN
        # the cache hit of full attention.
        hit_blocks_other_attn = self.single_type_managers[
            1].find_longest_cache_hit(block_hashes_dict[self.other_block_size],
                                      max_length=hit_length,
                                      kv_cache_group_ids=self.other_group_ids)
        hit_length = len(hit_blocks_other_attn[0]) * self.other_block_size
        assert hit_length % self.full_attention_block_size == 0

        # Truncate the full attention cache hit to the length of the
        # cache hit of the other attention.
        for i in range(len(hit_blocks_full_attn)):
            del hit_blocks_full_attn[i][hit_length //
                                        self.full_attention_block_size:]
        # Merge the hit blocks of full attention and other attention.
        hit_blocks = hit_blocks_other_attn
        for group_id, blocks in enumerate(hit_blocks_full_attn):
            del blocks[hit_length // self.full_attention_block_size:]
            # NOTE: there is only one full attention group in most cases. So
            # the time complexity of insert is fine.
            hit_blocks.insert(group_id, blocks)
        return hit_blocks, hit_length


def get_kv_cache_coordinator(
        kv_cache_config: KVCacheConfig, max_model_len: int, use_eagle: bool,
        enable_caching: bool, caching_hash_fn: Callable,
        enable_kv_cache_events: bool) -> KVCacheCoordinator:
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnifiedKVCacheCoordinator(kv_cache_config, max_model_len,
                                         use_eagle, enable_caching,
                                         caching_hash_fn,
                                         enable_kv_cache_events)
    else:
        return HybridKVCacheCoordinator(kv_cache_config, max_model_len,
                                        use_eagle, enable_caching,
                                        caching_hash_fn,
                                        enable_kv_cache_events)
