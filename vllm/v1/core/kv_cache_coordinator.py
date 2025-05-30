# SPDX-License-Identifier: Apache-2.0
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

        self.type0_group_ids, self.type1_group_ids = (
            self.verify_support_find_longest_cache_hit())

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
        return [[] for _ in self.kv_cache_config.kv_cache_groups], 0
        # if len(self.kv_cache_config.kv_cache_groups) == 1:
        #     # Return the cache hit blocks for the only kv cache group.
        #     block_size = self.kv_cache_config.kv_cache_groups[
        #         0].kv_cache_spec.block_size
        #     hit_blocks = self.single_type_managers[0].find_longest_cache_hit(
        #         block_hashes_dict[block_size], max_length=max_cache_hit_length) # noqa
        #     if len(hit_blocks) > 0:
        #         self.computed_blocks[request_id] = [hit_blocks]
        #     return [hit_blocks], len(hit_blocks) * block_size

        # elif len(self.kv_cache_config.kv_cache_groups) > 1:
        #     # For simplicity, we assume the first manager is for full
        #     # attention layers, and the block_size of full attention layers
        #     # is divisible by other attention layers. This has been verified
        #     # in verify_support_find_longest_cache_hit().

        #     block_size_0 = self.single_type_managers[0].block_size
        #     block_size_1 = self.single_type_managers[1].block_size

        #     # First, find the longest cache hit for full attention.
        #     hit_blocks_full_attn = self.single_type_managers[
        #         0].find_longest_cache_hit(block_hashes_dict[block_size_0],
        #                                   max_length=max_cache_hit_length)
        #     hit_length = len(hit_blocks_full_attn) * block_size_0

        #     # Next, find the cache hit for the other attention WITHIN
        #     # the cache hit of full attention.
        #     hit_blocks_other_attn = self.single_type_managers[
        #         1].find_longest_cache_hit(block_hashes_dict[block_size_1],
        #                                   max_length=hit_length)
        #     hit_length = len(hit_blocks_other_attn) * block_size_1
        #     assert hit_length % block_size_0 == 0

        #     # Truncate the full attention cache hit to the length of the
        #     # cache hit of the other attention.
        #     del hit_blocks_full_attn[hit_length // block_size_0:]

        #     hit_blocks_two_mgr = [hit_blocks_full_attn, hit_blocks_other_attn]
        #     if hit_length > 0:
        #         self.computed_blocks[request_id] = hit_blocks_two_mgr
        #     return hit_blocks_two_mgr, hit_length

        # else:
        #     raise AssertionError("This line should be unreachable as "
        #                          "unsupported cases should be caught by "
        #                          "verify_support_find_longest_cache_hit()")

    def generate_group_manager_map(
            self) -> tuple[list[list[int]], list[tuple[int, int]]]:
        """
        Generate the mapping between kv cache groups and managers.

        Returns:
            manager_to_group: list[list[int]], the kv cache groups managed by
                each manager.
            group_to_manager: list[tuple[int, int]], the manager id and the
                index of the group in the manager for each kv cache group.
        """
        groups_by_type_id: dict[str, list[int]] = defaultdict(list)
        full_attention_type_ids: set[str] = set()
        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            groups_by_type_id[g.kv_cache_spec.type_id].append(i)
            if isinstance(g.kv_cache_spec, FullAttentionSpec):
                full_attention_type_ids.add(g.kv_cache_spec.type_id)

        manager_to_group = []
        for type_id in full_attention_type_ids:
            manager_to_group.append(groups_by_type_id[type_id])
        for type_id in groups_by_type_id.keys() - full_attention_type_ids:
            manager_to_group.append(groups_by_type_id[type_id])

        group_to_manager_dict = {
            group_id: (manager_id, group_id_in_manager)
            for manager_id, group_ids in enumerate(manager_to_group)
            for group_id_in_manager, group_id in enumerate(group_ids)
        }
        group_to_manager = [
            group_to_manager_dict[i]
            for i in range(len(self.kv_cache_config.kv_cache_groups))
        ]
        return manager_to_group, group_to_manager

    def verify_support_find_longest_cache_hit(
            self) -> tuple[list[int], list[int]]:
        """
        For simplicity, find_longest_cache_hit makes some assumptions on the
        model architecture instead of provides a general solution. This function
        checks if the assumptions hold.
        NOTE(Chen): Please open an issue to discuss if you need other cases.

        TODO: add more notes
        """
        if len(self.kv_cache_config.kv_cache_groups) == 1:
            return list(range(len(self.kv_cache_config.kv_cache_groups))), []
        else:
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

            type0_group_ids = groups_by_type_id[next(
                iter(full_attention_type_ids))]
            type1_group_ids = groups_by_type_id[next(
                iter(groups_by_type_id.keys() - full_attention_type_ids))]

            block_size_0 = self.kv_cache_config.kv_cache_groups[
                type0_group_ids[0]].kv_cache_spec.block_size
            block_size_1 = self.kv_cache_config.kv_cache_groups[
                type1_group_ids[0]].kv_cache_spec.block_size
            if block_size_1 % block_size_0 != 0:
                raise NotImplementedError(
                    "KVCacheCoordinator assumes the block_size of the full "
                    "attention layer is divisible by other layers now.")

            return type0_group_ids, type1_group_ids
