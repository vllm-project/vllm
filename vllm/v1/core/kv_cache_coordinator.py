# SPDX-License-Identifier: Apache-2.0
from typing import Callable

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashType, GroupedKVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    SingleTypeKVCacheManager, get_manager_for_kv_cache_spec)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request


class KVCacheCoordinator:
    """
    Coordinator the KV cache of different KV cache groups.
    # TODO: docstring for this class
    """

    def __init__(self, kv_cache_config: KVCacheConfig, block_pool: BlockPool,
                 max_model_len: int, use_eagle: bool,
                 caching_hash_fn: Callable):
        self.block_pool = block_pool
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len

        # the kv cache groups managed by the each manager
        # manager_id -> list[kv_cache_group_id]
        self.manager_to_group, self.group_to_manager = (
            self.generate_group_manager_map())
        self.num_single_type_manager = len(self.manager_to_group)

        self.single_type_managers: list[SingleTypeKVCacheManager] = []
        for i in range(len(self.manager_to_group)):
            group_ids = self.manager_to_group[i]
            kv_cache_spec = kv_cache_config.kv_cache_groups[
                group_ids[0]].kv_cache_spec
            self.single_type_managers.append(
                get_manager_for_kv_cache_spec(
                    kv_cache_spec=kv_cache_spec,
                    block_pool=self.block_pool,
                    use_eagle=use_eagle,
                    num_kv_cache_groups=len(self.manager_to_group[i]),
                    manager_id=i,
                    caching_hash_fn=caching_hash_fn,
                ))

    def find_longest_cache_hit(
        self, request: Request, block_hashes_dict: dict[int,
                                                        list[BlockHashType]],
        max_cache_hit_length: int
    ) -> tuple[list[list[GroupedKVCacheBlock]], int]:
        """Find the longest cache hit for each kv cache group.
        TODO: add more notes
        """
        # TODO: implement this
        raise NotImplementedError("Not implemented")

    def remove_skipped_blocks(self, request_id: str,
                              num_computed_tokens: int) -> None:
        for manager in self.single_type_managers:
            manager.remove_skipped_blocks(request_id, num_computed_tokens)

    def get_num_blocks_to_allocate(
            self, request_id: str, num_tokens: int,
            new_computed_blocks: list[list[GroupedKVCacheBlock]]) -> int:
        num_blocks_to_allocate = 0
        for i, manager in enumerate(self.single_type_managers):
            num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                request_id, num_tokens, new_computed_blocks[i])
        return num_blocks_to_allocate

    def save_new_computed_blocks(
            self, request_id: str,
            new_computed_blocks: list[list[GroupedKVCacheBlock]]) -> None:
        for i, manager in enumerate(self.single_type_managers):
            manager.save_new_computed_blocks(request_id,
                                             new_computed_blocks[i])

    def cache_blocks(self, request: Request,
                     block_hashes: dict[int, list[BlockHashType]],
                     num_computed_tokens: int) -> None:
        for manager in self.single_type_managers:
            manager.cache_blocks(request, block_hashes[manager.block_size],
                                 num_computed_tokens)

    def allocate_new_blocks(
            self, request_id: str,
            num_tokens: int) -> list[list[GroupedKVCacheBlock]]:
        new_blocks = []
        for manager in self.single_type_managers:
            new_blocks.append(
                manager.allocate_new_blocks(request_id, num_tokens))
        return new_blocks

    def free(self, request_id: str) -> None:
        for manager in self.single_type_managers:
            manager.free(request_id)

    def get_num_common_prefix_blocks(
        self,
        request_id: str,
        num_running_requests: int,
    ) -> list[int]:
        num_blocks_per_manager = [
            manager.get_num_common_prefix_blocks(request_id,
                                                 num_running_requests)
            for manager in self.single_type_managers
        ]
        num_blocks_per_group = [
            num_blocks_per_manager[manager_id]
            for manager_id, _ in self.group_to_manager
        ]
        return num_blocks_per_group

    def generate_group_manager_map(
            self) -> tuple[list[list[int]], list[tuple[int, int]]]:
        # TODO: refactor this function to ensure full attention is the first
        # group
        type_ids = [
            g.kv_cache_spec.type_id
            for g in self.kv_cache_config.kv_cache_groups
        ]
        assert sorted(type_ids) == type_ids, "type_ids must be sorted"
        manager_to_group: list[list[int]] = []
        for i, type_id in enumerate(type_ids):
            if i == 0:
                manager_to_group.append([i])
            else:
                if type_id == type_ids[i - 1]:
                    manager_to_group[-1].append(i)
                else:
                    manager_to_group.append([i])
        print("manager_to_group", manager_to_group)
        group_to_manager = [(i, j) for i in range(len(manager_to_group))
                            for j in range(len(manager_to_group[i]))]
        print("group_to_manager", group_to_manager)
        return manager_to_group, group_to_manager
