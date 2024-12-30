from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, TypedDict
from vllm.utils import cdiv
from vllm.v1.core.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpecBase, SlidingWindowSpec
from vllm.v1.core.kv_cache_utils import BlockHashType, ComputedTokenRange, KVCacheBlock, ComputedTokens
from vllm.v1.utils import ConstantList


@dataclass
class MemoryPoolOperations:
    get_cached_block: Callable[[BlockHashType], Optional[KVCacheBlock]]


class CustomManager(ABC):
    block_size: int
    max_num_blocks_per_req: int

    def __init__(
        self,
        layer_spec: KVCacheSpecBase,
        memory_pool_operations: MemoryPoolOperations,
    ) -> None:
        self.block_size = layer_spec.block_size
        self.memory_pool_operations = memory_pool_operations

    @abstractmethod
    def get_computed_tokens(
        self, block_hashes: ConstantList[BlockHashType]
    ) -> Tuple[ComputedTokens, List[KVCacheBlock]]:
        raise NotImplementedError

    @abstractmethod
    def get_num_new_blocks(self, num_computed_tokens: int,
                           num_append_tokens: int,
                           num_allocated_blocks: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def remove_useless_blocks(self, block_table: List[KVCacheBlock],
                              num_computed_tokens: int, call_free: bool):
        # update block_table inplace
        raise NotImplementedError


class FullAttentionManager(CustomManager):

    def get_computed_tokens(
        self, block_hashes: ConstantList[BlockHashType]
    ) -> Tuple[ComputedTokens, List[KVCacheBlock]]:
        computed_blocks: List[KVCacheBlock] = []
        for block_hash in block_hashes:
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := self.memory_pool_operations.get_cached_block(
                    block_hash):
                computed_blocks.append(cached_block)
            else:
                break
        if len(computed_blocks) == 0:
            return [], []
        else:
            return [
                ComputedTokenRange(0,
                                   len(computed_blocks) * self.block_size)
            ], computed_blocks

    def get_num_new_blocks(self, num_computed_tokens: int,
                           num_append_tokens: int,
                           num_allocated_blocks: int) -> int:
        num_required_blocks = cdiv(num_computed_tokens + num_append_tokens,
                                   self.block_size)
        num_new_blocks = num_required_blocks - num_allocated_blocks
        return num_new_blocks

    def remove_useless_blocks(self, block_table: List[KVCacheBlock],
                              num_computed_tokens: int, call_free: bool):
        pass


class SlidingWindowManager(FullAttentionManager):
    # TODO: implement the sliding window manager
    pass


spec_manager_map = {
    FullAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager
}


def get_managers(
    kv_cache_config: KVCacheConfig,
    cached_block_hash_to_block: Dict[BlockHashType, Dict[int, KVCacheBlock]]
) -> Dict[str, CustomManager]:
    managers: Dict[str, CustomManager] = {}
    for group_id, layer_ids in kv_cache_config.groups.items():
        group_spec = kv_cache_config.kv_cache_spec[layer_ids[0]]
        manager_class = spec_manager_map[type(group_spec)]
        managers[group_id] = manager_class(group_spec,
                                           cached_block_hash_to_block)
    return managers
