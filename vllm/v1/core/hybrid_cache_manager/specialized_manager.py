from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, TypedDict
from vllm.utils import cdiv
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec, SlidingWindowSpec
from vllm.v1.core.kv_cache_utils import BlockHashType, KVCacheBlock
from vllm.v1.core.hybrid_cache_manager.utils import ComputedTokenRange, ComputedTokens
from vllm.v1.utils import ConstantList


@dataclass
class MemoryPoolOperations:
    get_cached_block: Callable[[BlockHashType], Optional[KVCacheBlock]]
    get_null_block: Callable[[], KVCacheBlock]


class SpecializedManager(ABC):
    block_size: int
    max_num_blocks_per_req: int

    def __init__(
        self,
        layer_spec: KVCacheSpec,
        memory_pool_operations: MemoryPoolOperations,
    ) -> None:
        self.block_size = layer_spec.block_size
        self.memory_pool_operations = memory_pool_operations

    @abstractmethod
    def get_computed_blocks_and_tokens(
        self, block_hashes: ConstantList[BlockHashType]
    ) -> Tuple[List[KVCacheBlock], ComputedTokens]:
        raise NotImplementedError

    @abstractmethod
    def get_num_new_blocks(self, num_computed_tokens: int,
                           num_append_tokens: int,
                           num_allocated_blocks: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def remove_dropped_blocks(self, block_table: List[KVCacheBlock],
                              num_computed_tokens: int):
        # update block_table inplace
        raise NotImplementedError


class FullAttentionManager(SpecializedManager):

    def get_computed_blocks_and_tokens(
        self, block_hashes: ConstantList[BlockHashType]
    ) -> Tuple[List[KVCacheBlock], ComputedTokens]:
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

    def remove_dropped_blocks(self, block_table: List[KVCacheBlock],
                              num_computed_tokens: int) -> List[KVCacheBlock]:
        return []


class SlidingWindowManager(FullAttentionManager):

    def __init__(self, layer_spec: SlidingWindowSpec,
                 memory_pool_operations: MemoryPoolOperations):
        super().__init__(layer_spec, memory_pool_operations)
        # +1 due to not aligned
        self.num_block_sliding_window = cdiv(layer_spec.sliding_window,
                                             self.block_size) + 1
        self._null_block = memory_pool_operations.get_null_block()

    def get_computed_blocks_and_tokens(
        self, block_hashes: ConstantList[BlockHashType]
    ) -> Tuple[List[KVCacheBlock], ComputedTokens]:
        # TODO: check the hit every num_block_sliding_window blocks, to optimize
        # the time complexity from O(num_block) to
        # O(num_block / num_block_sliding_window) + O(num_computed_block),
        # which is good for low cache hit rate senarios.
        start = 0
        ranges = []
        computed_blocks: List[KVCacheBlock] = []

        for i, block_hash in enumerate(block_hashes):
            if cached_block := self.memory_pool_operations.get_cached_block(
                    block_hash):
                computed_blocks.append(cached_block)
            else:
                if start == 0:
                    ranges.append(
                        ComputedTokenRange(start * self.block_size,
                                           i * self.block_size))
                elif i - start >= self.num_block_sliding_window:
                    ranges.append((ComputedTokenRange(
                        (start + self.num_block_sliding_window) *
                        self.block_size, i * self.block_size)))
                computed_blocks.append(
                    self.memory_pool_operations.get_null_block())
                start = i + 1
        return ranges, computed_blocks

    def remove_dropped_blocks(self, block_table: List[KVCacheBlock],
                              num_computed_tokens: int) -> List[KVCacheBlock]:
        num_block_should_free = cdiv(num_computed_tokens, self.block_size) - \
                self.num_block_sliding_window
        removed_blocks = deque()
        for i in range(num_block_should_free - 1, -1, -1):
            if block_table[i] == self._null_block:
                break
            removed_blocks.appendleft(block_table[i])
            block_table[i] = self._null_block
        return removed_blocks


spec_manager_map = {
    FullAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager
}


def get_managers(
        kv_cache_config: KVCacheConfig,
        memory_pool_operations: MemoryPoolOperations
) -> List[SpecializedManager]:
    managers: List[SpecializedManager] = []
    for g in kv_cache_config.groups:
        manager_class = spec_manager_map[type(g.kv_cache_spec)]
        manager = manager_class(g.kv_cache_spec, memory_pool_operations)
        managers.append(manager)
    return managers
