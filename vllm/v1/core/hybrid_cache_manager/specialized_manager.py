from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from itertools import chain
from typing import Callable, Deque, Dict, List, Optional, Tuple, TypedDict
from vllm.utils import cdiv
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec, SlidingWindowSpec
from vllm.v1.core.kv_cache_utils import BlockHashType, KVCacheBlock
from vllm.v1.core.hybrid_cache_manager.utils import PrefixLength, PrefixLengthRange
from vllm.v1.utils import ConstantList


@dataclass
class BlockPoolOperations:
    get_cached_block: Callable[[BlockHashType], Optional[KVCacheBlock]]
    get_null_block: Callable[[], KVCacheBlock]


class SpecializedManager(ABC):
    """
    An abstract base class for specialized managers that handle the kv
    cache management logic of different attention layers.
    """
    block_size: int
    max_num_blocks_per_req: int

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool_operations: BlockPoolOperations,
    ) -> None:
        """
        Initializes the SpecializedManager.

        Args:
            kv_cache_spec: The kv_cache_spec for this manager.
            block_pool_operations: Operations to interact with the block pool.

        Returns:
            None
        """

        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool_operations = block_pool_operations

    @abstractmethod
    def get_possible_cached_prefix(
        self, block_hashes: ConstantList[BlockHashType]
    ) -> Tuple[PrefixLength, List[KVCacheBlock]]:
        """
        Get the possible cached prefixes of a request based on its block hashes.
        If no cached prefixes are found, returns a tuple with a prefix length 
        range of [0, 0] and an empty list of blocks.

        Args:
            block_hashes: The block hashes of the request.

        Returns:
            A tuple containing:
                - A list of all possible cached prefix lengths.
                - The computed blocks that are cached.
        """

        raise NotImplementedError

    @abstractmethod
    def get_num_new_blocks(self, num_computed_tokens: int,
                           num_append_tokens: int,
                           num_allocated_blocks: int) -> int:
        """
        Calculate the number of new blocks needed by this manager.

        Args:
            num_computed_tokens: The number of tokens that have been computed.
            num_append_tokens: The number of tokens that need to be appended.
            num_allocated_blocks: The number of blocks that have already been 
            allocated.

        Returns:
            int: The number of new blocks needed.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_useless_blocks(self, block_table: List[KVCacheBlock],
                              num_computed_tokens: int) -> List[KVCacheBlock]:
        """
        Update the `block_table` in place to remove blocks that are no longer 
        needed. Replace the removed blocks with null_block and returns the 
        removed blocks. 
        The removed blocks should be in the order of the
        priority to be evicted, where the first block should have the highest
        priority.
        
        Args:
            block_table: The block table to be updated.
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            List[KVCacheBlock]: The removed blocks.
        """
        raise NotImplementedError


class FullAttentionManager(SpecializedManager):

    def get_possible_cached_prefix(
        self, block_hashes: ConstantList[BlockHashType]
    ) -> Tuple[List[PrefixLengthRange], List[KVCacheBlock]]:
        computed_blocks: List[KVCacheBlock] = []
        for block_hash in block_hashes:
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := self.block_pool_operations.get_cached_block(
                    block_hash):
                computed_blocks.append(cached_block)
            else:
                break
        return [PrefixLengthRange(0,
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
                              num_computed_tokens: int) -> List[KVCacheBlock]:
        return []


class SlidingWindowManager(FullAttentionManager):

    def __init__(self, kv_cache_spec: SlidingWindowSpec,
                 block_pool_operations: BlockPoolOperations):
        super().__init__(kv_cache_spec, block_pool_operations)
        self.sliding_window = kv_cache_spec.sliding_window
        self._null_block = block_pool_operations.get_null_block()

    def get_possible_cached_prefix(
        self, block_hashes: ConstantList[BlockHashType]
    ) -> Tuple[List[PrefixLengthRange], List[KVCacheBlock]]:
        # TODO: check the hit every num_block_sliding_window blocks, to optimize
        # the time complexity from O(num_block) to
        # O(num_block / num_block_sliding_window) + O(num_computed_block),
        # which is good for low cache hit rate senarios.
        start = 0
        ranges = []
        computed_blocks: List[KVCacheBlock] = []

        dummy_block_hash = BlockHashType(-1, (), -1)
        # Add a dummy block hash to support the case that the last block is
        # cached.
        for i, block_hash in enumerate(chain(block_hashes,
                                             [dummy_block_hash])):
            if cached_block := self.block_pool_operations.get_cached_block(
                    block_hash):
                computed_blocks.append(cached_block)
            else:
                if start == 0:
                    # All tokens between [0, i * block_size] are cached.
                    # All of them are possible cached prefix.
                    ranges.append(PrefixLengthRange(0, i * self.block_size))
                elif (i - start) * self.block_size >= self.sliding_window:
                    # All tokens between [start * block_size,
                    # i * block_size)] are cached. These tokens except the
                    # first `self.sliding_window - 1` ones are possible cached
                    # prefix.
                    first_cached_token = start * self.block_size
                    # should be first_cached_token + self.sliding_window - 1 + 1
                    # +1 is for converting the token index to the prefix length.
                    first_possible_length = first_cached_token + \
                        self.sliding_window
                    ranges.append(
                        PrefixLengthRange(first_possible_length,
                                          i * self.block_size))
                computed_blocks.append(self._null_block)
                start = i + 1
        computed_blocks = computed_blocks[:-1]  # remove the dummy block
        return ranges, computed_blocks

    def remove_useless_blocks(self, block_table: List[KVCacheBlock],
                              num_computed_tokens: int) -> List[KVCacheBlock]:
        # Remove the blocks that are no longer be in the sliding window.
        last_useful_token = num_computed_tokens - self.sliding_window
        last_useful_block = last_useful_token // self.block_size

        removed_blocks: List[KVCacheBlock] = []
        for i in range(last_useful_block - 1, -1, -1):
            if block_table[i] == self._null_block:
                # If the block is already a null block, the blocks before it
                # should also be null blocks.
                break
            removed_blocks.append(block_table[i])
            block_table[i] = self._null_block
        return removed_blocks


spec_manager_map: Dict[KVCacheSpec, SpecializedManager] = {
    FullAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager
}


def get_managers(
        kv_cache_config: KVCacheConfig,
        block_pool_operations: BlockPoolOperations
) -> List[SpecializedManager]:
    managers: List[SpecializedManager] = []
    for g in kv_cache_config.groups:
        manager_class = spec_manager_map[type(g.kv_cache_spec)]
        manager = manager_class(g.kv_cache_spec, block_pool_operations)
        managers.append(manager)
    return managers
