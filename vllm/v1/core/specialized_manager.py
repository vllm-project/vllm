# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from itertools import chain

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (BlockHashType, KVCacheBlock,
                                         PrefixLengthRange)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheSpec,
                                        SlidingWindowSpec)


class SpecializedManager(ABC):
    """
    An abstract base class for specialized managers that handle the kv
    cache management logic of different attention layers.
    """

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
    ) -> None:
        """
        Initializes the SpecializedManager.
        Args:
            kv_cache_spec: The kv_cache_spec for this manager.
            block_pool: The block pool.
        """

        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool

    @abstractmethod
    def get_possible_cached_prefix(
        self, block_hashes: list[BlockHashType]
    ) -> tuple[list[PrefixLengthRange], list[KVCacheBlock]]:
        """
        Get the possible cached prefixes of a request based on its block hashes.
        If no cached prefixes are found, returns a tuple with a prefix length 
        range of [0, 0] and an empty list of blocks.

        Args:
            block_hashes: The block hashes of the request.
        Returns:
            A tuple containing:
                - A list of all possible cached prefix lengths.
                - A list of cached blocks for each block hash. Use the null 
                block for blocks that are not cached. Can skip the last blocks
                that are not cached.
        """

        raise NotImplementedError

    @abstractmethod
    def remove_useless_blocks(self, block_table: list[KVCacheBlock],
                              num_computed_tokens: int) -> list[KVCacheBlock]:
        """
        Remove the blocks that are no longer needed from `block_table`. The 
        removed blocks should be replaced by null_blocks. Return the removed 
        blocks in eviction order, where the first returned block should be 
        evicted first.

        Args:
            block_table: The block table to be updated.
            num_computed_tokens: The number of tokens that have been computed.
        Returns:
            The removed blocks in eviction order.
        """
        raise NotImplementedError


class FullAttentionManager(SpecializedManager):

    def get_possible_cached_prefix(
        self, block_hashes: list[BlockHashType]
    ) -> tuple[list[PrefixLengthRange], list[KVCacheBlock]]:
        computed_blocks: list[KVCacheBlock] = []
        for block_hash in block_hashes:
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := self.block_pool.get_cached_block(block_hash):
                computed_blocks.append(cached_block)
            else:
                break
        return [PrefixLengthRange(0,
                                  len(computed_blocks) * self.block_size)
                ], computed_blocks

    def remove_useless_blocks(self, block_table: list[KVCacheBlock],
                              num_computed_tokens: int) -> list[KVCacheBlock]:
        # No need to remove blocks for full attention.
        return []


class SlidingWindowManager(SpecializedManager):

    def __init__(self, kv_cache_spec: SlidingWindowSpec,
                 block_pool: BlockPool):
        super().__init__(kv_cache_spec, block_pool)
        self.sliding_window = kv_cache_spec.sliding_window
        self.block_pool.init_real_null_block()
        self._null_block = block_pool.get_null_block()

    def get_possible_cached_prefix(
        self, block_hashes: list[BlockHashType]
    ) -> tuple[list[PrefixLengthRange], list[KVCacheBlock]]:
        # TODO: check the hit every num_block_sliding_window blocks, to optimize
        # the time complexity from O(num_block) to
        # O(num_block / num_block_sliding_window) + O(num_computed_block),
        # which is good for low cache hit rate scenarios.
        start = 0
        ranges = []
        computed_blocks: list[KVCacheBlock] = []

        dummy_block_hash = BlockHashType(-1, (), -1)
        # Add a dummy block hash to support the case that the last block is
        # cached.
        for i, block_hash in enumerate(chain(block_hashes,
                                             [dummy_block_hash])):
            if cached_block := self.block_pool.get_cached_block(block_hash):
                computed_blocks.append(cached_block)
            else:
                if start == 0:
                    # All tokens between [0, i * block_size] are cached.
                    # All of them are possible cached prefix.
                    ranges.append(PrefixLengthRange(0, i * self.block_size))
                elif (i - start) * self.block_size >= self.sliding_window:
                    # All tokens with index between [start * block_size,
                    # i * block_size) are cached. These tokens except the
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

    def remove_useless_blocks(self, block_table: list[KVCacheBlock],
                              num_computed_tokens: int) -> list[KVCacheBlock]:
        # Remove the blocks that are no longer be in the sliding window.
        last_useful_token = num_computed_tokens - self.sliding_window
        last_useful_block = last_useful_token // self.block_size

        removed_blocks: list[KVCacheBlock] = []
        for i in range(last_useful_block - 1, -1, -1):
            if block_table[i] == self._null_block:
                # If the block is already a null block, the blocks before it
                # should also have been set to null blocks by the previous calls
                # to this function.
                break
            removed_blocks.append(block_table[i])
            block_table[i] = self._null_block
        return removed_blocks


spec_manager_map: dict[type[KVCacheSpec], type[SpecializedManager]] = {
    FullAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager
}


def get_specialized_manager(kv_cache_spec: KVCacheSpec,
                            block_pool: BlockPool) -> SpecializedManager:
    manager_class = spec_manager_map[type(kv_cache_spec)]
    manager = manager_class(kv_cache_spec, block_pool)
    return manager
