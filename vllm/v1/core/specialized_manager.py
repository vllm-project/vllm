# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Sequence

from vllm.utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashType, KVCacheBlock
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
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHashType],
        group_ids: Sequence[int],
    ) -> list[dict[int, KVCacheBlock]]:
        """
        Get the longest cache hit prefix of the blocks. If no cache hit is 
        found, return an empty list.

        Args:
            block_hashes: The block hashes of the request.
        Returns:
            A list of cached blocks with skipped blocks replaced by null block.
            For example, sliding window manager should return a list like
            [NULL, NULL, KVCacheBlock(7), KVCacheBlock(8)] for block size 4 and 
            sliding window 8. 
        """

        raise NotImplementedError

    @abstractmethod
    def remove_skipped_blocks(
        self,
        blocks: list[dict[int, KVCacheBlock]],
        group_ids: Sequence[int],
        num_computed_tokens: int,
    ) -> list[dict[int, KVCacheBlock]]:
        """
        Remove the blocks that are no longer needed from `blocks`. The removed 
        blocks should be replaced by null_block. Return the removed blocks in 
        eviction order, where the first returned block should be evicted first.
        Don't free the removed blocks in this function.

        Args:
            blocks: The list of blocks to be updated.
            num_computed_tokens: The number of tokens that have been computed.
        Returns:
            The removed blocks in eviction order.
        """
        raise NotImplementedError


class FullAttentionManager(SpecializedManager):

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHashType],
        group_ids: Sequence[int],
    ) -> list[dict[int, KVCacheBlock]]:
        computed_blocks: list[KVCacheBlock] = []
        for block_hash in block_hashes:
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            cached_blocks = self.block_pool.get_cached_block(block_hash)
            if cached_blocks is None:
                break
            if all(group_id in cached_blocks for group_id in group_ids):
                computed_blocks.append(cached_blocks)
            else:
                break
        return computed_blocks

    def remove_skipped_blocks(
        self,
        blocks: list[dict[int, KVCacheBlock]],
        group_ids: Sequence[int],
        num_computed_tokens: int,
    ) -> list[dict[int, KVCacheBlock]]:
        # Full attention skips no blocks.
        return []


class SlidingWindowManager(SpecializedManager):

    def __init__(self, kv_cache_spec: SlidingWindowSpec,
                 block_pool: BlockPool):
        super().__init__(kv_cache_spec, block_pool)
        self.sliding_window = kv_cache_spec.sliding_window
        # The number of contiguous blocks needed for prefix cache hit.
        # -1 since the input token itself is also included in the window
        self.sliding_window_contiguous_blocks = cdiv(
            (kv_cache_spec.sliding_window - 1), self.block_size)
        self._null_block = block_pool.null_block

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHashType],
        group_ids: Sequence[int],
    ) -> list[dict[int, KVCacheBlock]]:
        # TODO: reduce i by sliding_window_contiguous_blocks when cache miss, to
        # optimize the time complexity from O(len(block_hashes)) to
        # O(len(block_hashes) / sliding_window_contiguous_blocks +
        # sliding_window_contiguous_blocks),
        # which is good for low cache hit rate scenarios.
        null_block = {group_id: self._null_block for group_id in group_ids}
        computed_blocks = [null_block] * len(block_hashes)
        num_contiguous_blocks = 0

        # Search from right to left and early stop when a match is found.
        for i in range(len(block_hashes) - 1, -1, -1):
            block_hash = block_hashes[i]
            cached_blocks = self.block_pool.get_cached_block(block_hash)
            if (cached_blocks is None or
                any(group_id not in cached_blocks for group_id in group_ids)):
                num_contiguous_blocks = 0
                continue

            computed_blocks[i] = cached_blocks
            num_contiguous_blocks += 1
            if (num_contiguous_blocks
                    >= self.sliding_window_contiguous_blocks):
                # Trim the trailing blocks.
                # E.g., [NULL, NULL, 8, 3, NULL, 9] -> [NULL, NULL, 8, 3]
                # when sliding_window_contiguous_blocks=2.
                del computed_blocks[i + num_contiguous_blocks:]
                return computed_blocks

        # The first `num_contiguous_blocks` is a cache hit even if
        # `num_contiguous_blocks < sliding_window_contiguous_blocks`.
        del computed_blocks[num_contiguous_blocks:]
        return computed_blocks

    def remove_skipped_blocks(
        self,
        blocks: list[dict[int, KVCacheBlock]],
        group_ids: Sequence[int],
        num_computed_tokens: int,
    ) -> list[dict[int, KVCacheBlock]]:
        # Remove the blocks that are no longer be in the sliding window and
        # skipped during the attention computation.
        last_useful_token = num_computed_tokens - self.sliding_window + 1
        last_useful_block = last_useful_token // self.block_size

        removed_blocks: list[dict[int, KVCacheBlock]] = []
        for i in range(last_useful_block - 1, -1, -1):
            removed_block: dict[int, KVCacheBlock] = {}
            is_null = False
            for group_id in group_ids:
                block = blocks[i][group_id]
                if block == self._null_block:
                    # If the block is already a null block, the blocks before it
                    # should also have been set to null blocks by the previous
                    # calls to this function.
                    is_null = True
                    break
                removed_block[group_id] = block
                blocks[i][group_id] = self._null_block
            if is_null:
                break
            removed_blocks.append(removed_block)
        return removed_blocks


spec_manager_map: dict[type[KVCacheSpec], type[SpecializedManager]] = {
    FullAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager,
}


def get_specialized_manager(kv_cache_spec: KVCacheSpec,
                            block_pool: BlockPool) -> SpecializedManager:
    manager_class = spec_manager_map[type(kv_cache_spec)]
    manager = manager_class(kv_cache_spec, block_pool)
    return manager
