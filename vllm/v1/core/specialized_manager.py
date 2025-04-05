# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Optional

from vllm.utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashType, KVCacheBlock
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheSpec,
                                        SlidingWindowSpec)
from vllm.v1.utils import ConstantList


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
        # for caching the intermediate states between multiple calls of
        # find_longest_cache_hit
        self.req_cached_blocks: dict[str, list[KVCacheBlock]] = {}

    def find_longest_cache_hit(
        self,
        request_id: str,
        block_hashes: list[BlockHashType],
    ) -> list[KVCacheBlock]:
        """
        Find the longest cache hit prefix of the blocks. If no cache hit is 
        found, return an empty list.

        Args:
            request_id: The request id.
            block_hashes: The block hashes of the request.
            return_const_list: Whether to return a ConstantList.
        """

        if req_cached_blocks := self.req_cached_blocks.pop(request_id, None):
            assert len(req_cached_blocks) >= len(block_hashes)

        req_cached_blocks = self._find_longest_cache_hit(
            block_hashes, req_cached_blocks)

        return req_cached_blocks

    def find_longest_cache_hit_multiple_calls(
        self,
        request_id: str,
        block_hashes: list[BlockHashType],
    ) -> ConstantList[KVCacheBlock]:
        req_cached_blocks = self.find_longest_cache_hit(
            request_id, block_hashes)
        self.req_cached_blocks[request_id] = req_cached_blocks
        return ConstantList(req_cached_blocks)

    @abstractmethod
    def _find_longest_cache_hit(
        self,
        block_hashes: list[BlockHashType],
        computed_blocks: Optional[list[KVCacheBlock]],
    ) -> list[KVCacheBlock]:
        """
        # TODO: update comment for multiple calls
        Get the longest cache hit prefix of the blocks. If no cache hit is 
        found, return an empty list. # TODO: add notes for computed_blocks
        will not be longer than block_hashes.

        Args:
            block_hashes: The block hashes of the request.
            computed_blocks: The cached blocks for the request returned from
                the previous call of this function.
        Returns:
            A list of cached blocks with skipped blocks replaced by null block.
            For example, sliding window manager should return a list like
            [NULL, NULL, KVCacheBlock(7), KVCacheBlock(8)] for block size 4 and 
            sliding window 8. 
        """

        raise NotImplementedError

    @abstractmethod
    def remove_skipped_blocks(self, blocks: list[KVCacheBlock],
                              num_computed_tokens: int) -> list[KVCacheBlock]:
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

    def _find_longest_cache_hit(
            self, block_hashes: list[BlockHashType],
            computed_blocks: Optional[list[KVCacheBlock]]
    ) -> list[KVCacheBlock]:
        if computed_blocks is None:
            computed_blocks = []
            for block_hash in block_hashes:
                # block_hashes is a chain of block hashes. If a block hash is
                # not in the cached_block_hash_to_id, the following block hashes
                # are not computed yet for sure.
                if cached_block := self.block_pool.get_cached_block(
                        block_hash):
                    computed_blocks.append(cached_block)
                else:
                    break
        else:
            assert len(computed_blocks) >= len(block_hashes)
            del computed_blocks[len(block_hashes):]
        return computed_blocks

    def remove_skipped_blocks(self, blocks: list[KVCacheBlock],
                              num_computed_tokens: int) -> list[KVCacheBlock]:
        # No need to remove blocks for full attention.
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

    def _find_longest_cache_hit(
            self, block_hashes: list[BlockHashType],
            computed_blocks: Optional[list[KVCacheBlock]]
    ) -> list[KVCacheBlock]:
        # TODO: reduce i by sliding_window_contiguous_blocks when cache miss, to
        # optimize the time complexity from O(len(block_hashes)) to
        # O(len(block_hashes) / sliding_window_contiguous_blocks +
        # sliding_window_contiguous_blocks),
        # which is good for low cache hit rate scenarios.
        if computed_blocks is None:
            num_contiguous_blocks = 0
            computed_blocks = [self._null_block] * len(block_hashes)
        else:
            if len(computed_blocks) == len(block_hashes):
                return computed_blocks
            # We are sure the last num_contiguous_blocks are not NULL and do
            # not need to check again.
            num_contiguous_blocks = max(
                self.sliding_window_contiguous_blocks -
                (len(computed_blocks) - len(block_hashes)), 0)
            del computed_blocks[len(block_hashes):]

        # Search from right to left and early stop when a match is found.
        for i in range(len(block_hashes) - num_contiguous_blocks - 1, -1, -1):
            if cached_block := self.block_pool.get_cached_block(
                    block_hashes[i]):
                computed_blocks[i] = cached_block
                num_contiguous_blocks += 1
                if (num_contiguous_blocks
                        >= self.sliding_window_contiguous_blocks):
                    # Trim the trailing blocks.
                    # E.g., [NULL, NULL, 8, 3, NULL, 9] -> [NULL, NULL, 8, 3]
                    # when sliding_window_contiguous_blocks=2.
                    del computed_blocks[i + num_contiguous_blocks:]
                    return computed_blocks
            else:
                num_contiguous_blocks = 0
        # The first `num_contiguous_blocks` is a cache hit even if
        # `num_contiguous_blocks < sliding_window_contiguous_blocks`.
        del computed_blocks[num_contiguous_blocks:]
        return computed_blocks

    def remove_skipped_blocks(self, blocks: list[KVCacheBlock],
                              num_computed_tokens: int) -> list[KVCacheBlock]:
        # Remove the blocks that are no longer be in the sliding window and
        # skipped during the attention computation.
        last_useful_token = num_computed_tokens - self.sliding_window + 1
        last_useful_block = last_useful_token // self.block_size

        removed_blocks: list[KVCacheBlock] = []
        for i in range(last_useful_block - 1, -1, -1):
            if blocks[i] == self._null_block:
                # If the block is already a null block, the blocks before it
                # should also have been set to null blocks by the previous calls
                # to this function.
                break
            removed_blocks.append(blocks[i])
            blocks[i] = self._null_block
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
