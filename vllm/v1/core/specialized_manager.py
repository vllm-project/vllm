# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional

from vllm.utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashType, GroupedKVCacheBlock, KVCacheBlock
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheSpec,
                                        SlidingWindowSpec)
from vllm.v1.request import Request
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
        kv_cache_manager_id: int,
        num_kv_cache_groups: int,
    ) -> None:
        """
        Initializes the SpecializedManager.
        Args:
            kv_cache_spec: The kv_cache_spec for this manager.
            block_pool: The block pool.
            kv_cache_manager_id: The id of the kv cache manager.
        """

        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.kv_cache_manager_id = kv_cache_manager_id
        self.num_kv_cache_groups = num_kv_cache_groups
        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: defaultdict[
            str, list[GroupedKVCacheBlock]] = defaultdict(list)

        # {req_id: The number of cached blocks for each kv cache group}
        # This is used to track the number of cached blocks for each request.
        # This is only used to track the RUNNING requests, we do not track the
        # data for reempted ones.
        self.num_cached_block: dict[str, int] = {}

    def get_num_needed_blocks(
            self, request_id: str, num_tokens: int,
            new_computed_block_list: list[GroupedKVCacheBlock]) -> int:
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = max(
            num_required_blocks - len(new_computed_block_list) -
            len(self.req_to_blocks[request_id]), 0)
        # If a computed block of a request is an eviction candidate (in the
        # free queue and ref_cnt == 0), it cannot be counted as a free block
        # when allocating this request. # TODO: update comment
        num_evictable_computed_blocks = sum(
            blks.blocks[0].ref_cnt == 0 for blks in new_computed_block_list)
        return ((num_new_blocks + num_evictable_computed_blocks) *
                self.num_kv_cache_groups)

    def allocate_new_blocks(self, request_id: str,
                            num_tokens: int) -> list[GroupedKVCacheBlock]:
        """
        return [group_id][block_of_that_group]
        """
        # TODO: group?
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = max(
            num_required_blocks - len(self.req_to_blocks[request_id]), 0)
        if num_new_blocks <= 0:
            return []
        else:
            flat_new_blocks = self.block_pool.get_new_blocks(
                num_new_blocks * self.num_kv_cache_groups)
            new_blocks = []
            for i in range(num_new_blocks):
                blocks = flat_new_blocks[i * self.num_kv_cache_groups:(i + 1) *
                                         self.num_kv_cache_groups]
                grouped_block = GroupedKVCacheBlock.from_kv_cache_blocks(
                    tuple(blocks))
                new_blocks.append(grouped_block)
            self.req_to_blocks[request_id].extend(new_blocks)
            return new_blocks

    def cache_blocks(self, request: Request,
                     new_computed_blocks: list[GroupedKVCacheBlock],
                     block_hashes: list[BlockHashType],
                     num_computed_tokens: int, num_tokens: int) -> None:
        # Use `new_computed_blocks` for a new request, and
        # `num_cached_block` for a running request.
        num_cached_blocks = self.num_cached_block.get(request.request_id,
                                                      len(new_computed_blocks))
        # Speculated tokens might be rejected in the future, so we does
        # not cache any speculated tokens. We only cache blocks with
        # generated (accepted) tokens.
        num_full_blocks_after_append = (num_computed_tokens + num_tokens - len(
            request.spec_token_ids)) // self.block_size

        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id],
            block_hashes=block_hashes,
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks_after_append,
            block_size=self.block_size,
            manager_id=self.kv_cache_manager_id,
        )

        self.num_cached_block[
            request.request_id] = num_full_blocks_after_append

    def free(self, request_id: str) -> None:
        # Default to [] in case a request is freed (aborted) before alloc.
        blocks = self.req_to_blocks.pop(request_id, None)
        if blocks is not None:
            self.block_pool.free_blocks(reversed(blocks))

        self.num_cached_block.pop(request_id, None)

    @abstractmethod
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHashType],
        computed_blocks: Optional[list[GroupedKVCacheBlock]] = None,
    ) -> list[GroupedKVCacheBlock]:
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
    def remove_skipped_blocks(self, request_id: str,
                              num_computed_tokens: int) -> None:
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
        computed_blocks: Optional[list[GroupedKVCacheBlock]] = None
    ) -> list[GroupedKVCacheBlock]:
        if computed_blocks is None:
            computed_blocks = []
            for block_hash in block_hashes:
                # block_hashes is a chain of block hashes. If a block hash is
                # not in the cached_block_hash_to_id, the following block hashes
                # are not computed yet for sure.
                if cached_block := self.block_pool.get_cached_block(
                        block_hash, self.kv_cache_manager_id):
                    computed_blocks.append(cached_block)
                else:
                    break
        else:
            assert len(computed_blocks) >= len(block_hashes)
            del computed_blocks[len(block_hashes):]
        return computed_blocks

    def remove_skipped_blocks(self, request_id: str,
                              num_computed_tokens: int) -> None:
        # No need to remove blocks for full attention.
        pass


class SlidingWindowManager(SpecializedManager):

    def __init__(self, kv_cache_spec: SlidingWindowSpec, block_pool: BlockPool,
                 kv_cache_manager_id: int, num_kv_cache_groups: int):
        super().__init__(kv_cache_spec, block_pool, kv_cache_manager_id,
                         num_kv_cache_groups)
        self.sliding_window = kv_cache_spec.sliding_window
        # The number of contiguous blocks needed for prefix cache hit.
        # -1 since the input token itself is also included in the window
        self.sliding_window_contiguous_blocks = cdiv(
            (kv_cache_spec.sliding_window - 1), self.block_size)
        self._null_block = block_pool.null_block

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHashType],
        computed_blocks: Optional[list[GroupedKVCacheBlock]] = None
    ) -> list[GroupedKVCacheBlock]:
        # TODO: reduce i by sliding_window_contiguous_blocks when cache miss, to
        # optimize the time complexity from O(len(block_hashes)) to
        # O(len(block_hashes) / sliding_window_contiguous_blocks +
        # sliding_window_contiguous_blocks),
        # which is good for low cache hit rate scenarios.
        if computed_blocks is None:
            num_contiguous_blocks = 0
            computed_blocks = [
                GroupedKVCacheBlock.from_kv_cache_blocks(
                    tuple([self._null_block] * self.num_kv_cache_groups))
                for _ in range(len(block_hashes))
            ]
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
                    block_hashes[i], self.kv_cache_manager_id):
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

    def remove_skipped_blocks(self, request_id: str,
                              num_computed_tokens: int) -> None:
        # Remove the blocks that are no longer be in the sliding window and
        # skipped during the attention computation.
        last_useful_token = num_computed_tokens - self.sliding_window + 1
        last_useful_block = last_useful_token // self.block_size
        blocks = self.req_to_blocks[request_id]

        removed_blocks: list[GroupedKVCacheBlock] = []
        for i in range(last_useful_block - 1, -1, -1):
            if blocks[i].blocks[0] == self._null_block:
                # If the block is already a null block, the blocks before it
                # should also have been set to null blocks by the previous calls
                # to this function.
                break
            removed_blocks.append(blocks[i])
            blocks[i] = GroupedKVCacheBlock.from_kv_cache_blocks(
                tuple([self._null_block] * self.num_kv_cache_groups))
        self.block_pool.free_blocks(removed_blocks)


spec_manager_map: dict[type[KVCacheSpec], type[SpecializedManager]] = {
    FullAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager,
}


def get_specialized_manager(kv_cache_spec: KVCacheSpec, block_pool: BlockPool,
                            kv_cache_manager_id: int,
                            num_kv_cache_groups: int) -> SpecializedManager:
    manager_class = spec_manager_map[type(kv_cache_spec)]
    manager = manager_class(kv_cache_spec, block_pool, kv_cache_manager_id,
                            num_kv_cache_groups)
    return manager
