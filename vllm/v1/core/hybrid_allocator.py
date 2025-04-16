# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Optional

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashType, KVCacheBlock
from vllm.v1.core.specialized_manager import (FullAttentionAllocator,
                                              SlidingWindowAllocator,
                                              SpecializedAllocator)
from vllm.v1.request import Request


class HybridMemoryAllocator(ABC):

    def __init__(self, block_pool: BlockPool):
        self.block_pool = block_pool
        # req_id -> group_id -> block_hashes
        # This is to avoid recomputing the block hashes for each call of
        # `get_block_hashes`.
        # NOTE: These entries must be freed when requests are finished
        # to prevent memory leaks.
        self.req_to_block_hashes: dict[str, list[list[BlockHashType]]] = {}

    def get_block_hashes(
        self,
        request: Request,
        hash_fn: Any,
    ) -> list[list[BlockHashType]]:
        # The block hashes for the request may already be computed
        # if the scheduler has tried to schedule the request before.
        block_hashes = self.req_to_block_hashes.get(request.request_id)
        if block_hashes is None:
            block_hashes = self._get_block_hashes(request, hash_fn)
            self.req_to_block_hashes[request.request_id] = block_hashes
        return block_hashes

    @abstractmethod
    def _get_block_hashes(
        self,
        request: Request,
        hash_fn: Any,
    ) -> list[list[BlockHashType]]:
        raise NotImplementedError

    @abstractmethod
    def find_longest_cache_hit(
        self,
        block_hashes: list[list[BlockHashType]],
        num_tokens: int,
    ) -> tuple[list[list[KVCacheBlock]], int]:
        raise NotImplementedError

    @abstractmethod
    def remove_skipped_blocks(
        self,
        blocks: list[list[KVCacheBlock]],
        num_computed_tokens: int,
    ) -> Iterable[KVCacheBlock]:
        raise NotImplementedError

    def allocate_blocks(
        self,
        total_num_tokens: int,
        num_computed_tokens: int,
        new_computed_blocks: list[list[KVCacheBlock]],
        allocated_blocks: list[list[KVCacheBlock]],
    ) -> Optional[list[list[KVCacheBlock]]]:
        num_new_blocks = self._get_num_new_blocks(
            total_num_tokens,
            num_computed_tokens,
            new_computed_blocks,
            allocated_blocks,
        )
        total_num_new_blocks = sum(num_new_blocks)
        if total_num_new_blocks <= 0:
            # No new block is needed.
            return []
        flattened_new_computed_blocks = sum(new_computed_blocks, [])

        # If a computed block of a request is an eviction candidate (in the
        # free queue and ref_cnt == 0), it cannot be counted as a free block
        # when allocating this request.
        num_evictable_computed_blocks = sum(
            1 for blk in flattened_new_computed_blocks if blk.ref_cnt == 0)
        if (total_num_new_blocks > self.block_pool.get_num_free_blocks() -
                num_evictable_computed_blocks):
            # Cannot allocate new blocks
            return None

        # Touch the computed blocks to make sure they won't be evicted.
        if flattened_new_computed_blocks:
            self.block_pool.touch(flattened_new_computed_blocks)

        total_new_blocks = self.block_pool.get_new_blocks(total_num_new_blocks)
        new_blocks: list[list[KVCacheBlock]] = []
        start = 0
        for n in num_new_blocks:
            end = start + n
            new_blocks.append(total_new_blocks[start:end])
            start = end
        return new_blocks

    @abstractmethod
    def _get_num_new_blocks(
        self,
        total_num_tokens: int,
        num_computed_tokens: int,
        new_computed_blocks: list[list[KVCacheBlock]],
        allocated_blocks: list[list[KVCacheBlock]],
    ) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def cache_blocks(
        self,
        request: Request,
        blocks: list[list[KVCacheBlock]],
        num_computed_tokens: int,
        num_new_tokens: int,
        num_cached_blocks: list[int],
        hash_fn: Any,
    ) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def sort_by_eviction_order(
        self,
        blocks: list[list[KVCacheBlock]],
    ) -> Iterable[KVCacheBlock]:
        raise NotImplementedError


class SingleMemoryAllocator(HybridMemoryAllocator):
    """Memory allocator for a single attention type.

    For example, models with full attention only (e.g., Llama 3, DeepSeek) and
    models with sliding window attention only (e.g., an old version of Mistral)
    use this allocator.
    """

    def __init__(
        self,
        block_pool: BlockPool,
        allocator: SpecializedAllocator,
    ):
        super().__init__(block_pool)
        self.allocator = allocator
        self.block_size = allocator.block_size
        self.group_ids = (0, )

    def _get_block_hashes(
        self,
        request: Request,
        hash_fn: Any,
    ) -> list[list[BlockHashType]]:
        return [self.allocator.get_block_hashes(request, hash_fn)]

    def find_longest_cache_hit(
        self,
        block_hashes: list[list[BlockHashType]],
        num_tokens: int,
    ) -> tuple[list[list[KVCacheBlock]], int]:
        block_hashes = block_hashes[0]
        if len(block_hashes) * self.block_size == num_tokens:
            block_hashes = block_hashes[:-1]
        return self.allocator.find_longest_cache_hit(block_hashes,
                                                     self.group_ids)

    def remove_skipped_blocks(
        self,
        blocks: dict[int, list[KVCacheBlock]],
        num_computed_tokens: int,
    ) -> Iterable[KVCacheBlock]:
        return self.allocator.remove_skipped_blocks(blocks, self.group_ids,
                                                    num_computed_tokens)

    def _get_num_new_blocks(
        self,
        total_num_tokens: int,
        num_computed_tokens: int,
        new_computed_blocks: list[list[KVCacheBlock]],
        allocated_blocks: list[list[KVCacheBlock]],
    ) -> list[int]:
        num_new_blocks = self.allocator.get_num_new_blocks(
            total_num_tokens,
            num_computed_tokens,
            new_computed_blocks,
            allocated_blocks,
            self.group_ids,
        )
        return [num_new_blocks[0]]

    def cache_blocks(
        self,
        request: Request,
        blocks: list[list[KVCacheBlock]],
        num_computed_tokens: int,
        num_new_tokens: int,
        num_cached_blocks: list[int],
        hash_fn: Any,
    ) -> list[int]:
        return self.allocator.cache_blocks(
            request,
            blocks[0],
            num_computed_tokens,
            num_new_tokens,
            num_cached_blocks[0],
            hash_fn,
        )

    def sort_by_eviction_order(
        self,
        blocks: list[list[KVCacheBlock]],
    ) -> Iterable[KVCacheBlock]:
        return self.allocator.sort_by_eviction_order(blocks[0])


class FullAndSwaMemoryAllocator(HybridMemoryAllocator):
    """Memory allocator for full and sliding window attention.

    For example, models like Gemma 2 (1:1 full/swa) and Gemma 3 (1:5 full/swa)
    use this allocator.
    """

    def __init__(
        self,
        block_pool: BlockPool,
        full_attn_allocator: FullAttentionAllocator,
        full_attn_group_ids: tuple[int, ...],
        swa_allocator: SlidingWindowAllocator,
        swa_group_ids: tuple[int, ...],
    ):
        super().__init__(block_pool)
        self.full_attn_allocator = full_attn_allocator
        self.full_attn_group_ids = full_attn_group_ids
        self.swa_allocator = swa_allocator
        self.swa_group_ids = swa_group_ids

        self.all_group_ids = sorted(full_attn_group_ids + swa_group_ids)
        self.num_groups = len(self.all_group_ids)
        self.block_size = full_attn_allocator.block_size
        if self.block_size != swa_allocator.block_size:
            raise ValueError(
                f"The block size of full attention ({self.block_size}) and "
                f"sliding window attention ({swa_allocator.block_size}) must be "
                "the same.")

    def _get_block_hashes(
        self,
        request: Request,
        hash_fn: Any,
    ) -> list[list[BlockHashType]]:
        # The full attention and sliding window attention use the same block
        # size.
        block_hashes = self.full_attn_allocator.get_block_hashes(
            request, hash_fn)
        # TODO(woosuk): Optimize this.
        return [block_hashes] * self.num_groups

    def find_longest_cache_hit(
        self,
        block_hashes: list[list[BlockHashType]],
        num_tokens: int,
    ) -> tuple[list[list[KVCacheBlock]], int]:
        # Because the full attention and sliding window attention use the same
        # block size, we can just use the block hashes for any group.
        block_hashes = block_hashes[0]
        if len(block_hashes) * self.block_size == num_tokens:
            block_hashes = block_hashes[:-1]

        # First, find the longest cache hit for full attention.
        full_attn_blocks, num_full_attn_tokens = (
            self.full_attn_allocator.find_longest_cache_hit(
                block_hashes, self.full_attn_group_ids))
        num_full_attn_blocks = num_full_attn_tokens // self.block_size
        if num_full_attn_blocks == 0:
            # No cache hit.
            return [[]] * self.num_groups, 0

        # Next, find the cache hit for sliding window attention WITHIN the
        # cache hit of full attention.
        block_hashes = block_hashes[:num_full_attn_blocks]
        swa_attn_blocks, num_swa_attn_tokens = (
            self.swa_allocator.find_longest_cache_hit(block_hashes,
                                                      self.swa_group_ids))
        num_swa_attn_blocks = num_swa_attn_tokens // self.block_size
        if num_swa_attn_blocks == 0:
            # No cache hit.
            return [[]] * self.num_groups, 0

        # Truncate the full attention cache hit to the length of the
        # sliding window cache hit.
        num_blocks = num_swa_attn_blocks
        num_computed_tokens = num_swa_attn_tokens

        combined_blocks: list[list[KVCacheBlock]] = []
        for group_id in self.all_group_ids:
            if group_id in self.full_attn_group_ids:
                combined_blocks.append(full_attn_blocks[group_id][:num_blocks])
            else:
                # We don't need `[:num_blocks]` here.
                combined_blocks.append(swa_attn_blocks[group_id])
        return combined_blocks, num_computed_tokens

    def remove_skipped_blocks(
        self,
        blocks: list[list[KVCacheBlock]],
        num_computed_tokens: int,
    ) -> list[KVCacheBlock]:
        return self.swa_allocator.remove_skipped_blocks(
            blocks, self.swa_group_ids, num_computed_tokens)

    def _get_num_new_blocks(
        self,
        total_num_tokens: int,
        num_computed_tokens: int,
        new_computed_blocks: list[list[KVCacheBlock]],
        allocated_blocks: list[list[KVCacheBlock]],
    ) -> list[int]:
        # OPTIMIZATION(woosuk):
        group_id = self.full_attn_group_ids[0]
        num_new_blocks = self.full_attn_allocator.get_num_new_blocks(
            total_num_tokens,
            num_computed_tokens,
            new_computed_blocks[group_id],
            allocated_blocks[group_id],
            group_ids=(group_id, ),
        )
        return [num_new_blocks] * self.num_groups

    def sort_by_eviction_order(
        self,
        blocks: list[list[KVCacheBlock]],
    ) -> Iterable[KVCacheBlock]:
        pass
