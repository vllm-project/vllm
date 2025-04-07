from abc import ABC, abstractmethod

from vllm.v1.core.specialized_manager import (FullAttentionManager,
                                              SlidingWindowManager,
                                              SpecializedManager)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheSpec, BlockHashType, KVCacheBlock


class HybridMemoryAllocator(ABC):

    @abstractmethod
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHashType],
    ) -> list[dict[int, KVCacheBlock]]:
        raise NotImplementedError

    @abstractmethod
    def remove_skipped_blocks(
        self,
        blocks: list[dict[int, KVCacheBlock]],
        num_computed_tokens: int,
    ) -> list[KVCacheBlock]:
        raise NotImplementedError


class SingleMemoryAllocator(HybridMemoryAllocator):
    """Memory allocator for a single attention type.

    For example, models with full attention only (e.g., Llama 3, DeepSeek) and
    models with sliding window attention only (e.g., an old version of Mistral)
    use this allocator.
    """

    def __init__(
        self,
        manager: SpecializedManager,
    ):
        self.manager = manager
        self.group_ids = (0, )

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHashType],
    ) -> list[dict[int, KVCacheBlock]]:
        return self.manager.find_longest_cache_hit(block_hashes, self.group_ids)

    def remove_skipped_blocks(
        self,
        blocks: list[dict[int, KVCacheBlock]],
        num_computed_tokens: int,
    ) -> list[KVCacheBlock]:
        blocks = self.manager.remove_skipped_blocks(
            blocks, self.group_ids, num_computed_tokens)
        return [block[0] for block in blocks]


class FullAndSwaMemoryAllocator(HybridMemoryAllocator):
    """Memory allocator for full and sliding window attention.

    For example, models like Gemma 2 (1:1 full/swa) and Gemma 3 (1:5 full/swa)
    use this allocator.
    """

    def __init__(
        self,
        full_attn_manager: FullAttentionManager,
        full_attn_group_ids: tuple[int, ...],
        swa_manager: SlidingWindowManager,
        swa_group_ids: tuple[int, ...],
    ):
        self.full_attn_manager = full_attn_manager
        self.full_attn_group_ids = full_attn_group_ids
        self.swa_manager = swa_manager
        self.swa_group_ids = swa_group_ids
        self.num_groups = len(full_attn_group_ids) + len(swa_group_ids)

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHashType],
    ) -> list[dict[int, KVCacheBlock]]:
        # First, find the longest cache hit for full attention.
        full_attn_blocks = self.full_attn_manager.find_longest_cache_hit(
            block_hashes, self.full_attn_group_ids)
        if not full_attn_blocks:
            # No cache hit.
            return []

        # Next, find the cache hit for sliding window attention WITHIN the
        # cache hit of full attention.
        # TODO(woosuk): Avoid the list slicing.
        block_hashes = block_hashes[:len(full_attn_blocks)]
        swa_attn_blocks = self.swa_manager.find_longest_cache_hit(
            block_hashes, self.swa_group_ids)
        if not swa_attn_blocks:
            # No cache hit.
            return []

        # Truncate the full attention cache hit to the length of the
        # sliding window cache hit.
        num_blocks = len(swa_attn_blocks)
        combined_blocks = [{} for _ in range(num_blocks)]
        # TODO(woosuk): The below loop can be very slow. Optimize it.
        for i in range(num_blocks):
            block = combined_blocks[i]
            full_attn_block = full_attn_blocks[i]
            for group_id in self.full_attn_group_ids:
                block[group_id] = full_attn_block[group_id]
            swa_attn_block = swa_attn_blocks[i]
            for group_id in self.swa_group_ids:
                block[group_id] = swa_attn_block[group_id]
        return combined_blocks

    def remove_skipped_blocks(
        self,
        blocks: list[dict[int, KVCacheBlock]],
        num_computed_tokens: int,
    ) -> list[KVCacheBlock]:
        removed_blocks = self.swa_manager.remove_skipped_blocks(
            blocks, self.swa_group_ids, num_computed_tokens)
        # Sort the removed blocks by the eviction order.
        flattened: list[KVCacheBlock] = []
        # TODO(woosuk): The loop can be slow. Optimize it.
        for block in removed_blocks:
            flattened.extend(block.values())
        return flattened
