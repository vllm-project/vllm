# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from collections.abc import Iterable
from typing import Callable, Optional

from vllm.distributed.kv_events import (AllBlocksCleared, BlockRemoved,
                                        BlockStored, KVCacheEvent)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (BlockHashType, FreeKVCacheBlockQueue,
                                         KVCacheBlock,
                                         generate_block_hash_extra_keys,
                                         hash_block_tokens)
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        enable_kv_cache_events: bool = False,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # {block_hash: {block ID: block}}. A cached block is
        # a full block with a block hash that can be used for prefix caching.
        # The cached block may be used by running requests or in the
        # free_block_queue that could potentially be evicted.
        # NOTE: We currently don't de-duplicate the blocks in the cache,
        # meaning that if a block becomes full and is cached, we don't check
        # if there is already an identical block in the cache. This is because
        # we want to make sure the allocated block IDs won't change so that
        # block tables are append-only.
        self.cached_block_hash_to_block: dict[BlockHashType, dict[
            int, KVCacheBlock]] = defaultdict(dict)

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

    def get_cached_block(self,
                         block_hash: BlockHashType) -> Optional[KVCacheBlock]:
        """Get a cached block by the block hash, or None if cache miss.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.

        Returns:
            The cached block if it exists, or None.
        """
        cached_blocks = self.cached_block_hash_to_block.get(block_hash)
        if not cached_blocks:
            return None
        first_block_id = next(iter(cached_blocks))
        return cached_blocks[first_block_id]

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        block_hashes: list[BlockHashType],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        hash_fn: Callable,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it computes the
        block hashes for the blocks starting from `num_cached_blocks` to
        `num_full_blocks`, updating the metadata for each block
        and caching them in the `cached_block_hash_to_block`.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            block_hashes: Block hashes of the blocks in the request. Note that
            this list may be shorter than the blocks list. In this case the
            missed block hash will be computed in this function.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            hash_fn: The hash function to use for block hashes.
        """
        if num_cached_blocks == num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(block_hashes) >= num_cached_blocks
        new_block_hashes = block_hashes[num_cached_blocks:]

        # Update the new blocks with the block hashes through the chain.
        if num_cached_blocks == 0:
            prev_block_hash_value = None
        else:
            prev_block = blocks[num_cached_blocks - 1]
            assert prev_block.block_hash is not None
            prev_block_hash_value = prev_block.block_hash.hash_value

        parent_block_hash = prev_block_hash_value
        new_hashes: Optional[list[int]] = ([] if self.enable_kv_cache_events
                                           else None)
        for i, blk in enumerate(new_full_blocks):
            assert blk.block_hash is None

            if i < len(new_block_hashes):
                # The block hash may already be computed in
                # "get_computed_blocks" if the tokens are not generated by
                # this request (either the prompt tokens or the previously
                # generated tokens with preemption). In this case we simply
                # reuse the block hash.
                block_hash = new_block_hashes[i]
            else:
                # Otherwise compute the block hash and cache it in the request
                # in case it will be preempted in the future.
                blk_idx = num_cached_blocks + i
                start_token_idx = blk_idx * block_size
                end_token_idx = (blk_idx + 1) * block_size
                block_tokens = request.all_token_ids[
                    start_token_idx:end_token_idx]
                assert len(block_tokens) == block_size, (
                    f"Expected {block_size} tokens, got "
                    f"{len(block_tokens)} at {blk_idx}th block for request "
                    f"{request.request_id}({request})")

                # Generate extra keys for multi-modal inputs. Note that since
                # we reach to this branch only when the block is completed with
                # generated tokens, we only need to consider the last mm input.
                extra_keys, _ = generate_block_hash_extra_keys(
                    request, start_token_idx, end_token_idx, -1)

                # Compute the hash of the current block.
                block_hash = hash_block_tokens(hash_fn, prev_block_hash_value,
                                               block_tokens, extra_keys)
                block_hashes.append(block_hash)

            # Update and added the full block to the cache.
            blk.block_hash = block_hash
            self.cached_block_hash_to_block[block_hash][blk.block_id] = blk
            if new_hashes is not None:
                new_hashes.append(block_hash.hash_value)
            prev_block_hash_value = block_hash.hash_value

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.
                    all_token_ids[num_cached_blocks *
                                  block_size:num_full_blocks * block_size],
                    block_size=block_size,
                    lora_id=request.lora_request.id
                    if request.lora_request else None,
                ))

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the pool")

        ret: list[KVCacheBlock] = []
        idx = 0
        while idx < num_blocks:
            # First allocate blocks.
            curr_block = self.free_block_queue.popleft()
            assert curr_block.ref_cnt == 0

            # If the block is cached, evict it.
            if self.enable_caching:
                self._maybe_evict_cached_block(curr_block)

            curr_block.incr_ref()
            ret.append(curr_block)
            idx += 1

        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        block_hash = block.block_hash
        if block_hash and block_hash in self.cached_block_hash_to_block:
            block.reset_hash()
            del self.cached_block_hash_to_block[block_hash][block.block_id]

            if len(self.cached_block_hash_to_block[block_hash]) == 0:
                del self.cached_block_hash_to_block[block_hash]

            if self.enable_kv_cache_events:
                self.kv_event_queue.append(
                    BlockRemoved(block_hashes=[block_hash.hash_value]))
            return True
        return False

    def touch(self, blocks: list[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0 and block != self.null_block:
                self.free_block_queue.remove(block)
            block.incr_ref()

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
        """
        for block in ordered_blocks:
            block.decr_ref()
            # null_block should not be added to the free list.
            if block.ref_cnt == 0 and block != self.null_block:
                self.free_block_queue.append(block)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = (self.num_gpu_blocks - self.get_num_free_blocks())
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet", num_used_blocks - 1)
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = defaultdict(dict)

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return 1.0 - (self.get_num_free_blocks() / self.num_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.
        
        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
