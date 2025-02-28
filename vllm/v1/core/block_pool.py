# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (BlockHashType, FreeKVCacheBlockQueue,
                                         KVCacheBlock,
                                         generate_block_hash_extra_keys,
                                         hash_block_tokens)
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockPool:

    def __init__(self, num_gpu_blocks: int, enable_caching: bool):
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        # A Block pool of all kv-cache blocks.
        self._block_pool: List[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(self.num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self._free_block_queue = FreeKVCacheBlockQueue(self._block_pool)

        # {block_hash: {block ID: block}}. A cached block is
        # a full block with a block hash that can be used for prefix caching.
        # The cached block may be used by running requests or in the
        # free_block_queue that could potentially be evicted.
        # NOTE: We currently don't de-duplicate the blocks in the cache,
        # meaning that if a block becomes full and is cached, we don't check
        # if there is already an identical block in the cache. This is because
        # we want to make sure the allocated block IDs won't change so that
        # block tables are append-only.
        self._cached_block_hash_to_block: Dict[BlockHashType, Dict[
            int, KVCacheBlock]] = defaultdict(dict)

        self._null_block: KVCacheBlock = KVCacheBlock(-1)

    def get_cached_block(self,
                         block_hash: BlockHashType) -> Optional[KVCacheBlock]:
        """Get a cached block by the block hash, or None if cache miss.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.

        Returns:
            The cached block if it exists, or None.
        """
        if block_hash in self._cached_block_hash_to_block:
            first_block_id = list(
                self._cached_block_hash_to_block[block_hash].keys())[0]
            return self._cached_block_hash_to_block[block_hash][first_block_id]
        return None

    def cache_full_blocks(self,
                          request: Request,
                          blocks: List[KVCacheBlock],
                          block_hashes: List[BlockHashType],
                          old_num_computed_tokens: int,
                          new_num_computed_tokens: int,
                          block_size: int,
                          virtual_layer_id: int = 0) -> None:
        """Cache a list of full blocks for prefix caching.

        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it computes the
        block hashes for the blocks starting from `blk_start_idx` to the end
        of the request's full blocks, updating the metadata for each block
        and caching them in the `cached_block_hash_to_block`.

        Args:
            request: The request to cache the blocks.
            blk_start_idx: The index of the first block in the request's blocks
                to cache.
            full_blocks: The list of blocks to update hash metadata.
            prev_block: The previous block in the chain.
        """
        num_full_blocks = new_num_computed_tokens // block_size
        num_computed_full_blocks = old_num_computed_tokens // block_size
        new_full_blocks = blocks[num_computed_full_blocks:num_full_blocks]
        if not new_full_blocks:
            return
        num_cached_block_hashes = len(block_hashes)

        if num_computed_full_blocks == 0:
            prev_block_hash_value = None
        else:
            prev_block = blocks[num_computed_full_blocks - 1]
            assert prev_block.block_hash is not None
            prev_block_hash_value = prev_block.block_hash.hash_value
        # Update the new blocks with the block hashes through the chain.

        # Find the first uncached block. This case should only happen when
        # speculative decoding is used.
        offset = 0
        for blk in new_full_blocks:
            if blk.block_hash is None:
                break
            else:
                prev_block_hash_value = blk.block_hash.hash_value
                offset += 1
        else:
            # All blocks are cached.
            return

        for i, blk in enumerate(new_full_blocks[offset:]):
            blk_idx = num_computed_full_blocks + offset + i
            assert blk.block_hash is None

            if blk_idx < num_cached_block_hashes:
                # The block hash may already be computed in
                # "get_computed_blocks" if the tokens are not generated by
                # this request (either the prompt tokens or the previously
                # generated tokens with preemption). In this case we simply
                # reuse the block hash.
                block_hash = block_hashes[blk_idx]
            else:
                # Otherwise compute the block hash and cache it in the request
                # in case it will be preempted in the future.
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
                block_hash = hash_block_tokens(prev_block_hash_value,
                                               block_tokens, virtual_layer_id,
                                               extra_keys)
                block_hashes.append(block_hash)

            # Update and added the full block to the cache.
            blk.block_hash = block_hash
            self._cached_block_hash_to_block[block_hash][blk.block_id] = blk
            prev_block_hash_value = block_hash.hash_value

    def get_new_blocks(self, num_blocks: int) -> List[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if num_blocks > self._free_block_queue.num_free_blocks:
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the pool")

        ret: List[KVCacheBlock] = []
        idx = 0
        while idx < num_blocks:
            # First allocate blocks.
            curr_block = self._free_block_queue.popleft()
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
        if block_hash and block_hash in self._cached_block_hash_to_block:
            block.reset_hash()
            del self._cached_block_hash_to_block[block_hash][block.block_id]

            if len(self._cached_block_hash_to_block[block_hash]) == 0:
                del self._cached_block_hash_to_block[block_hash]

            return True
        return False

    def touch(self, blocks: List[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0:
                self._free_block_queue.remove(block)
            block.incr_ref()

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """
        TODO: add docstring
        the first block will be evicted first
        """
        for block in ordered_blocks:
            if block == self._null_block:
                continue
            block.decr_ref()
            if block.ref_cnt == 0:
                self._free_block_queue.append(block)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = (self.num_gpu_blocks -
                           self._free_block_queue.num_free_blocks)
        if num_used_blocks > 0:
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet", num_used_blocks)
            return False

        # Remove all hashes so that no new blocks will hit.
        self._cached_block_hash_to_block = defaultdict(dict)

        # Remove all hashes from all blocks.
        for block in self._block_pool:
            block.reset_hash()

        logger.info("Successfully reset prefix cache")
        return True

    def get_num_free_blocks(self) -> int:
        return self._free_block_queue.num_free_blocks

    def get_null_block(self) -> KVCacheBlock:
        return self._null_block

    def get_usage(self) -> float:
        return 1.0 - (self.get_num_free_blocks() / self.num_gpu_blocks)
