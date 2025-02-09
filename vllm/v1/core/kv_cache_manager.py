# SPDX-License-Identifier: Apache-2.0

import math
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.core.kv_cache_utils import (BlockHashType, FreeKVCacheBlockQueue,
                                         KVCacheBlock, PrefixLength,
                                         ReqKVCacheBlocks,
                                         generate_block_hash_extra_keys,
                                         hash_block_tokens,
                                         hash_request_tokens, intersect_ranges)
from vllm.v1.core.specialized_manager import BlockPoolOperations, GroupedManager
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class KVCacheManager:
    """
    The KVCacheManager for models with one KV cache type (e.g., Llama) and
    thus one kv cache group (Refer to class `KVCacheConfig` for the meaning of 
    kv cache group).
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        num_preallocate_tokens: int = 64,
    ) -> None:
        self.kv_cache_config = kv_cache_config
        self.num_gpu_blocks = kv_cache_config.num_blocks
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        # NOTE(woosuk): To avoid frequent block allocation, we preallocate some
        # blocks for each request. For example, when a request reaches the end
        # of its block table, we preallocate N blocks in advance. This way, we
        # reduce the overhead of updating free_block_ids and ref_cnts for each
        # request every step (at the cost of some memory waste).
        # NOTE(woosuk): This is different from the "lookahead" slots since this
        # does not guarantee that the request always has N empty blocks. After
        # the request gets N empty blocks, it starts to use the blocks without
        # further allocation. When it uses up all the N empty blocks, it gets
        # N new empty blocks.
        # NOTE(Chen): For simplicity, we keep the number of preallocated blocks
        # the same for all kv cache groups, which will result in different
        # preallocated tokens for different groups if their block sizes are
        # different.
        self.num_preallocate_tokens = num_preallocate_tokens
        # NOTE(Chen): For simplicity, we keep the number of preallocated blocks
        # the same for all kv cache groups, which will result in different
        # preallocated tokens for different groups if their block sizes are
        # different.
        self.num_preallocate_blocks = cdiv(
            num_preallocate_tokens,
            max(g.kv_cache_spec.block_size for g in kv_cache_config.groups))

        self._null_block: KVCacheBlock = KVCacheBlock(-1)

        # Specialized managers for each kv cache group, which handle the
        # different kv cache management logic of different attention layers.
        self.managers = GroupedManager(
            kv_cache_config,
            max_model_len,
            enable_caching,
            BlockPoolOperations(
                get_cached_block=self._get_cached_block,
                get_null_block=self.get_null_block,
                cache_full_blocks=self._cache_full_blocks,
                get_new_blocks=self._get_new_blocks,
                touch=self._touch,
            ),
        )

        # A Block pool of all kv-cache blocks.
        self.block_pool: List[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(self.num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.block_pool)

        # {block_hash: {block ID: block}}. A cached block is
        # a full block with a block hash that can be used for prefix caching.
        # The cached block may be used by running requests or in the
        # free_block_queue that could potentially be evicted.
        # NOTE: We currently don't de-duplicate the blocks in the cache,
        # meaning that if a block becomes full and is cached, we don't check
        # if there is already an identical block in the cache. This is because
        # we want to make sure the allocated block IDs won't change so that
        # block tables are append-only.
        self.cached_block_hash_to_block: Dict[BlockHashType, Dict[
            int, KVCacheBlock]] = defaultdict(dict)

    @property
    def usage(self) -> float:
        return 1.0 - (self.free_block_queue.num_free_blocks /
                      self.num_gpu_blocks)

    def get_computed_blocks(self,
                            request: Request) -> Tuple[ReqKVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - The blocks that are computed for the request
                - The number of computed tokens.
        """
        if not self.enable_caching:
            # Prefix caching is disabled.
            return self.managers.construct(list), 0

        # The block hashes for the request may already be computed
        # if the scheduler has tried to schedule the request before.
        block_hashes = self.managers.hash_request_tokens(request)

        prefix_length, computed_blocks = self.managers.get_possible_cached_prefix(
            block_hashes)

        num_computed_tokens = self.managers._get_common_computed_tokens(
            prefix_length)

        computed_blocks = self.managers.truncate_computed_blocks(
            computed_blocks, num_computed_tokens)
        return computed_blocks, num_computed_tokens

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: Optional[ReqKVCacheBlocks] = None,
        num_new_computed_tokens: int = 0,
    ) -> Optional[ReqKVCacheBlocks]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_tokens: The number of tokens to allocate. Note that this does
                not include the tokens that have already been computed.
            new_computed_blocks_all_groups: A list of new computed blocks 
                just hitting the prefix caching.

        Blocks layout:
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """
        if num_tokens == 0:
            raise ValueError("num_tokens must be greater than 0")

        new_computed_blocks = new_computed_blocks or self.managers.construct(
            list)

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_computed_tokens = (request.num_computed_tokens +
                               num_new_computed_tokens)

        # We can free blocks that are no longer needed even if we cannot
        # schedule this request due to the limit of free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        blocks_to_free = self.managers._remove_useless_blocks(
            request, num_computed_tokens)
        self._free_blocks(blocks_to_free)

        num_new_blocks = self.managers.get_req_num_new_blocks(
            request, new_computed_blocks, num_computed_tokens, num_tokens)
        total_new_blocks = sum(max(x, 0) for x in num_new_blocks)

        # If a computed block of a request is an eviction candidate (in the
        # free queue and ref_cnt == 0), it cannot be counted as a free block
        # when allocating this request.
        num_evictable_computed_blocks = sum(
            1 for blk in self.managers.iter_all(new_computed_blocks)
            if blk.ref_cnt == 0)

        if (total_new_blocks > self.free_block_queue.num_free_blocks -
                num_evictable_computed_blocks):
            # Cannot allocate new blocks.
            return None

        # Truncate the number of pre-allocated blocks to ensure that we can
        # have at least `num_new_blocks` free blocks for each group.
        num_preallocate_blocks = min(
            self.num_preallocate_blocks,
            (self.free_block_queue.num_free_blocks -
             num_evictable_computed_blocks - total_new_blocks) //
            len(self.kv_cache_config.groups))

        new_blocks = self.managers.allocate_slots(request, new_computed_blocks,
                                                  num_new_blocks,
                                                  num_preallocate_blocks,
                                                  num_computed_tokens,
                                                  num_tokens)

        return new_blocks

    def _free_blocks(self,
                     blocks: ReqKVCacheBlocks,
                     need_reverse: bool = False) -> None:
        ordered_blocks = self.managers._sort_blocks_by_eviction_order(
            blocks, need_reverse)
        for block in ordered_blocks:
            if block == self._null_block:
                continue
            block.decr_ref()
            if block.ref_cnt == 0:
                self.free_block_queue.append(block)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        When caching is enabled, we free the blocks in reverse order so that
        the tail blocks are evicted first.

        Args:
            request: The request to free the blocks.
        """
        # Default to None in case a request is freed (aborted) before alloc.
        blocks = self.managers.pop_blocks_of_request(request.request_id)
        if blocks is None:
            # This request is freed before alloc. just return
            return
        else:
            # Reverse the blocks so that the tail blocks can have higher
            # eviction priority.
            self._free_blocks(blocks, need_reverse=True)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = (self.num_gpu_blocks -
                           self.free_block_queue.num_free_blocks)
        if num_used_blocks > 0:
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet", num_used_blocks)
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = defaultdict(dict)

        # Remove all hashes from all blocks.
        for block in self.block_pool:
            block.reset_hash()

        logger.info("Successfully reset prefix cache")
        return True

    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running_requests: int,
    ) -> List[int]:
        """Calculate the number of common prefix blocks shared by all requests
        in the RUNNING state.

        The function determines this by selecting any request and iterating
        through its blocks.  A block is considered a common prefix block if its
        `ref_cnt` equals the total number of requests in the RUNNING state.

        NOTE(woosuk): The number of requests in the RUNNING state is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because the RUNNING state only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must be in the RUNNING state, the inverse
        is not necessarily true. There may be RUNNING requests that are not
        scheduled in the current step. As of 1/1/2025, the scheduler does not
        allow this case, but it is possible in the future, as we allow more
        flexible scheduling.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled RUNNING requests that do not
        share the common prefix. Currently, this case cannot be easily detected,
        so the function returns 0 in such cases.

        Args:
            request: Any request in the RUNNING state, used to identify the
                common prefix blocks.
            num_running_requests: The total number of requests in the RUNNING
                state. This can be different from the number of scheduled
                requests in the current step.

        Returns:
            List[int]: The number of common prefix blocks per KV cache group.
        """
        assert request.status == RequestStatus.RUNNING
        return self.managers.get_num_common_prefix_blocks(
            request, num_running_requests)

    def _get_new_blocks(self, num_blocks: int) -> List[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if num_blocks > self.free_block_queue.num_free_blocks:
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the pool")

        ret: List[KVCacheBlock] = []
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

            return True
        return False

    def _get_cached_block(self,
                          block_hash: BlockHashType) -> Optional[KVCacheBlock]:
        """Get a cached block by the block hash, or None if cache miss.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.

        Returns:
            The cached block if it exists, or None.
        """
        if block_hash in self.cached_block_hash_to_block:
            first_block_id = list(
                self.cached_block_hash_to_block[block_hash].keys())[0]
            return self.cached_block_hash_to_block[block_hash][first_block_id]
        return None

    def _cache_full_blocks(
        self,
        request: Request,
        block_hashes: List[BlockHashType],
        blk_start_idx: int,
        full_blocks: List[KVCacheBlock],
        prev_block: Optional[KVCacheBlock],
        kv_cache_group_id: int,
    ) -> None:
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
            kv_cache_group_id: The KV cache group that the blocks belong to
        """
        num_cached_block_hashes = len(block_hashes)

        # Update the new blocks with the block hashes through the chain.
        prev_block_hash_value = None
        if prev_block is not None:
            # Previous block must have a block hash because it must be
            # a full, cached block.
            assert prev_block.block_hash is not None
            prev_block_hash_value = prev_block.block_hash.hash_value

        block_size = self.kv_cache_config.groups[
            kv_cache_group_id].kv_cache_spec.block_size
        # Find the first uncached block. This case should only happen when
        # speculative decoding is used.
        offset = 0
        for blk in full_blocks:
            if blk.block_hash is None:
                break
            else:
                prev_block_hash_value = blk.block_hash.hash_value
                offset += 1
        else:
            # All blocks are cached.
            return

        for i, blk in enumerate(full_blocks[offset:]):
            blk_idx = blk_start_idx + offset + i
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
                                               block_tokens, kv_cache_group_id,
                                               extra_keys)
                block_hashes.append(block_hash)

            # Update and added the full block to the cache.
            blk.block_hash = block_hash
            self.cached_block_hash_to_block[block_hash][blk.block_id] = blk
            prev_block_hash_value = block_hash.hash_value

    def _touch(self, blocks: ReqKVCacheBlocks) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0 and block != self._null_block:
                self.free_block_queue.remove(block)
            block.incr_ref()

    def free_block_hashes(self, request: Request) -> None:
        """Discard the block hashes for the request.

        NOTE: Unlike `free`, this method should be called only when the request
        is finished, not when it is preempted.
        """
        self.managers.pop_block_hashes_of_request(request.request_id)

    def get_null_block(self) -> KVCacheBlock:
        return self._null_block
