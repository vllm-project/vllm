from collections import defaultdict
import math
from typing import Dict, Iterable, List, Optional, Tuple

from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.core.hybrid_cache_manager.specialized_manager import MemoryPoolOperations, get_managers
from vllm.v1.core.kv_cache_utils import (BlockHashType, FreeKVCacheBlockQueue,
                                         KVCacheBlock, KVCacheBlocks,
                                         ReqKVCacheBlocks,
                                         generate_block_hash_extra_keys,
                                         hash_block_tokens,
                                         hash_request_tokens)
from vllm.v1.core.hybrid_cache_manager.utils import ComputedTokens, intersect_ranges
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
        self.max_num_blocks_per_req = [
            cdiv(max_model_len, g.kv_cache_spec.block_size)
            for g in kv_cache_config.groups
        ]
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
        # TODO: update comment
        self.num_preallocate_tokens = num_preallocate_tokens
        # TODO: min or max?
        self.num_preallocate_blocks = cdiv(
            num_preallocate_tokens,
            min(g.kv_cache_spec.block_size for g in kv_cache_config.groups))

        self._null_block: KVCacheBlock = KVCacheBlock(-1)

        # TODO(Chen): add comments
        self.managers = get_managers(
            kv_cache_config,
            MemoryPoolOperations(get_cached_block=self._get_cached_block,
                                 get_null_block=self.get_null_block),
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

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: Dict[str, ReqKVCacheBlocks] = {}

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
            return [[] for _ in self.managers], 0

        # The block hashes for the request may already be computed
        # if the request was preempted and resumed.
        if not request.kv_block_hashes:
            request.set_kv_block_hashes([
                hash_request_tokens(manager.block_size, request, i)
                for i, manager in enumerate(self.managers)
            ])

        computed_blocks: ReqKVCacheBlocks = []  # group_id->[blocks]
        computed_tokens: List[ComputedTokens] = []  # group_id->ComputedTokens
        block_hashes = request.kv_block_hashes
        for i, manager in enumerate(self.managers):
            computed_tokens_i, computed_blocks_i = (
                manager.get_computed_blocks_and_tokens(block_hashes[i]))
            computed_blocks.append(computed_blocks_i)
            computed_tokens.append(computed_tokens_i)

        if len(self.kv_cache_config.groups) == 1:
            # If there is only one group, we return the computed blocks and
            # tokens directly.
            # NOTE(woosuk): Since incomplete blocks are not eligible for
            # sharing, `num_computed_tokens` is always a multiple of
            # `block_size`.
            if len(computed_tokens[0]) == 0:
                num_computed_tokens = 0
            else:
                num_computed_tokens = computed_tokens[0][-1].end
        else:
            # find the common cached prefix of all groups. This path also works
            # for the single group case, but it is less efficient.
            num_computed_tokens = self._get_common_computed_tokens(
                computed_tokens)

            for i, manager in enumerate(self.managers):
                computed_blocks[i] = computed_blocks[:num_computed_tokens //
                                                     manager.block_size]
        self._free_blocks_for_sliding_window(computed_blocks,
                                             num_computed_tokens)
        return computed_blocks, num_computed_tokens

    def append_slots(
        self,
        request: Request,
        num_tokens: int,
    ) -> Optional[ReqKVCacheBlocks]:
        """Append slots to the block table of the request.
        We first append slots to already allocated blocks. If the allocated
        blocks are not enough, we allocate new blocks.

        Args:
            request: The request to append slots.
            num_tokens: The number of tokens to append.

        Returns:
            The new blocks if new blocks are allocated, or None if new blocks
            are required but cannot be allocated.
        """
        # we can free blocks even if we cannot schedule it
        self._free_blocks_for_sliding_window(
            self.req_to_blocks[request.request_id],
            request.num_computed_tokens)
        req_blocks = self.req_to_blocks[request.request_id]

        num_new_blocks = [
            manager.get_num_new_blocks(request.num_computed_tokens, num_tokens,
                                       len(req_blocks_of_group))
            for manager, req_blocks_of_group in zip(self.managers, req_blocks)
        ]
        total_new_blocks = sum(max(x, 0) for x in num_new_blocks)

        if total_new_blocks > self.free_block_queue.num_free_blocks:
            # Need to allocate new blocks due to insufficient pre-allocated
            # slots, but we cannot allocate new blocks due to the limit.
            return None

        # TODO(Chen): add comments
        num_preallocate_blocks = min(
            self.num_preallocate_blocks,
            (self.free_block_queue.num_free_blocks - total_new_blocks) //
            len(self.managers))

        new_blocks = []

        for i in range(len(self.kv_cache_config.groups)
                       ):  # TODO: self.num_kv_cache_groups
            if num_new_blocks[i] <= 0:
                # No new block is needed.
                new_blocks.append([])
            else:
                # Get new blocks from the free block pool considering
                # preallocated blocks.
                num_block_to_allocate = min(
                    num_new_blocks[i] + num_preallocate_blocks,
                    # Should not exceed the maximum number of blocks per request.
                    # This is especially because the block table has the shape
                    # [..., max_num_blocks_per_req].
                    # TODO(woosuk): Check and reject requests if
                    # num_prompt_tokens + max_tokens > max_model_len.
                    self.max_num_blocks_per_req[i] - len(req_blocks[i]),
                )
                assert num_block_to_allocate > 0

                new_blocks_of_group = self._get_new_blocks(num_new_blocks)
                new_blocks.append(new_blocks_of_group)
                req_blocks[i].extend(new_blocks)

        if not self.enable_caching:
            return new_blocks

        for i, manager in enumerate(self.managers):
            num_computed_full_blocks = (request.num_computed_tokens //
                                        manager.block_size)

            # NOTE(rickyx): We are assuming the `num_tokens` are actual  tokens
            # rather than lookahead slots (e.g. for speculative decoding).
            # TODO(rickyx): When supporting speculative decoding, we will need
            # to differentiate between them so that we can know how many blocks
            # are full after appending the actual tokens.
            num_full_blocks_after_append = (request.num_computed_tokens +
                                            num_tokens) // manager.block_size
            assert num_full_blocks_after_append <= len(req_blocks)

            new_full_blocks = req_blocks[i][
                num_computed_full_blocks:num_full_blocks_after_append]
            if new_full_blocks:
                self._cache_full_blocks(
                    request=request,
                    blk_start_idx=num_computed_full_blocks,
                    full_blocks=new_full_blocks,
                    prev_block=req_blocks[i][num_computed_full_blocks - 1]
                    if num_computed_full_blocks >= 1 else None,
                    kv_cache_group_id=i,
                )

        return new_blocks

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        computed_blocks: ReqKVCacheBlocks,
    ) -> Optional[ReqKVCacheBlocks]:
        """Allocate slots for a new request.

        Args:
            request: The request to allocate slots.
            num_tokens: The number of tokens to allocate. Note that this does
                not include the tokens that have already been computed.
            computed_blocks: The computed blocks.

        Returns:
           The new blocks if new blocks are allocated, or None if new blocks
            are required but cannot be allocated.
        """
        if num_tokens == 0:
            raise ValueError(
                f"num_tokens must be greater than 0, got {num_tokens}")

        # If a computed block of a request is an eviction candidate (in the
        # free queue and ref_cnt == 0), it cannot be counted as a free block
        # when allocating this request.
        num_evictable_computed_blocks = sum(1 for blk_group in computed_blocks
                                            for blk in blk_group
                                            if blk.ref_cnt == 0)

        num_new_blocks = [
            manager.get_num_new_blocks(request.num_computed_tokens, num_tokens,
                                       len(computed_blocks_of_group))
            for manager, computed_blocks_of_group in zip(
                self.managers, computed_blocks)
        ]

        total_new_blocks = sum(max(x, 0) for x in num_new_blocks)

        if (total_new_blocks > self.free_block_queue.num_free_blocks -
                num_evictable_computed_blocks):
            # Cannot allocate new blocks.
            return None

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self._touch(computed_blocks)
        else:
            assert not computed_blocks, (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        # TODO(Chen): add comments
        num_preallocate_blocks = min(
            self.num_preallocate_blocks,
            (self.free_block_queue.num_free_blocks - total_new_blocks) //
            len(self.managers))

        new_blocks = []
        req_to_blocks = []

        for i in range(len(self.managers)):
            # Determine the number of new blocks to allocate considering
            # preallocated blocks.
            num_block_to_allocate = min(
                num_new_blocks[i] + num_preallocate_blocks,
                # Should not exceed the maximum number of blocks per request.
                # This is especially because the block table has the shape
                # [..., max_num_blocks_per_req].
                # TODO(woosuk): Check and reject requests if
                # num_prompt_tokens + max_tokens > max_model_len.
                self.max_num_blocks_per_req[i] - len(computed_blocks[i]),
            )
            assert num_block_to_allocate > 0

            new_blocks_of_group = self._get_new_blocks(num_block_to_allocate)
            new_blocks.append(new_blocks_of_group)
            # Concatenate the computed block IDs and the new block IDs.
            req_to_blocks.append(computed_blocks[i] + new_blocks_of_group)

        self.req_to_blocks[request.request_id] = req_to_blocks

        if not self.enable_caching:
            return new_blocks

        for i, manager in enumerate(self.managers):
            num_computed_tokens = len(computed_blocks) * manager.block_size
            num_full_blocks = (num_computed_tokens +
                               num_tokens) // manager.block_size

            new_full_blocks = req_to_blocks[i][len(computed_blocks
                                                   ):num_full_blocks]
            if new_full_blocks:
                self._cache_full_blocks(
                    request=request,
                    blk_start_idx=len(computed_blocks),
                    # The new full blocks are the full blocks that are not computed.
                    full_blocks=new_full_blocks,
                    prev_block=computed_blocks[-1]
                    if computed_blocks else None,
                    kv_cache_group_id=i,
                )

        return new_blocks

    def _get_ordered_blocks_one_kv_cache_group(
            self, blocks: KVCacheBlocks) -> Iterable[KVCacheBlock]:
        ordered_blocks: Iterable[KVCacheBlock] = blocks
        if self.enable_caching:
            # Free blocks in reverse order so that the tail blocks are
            # freed first.
            ordered_blocks = reversed(blocks)
        return ordered_blocks

    def _get_ordered_blocks_multiple_kv_cache_groups(
            self, blocks: ReqKVCacheBlocks) -> Iterable[KVCacheBlock]:
        # Fast path: if all blocks are empty, return. This will happen during
        # append_slots
        blocks = [b for b in blocks if len(b) > 0]
        if len(blocks) == 0:
            return []
        # Free blocks in reverse order so that the tail blocks are
        # freed first.
        if self.enable_caching:
            # TODO(Chen): add comments
            # merge blocks from different groups based on the block size
            block_size_set = set(manager.block_size
                                 for manager in self.managers)
            if len(block_size_set) == 1:
                # O(n) time complexity if block_size of all groups are the same
                ordered_blocks = []
                for i in range(len(blocks[0]) - 1, -1, -1):
                    for blocks_of_group in blocks:
                        ordered_blocks.append(blocks_of_group[i])
            else:
                # O(n * log(n)) time complexity
                # TODO(Chen): optimize it to O(n*len(self.managers)) time complexity
                # NOTE: untested
                ordered_blocks_with_key = []

                for i, blocks_of_group in enumerate(blocks):
                    block_size = self.managers[i].block_size
                    for i, block in enumerate(blocks_of_group):
                        ordered_blocks_with_key.append((block_size * i, block))

                ordered_blocks_with_key.sort(reverse=True)
                ordered_blocks = [
                    block for _, block in ordered_blocks_with_key
                ]
        else:
            # TODO: need to implement this path
            raise NotImplementedError

        return ordered_blocks

    def _free_blocks(self, blocks: ReqKVCacheBlocks) -> None:
        if len(self.kv_cache_config.groups) == 1:
            ordered_blocks = self._get_ordered_blocks_one_kv_cache_group(
                blocks[0])
        else:
            ordered_blocks = self._get_ordered_blocks_multiple_kv_cache_groups(
                blocks)
        for block in ordered_blocks:
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
        # Default to [] in case a request is freed (aborted) before alloc.
        blocks = self.req_to_blocks.pop(request.request_id, [])
        if len(blocks) == 0:
            # This request is freed before alloc. just return
            return
        else:
            self._free_blocks(blocks)

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
        blocks = self.req_to_blocks[request.request_id]
        num_common_blocks_per_group = []
        for blocks_of_group in blocks:
            num_common_blocks = 0
            for block in blocks_of_group:
                if block.ref_cnt == num_running_requests:
                    num_common_blocks += 1
                else:
                    break
            num_common_blocks_per_group.append(num_common_blocks)
        return num_common_blocks_per_group

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
                self._evict_cached_block(curr_block)

            curr_block.incr_ref()
            ret.append(curr_block)
            idx += 1

        return ret

    def _evict_cached_block(self, block: KVCacheBlock) -> None:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.
        """
        block_hash = block.block_hash
        if block_hash and block_hash in self.cached_block_hash_to_block:
            block.reset_hash()
            del self.cached_block_hash_to_block[block_hash][block.block_id]

            if len(self.cached_block_hash_to_block[block_hash]) == 0:
                del self.cached_block_hash_to_block[block_hash]

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

    def _touch(self, blocks: ReqKVCacheBlocks) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        for blocks_of_group in blocks:
            for block in blocks_of_group:
                # ref_cnt=0 means this block is in the free list (i.e. eviction
                # candidate), so remove it.
                if block.ref_cnt == 0 and block != self._null_block:
                    self.free_block_queue.remove(block)
                block.incr_ref()

    def _cache_full_blocks(
        self,
        request: Request,
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
        num_cached_block_hashes = len(
            request.kv_block_hashes[kv_cache_group_id])

        # Update the new blocks with the block hashes through the chain.
        prev_block_hash_value = None
        if prev_block is not None:
            # Previous block must have a block hash because it must be
            # a full, cached block.
            assert prev_block.block_hash is not None
            prev_block_hash_value = prev_block.block_hash.hash_value

        block_size = self.kv_cache_config.groups[
            kv_cache_group_id].kv_cache_spec.block_size
        for i, blk in enumerate(full_blocks):
            blk_idx = blk_start_idx + i

            if blk_idx < num_cached_block_hashes:
                # The block hash may already be computed in
                # "get_computed_blocks" if the tokens are not generated by
                # this request (either the prompt tokens or the previously
                # generated tokens with preemption). In this case we simply
                # reuse the block hash.
                block_hash = request.kv_block_hashes[kv_cache_group_id][
                    blk_idx]
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
                                               block_tokens, extra_keys)
                request.append_kv_block_hashes(kv_cache_group_id, block_hash)

            # Update and added the full block to the cache.
            blk.block_hash = block_hash
            self.cached_block_hash_to_block[block_hash][blk.block_id] = blk
            prev_block_hash_value = block_hash.hash_value

    def get_null_block(self) -> KVCacheBlock:
        return self._null_block

    def _get_common_computed_tokens(self,
                                    computed_tokens: KVCacheBlocks) -> int:
        # TODO: add comments: the largest in the intersection, and alignment
        intersection = intersect_ranges(computed_tokens)

        # Since incomplete blocks are not eligible for sharing,
        # `num_computed_tokens` should be a multiple of `block_size` of
        # all managers, so we take the least common multiple (LCM) of them
        alignment = math.lcm(
            *[manager.block_size for manager in self.managers])

        num_computed_tokens = 0
        for range_ in intersection:
            aligned_end = cdiv(range_.end, alignment) * alignment
            if aligned_end > range_.start:
                num_computed_tokens = aligned_end
                break

        return num_computed_tokens

    def _free_blocks_for_sliding_window(self, req_blocks: ReqKVCacheBlocks,
                                        num_computed_tokens: int) -> None:
        # NOTE(Chen): do all free before allocation to make less eviction
        # req_blocks = self.req_to_blocks[request.request_id]
        removed_blocks = []
        for manager, req_blocks_of_group in zip(self.managers, req_blocks):
            removed_blocks.append(
                manager.remove_dropped_blocks(req_blocks_of_group,
                                              num_computed_tokens))
        # TODO: better handling of free order (e.g., this order have problem
        # when different layer has different sliding window size)
        self._free_blocks(removed_blocks)
