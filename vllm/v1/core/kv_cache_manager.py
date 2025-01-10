from collections import defaultdict
import math
from typing import Dict, Iterable, List, Optional, Tuple

from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.core.custom_manager import MemoryPoolOperations, get_managers
from vllm.v1.core.kv_cache_interface import KVCacheConfig
from vllm.v1.core.kv_cache_utils import (BlockHashType, FreeKVCacheBlockQueue,
                                         KVCacheBlock,
                                         generate_block_hash_extra_keys,
                                         hash_block_tokens,
                                         hash_request_tokens, ComputedTokens,
                                         intersect_ranges)
from vllm.v1.request import Request

logger = init_logger(__name__)

KVCacheBlocks = List[List[KVCacheBlock]]  # group_id -> [blocks]


class KVCacheManager:

    def __init__(
        self,
        num_gpu_blocks: int,
        max_model_len: int,
        kv_cache_config: KVCacheConfig,
        enable_caching: bool = True,
        num_preallocate_tokens: int = 64,
    ) -> None:
        # self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.max_model_len = max_model_len
        # self.max_num_blocks_per_req = cdiv(max_model_len, block_size)
        self.enable_caching = enable_caching

        self._null_block: KVCacheBlock = KVCacheBlock(-1)

        # TODO(Chen): add comments
        self.managers = get_managers(
            kv_cache_config,
            MemoryPoolOperations(get_cached_block=self._get_cached_block,
                                 get_null_block=self.get_null_block),
        )  # group_name -> manager

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
        self.num_preallocate_tokens = num_preallocate_tokens
        # TODO(Chen): add comments
        self.num_preallocate_blocks = cdiv(
            num_preallocate_tokens,
            max(manager.block_size for manager in self.managers))

        # A Block pool of all kv-cache blocks.
        self.block_pool: List[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
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
        self.req_to_blocks: Dict[str, KVCacheBlocks] = {}

    def get_computed_tokens(self,
                            request: Request) -> Tuple[int, KVCacheBlocks]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A list of blocks that are computed for the request.
        """
        if not self.enable_caching:
            # Prefix caching is disabled.
            return 0, [[] for _ in self.managers]

        # The block hashes for the request may already be computed
        # if the request was preempted and resumed.
        if not request.kv_block_hashes:
            request.set_kv_block_hashes([
                hash_request_tokens(manager.block_size, request, i)
                for i, manager in enumerate(self.managers)
            ])
        request.kv_block_hashes
        block_hashes = []

        for i, manager in enumerate(self.managers):
            if request.num_tokens % manager.block_size == 0:
                # When prompt length is divisible by the block size and all
                # blocks are cached, we need to force the recomputation of
                # the last block. We remove the hash of the last block so that
                # get_computed_tokens will skip the last block. Note that we
                # have to re-compute an entire block because allocate_slots()
                # assumes num_computed_tokens is always a multiple of the block
                # size. This limitation can potentially be removed in the future
                # to slightly improve the performance.
                block_hashes.append(request.kv_block_hashes[i][:-1])
            else:
                block_hashes.append(request.kv_block_hashes[i])

        computed_blocks: KVCacheBlocks = []  # group_id->[blocks]
        computed_tokens: List[ComputedTokens] = []  # group_id->ComputedTokens
        for i, manager in enumerate(self.managers):
            computed_tokens_i, computed_blocks_i = (
                manager.get_computed_tokens(block_hashes[i]))
            computed_blocks.append(computed_blocks_i)
            computed_tokens.append(computed_tokens_i)

        # find the common cached prefix of all groups
        num_computed_tokens = self.get_common_computed_tokens(computed_tokens)

        for i, manager in enumerate(self.managers):
            computed_blocks[i] = computed_blocks[:num_computed_tokens //
                                                 manager.block_size]
            manager.remove_useless_blocks(computed_blocks[i],
                                          num_computed_tokens)

        return num_computed_tokens, computed_blocks

    def append_slots(
        self,
        request: Request,
        num_tokens: int,
    ) -> Optional[KVCacheBlocks]:
        """Append slots to the block table of the request.
        We first append slots to already allocated blocks. If the allocated
        blocks are not enough, we allocate new blocks.

        Args:
            request: The request to append slots.
            num_tokens: The number of tokens to append.

        Returns:
            A list of new blocks if new blocks are allocated, or None
            if new blocks are required but cannot be allocated.
        """
        # we can free blocks even if we cannot schedule it
        self.free_blocks_for_sliding_window(request)

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

        for i in range(len(self.managers)):
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
                    # TODO(Chen): update comments about max_num_blocks_per_req
                    cdiv(self.max_model_len, self.managers[i].block_size) -
                    len(req_blocks[i]),
                )
                assert total_new_blocks > 0

                new_blocks_of_group = self._get_new_blocks(
                    num_block_to_allocate)
                new_blocks.append(new_blocks_of_group)
                req_blocks[i].extend(new_blocks_of_group)

        if not self.enable_caching:
            return new_blocks

        for i, manager in enumerate(self.managers):
            num_computed_full_blocks = (request.num_computed_tokens //
                                        manager.block_size)

            # NOTE(rickyx): We are assuming the `num_tokens` are actual
            # tokens rather than lookahead slots (e.g. for speculative decoding).
            # TODO(rickyx): When supporting speculative decoding, we will need to
            # differentiate between them so that we can know how many blocks are
            # full after appending the actual tokens.
            num_full_blocks_after_append = (request.num_computed_tokens +
                                            num_tokens) // manager.block_size
            assert num_full_blocks_after_append <= len(req_blocks[i])

            new_full_blocks = req_blocks[i][
                num_computed_full_blocks:num_full_blocks_after_append]
            if new_full_blocks:
                self._cache_full_blocks(
                    request=request,
                    blk_start_idx=num_computed_full_blocks,
                    full_blocks=new_full_blocks,
                    prev_block=req_blocks[i][num_computed_full_blocks - 1]
                    if num_computed_full_blocks >= 1 else None,
                    group_id=i)

        return new_blocks

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        computed_blocks: KVCacheBlocks,
    ) -> Optional[KVCacheBlocks]:
        """Allocate slots for a new request.

        Args:
            request: The request to allocate slots.
            num_tokens: The number of tokens to allocate. Note that this does
                not include the tokens that have already been computed.
            computed_blocks: The blocks that have already been computed.

        Returns:
            A list of new allocated blocks.
        """
        if num_tokens == 0:
            raise ValueError(
                f"num_tokens must be greater than 0, got {num_tokens}")

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self._touch(computed_blocks)
        else:
            assert not computed_blocks, (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        num_new_blocks = [
            manager.get_num_new_blocks(request.num_computed_tokens, num_tokens,
                                       len(computed_blocks_of_group))
            for manager, computed_blocks_of_group in zip(
                self.managers, computed_blocks)
        ]

        total_new_blocks = sum(max(x, 0) for x in num_new_blocks)
        if total_new_blocks > self.free_block_queue.num_free_blocks:
            # Cannot allocate new blocks.
            return None

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
                # TODO(Chen): update comments about max_num_blocks_per_req
                cdiv(self.max_model_len, self.managers[i].block_size) -
                len(computed_blocks[i]),
            )
            if num_block_to_allocate < 0:
                return
            assert num_block_to_allocate > 0

            new_blocks_of_group = self._get_new_blocks(num_block_to_allocate)
            new_blocks.append(new_blocks_of_group)
            # Concatenate the computed block IDs and the new block IDs.
            req_to_blocks.append(computed_blocks[i] + new_blocks_of_group)

        self.req_to_blocks[request.request_id] = req_to_blocks

        if not self.enable_caching:
            return new_blocks

        for i, manager in enumerate(self.managers):
            num_computed_tokens = len(computed_blocks[i]) * manager.block_size
            num_full_blocks = (num_computed_tokens +
                               num_tokens) // manager.block_size

            new_full_blocks = self.req_to_blocks[
                request.request_id][i][len(computed_blocks):num_full_blocks]
            if new_full_blocks:
                self._cache_full_blocks(
                    request=request,
                    blk_start_idx=len(computed_blocks),
                    # new_full_blocks are the full blocks that are not computed.
                    full_blocks=new_full_blocks,
                    prev_block=computed_blocks[i][-1]
                    if computed_blocks[i] else None,
                    group_id=i,
                )

        return new_blocks

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        When caching is enabled, we free the blocks in reverse order so that
        the tail blocks are evicted first.

        Args:
            request: The request to free the blocks.
        """
        # Default to {} in case a request is freed (aborted) before alloc.
        blocks = self.req_to_blocks.pop(request.request_id, [])
        if len(blocks) == 0:
            # This request is freed before alloc. just return
            return
        else:
            self._free_blocks(blocks)

    def free_blocks_for_sliding_window(self, request: Request) -> None:
        # NOTE(Chen): do all free before allocation to make less eviction
        req_blocks = self.req_to_blocks[request.request_id]
        removed_blocks = []
        for manager, req_blocks_of_group in zip(self.managers, req_blocks):
            removed_blocks.append(
                manager.remove_useless_blocks(req_blocks_of_group,
                                              request.num_computed_tokens))
        # TODO: better handling of free order (e.g., this order have problem
        # when different layer has different sliding window size)
        self._free_blocks(removed_blocks)

    def _free_blocks(self, blocks: KVCacheBlocks) -> None:
        # Fast path: if all blocks are empty, return. This will happen during
        # append_slots
        blocks = [b for b in blocks if len(b) > 0]
        if len(blocks) == 0:
            return
        # Free blocks in reverse order so that the tail blocks are
        # freed first.
        if self.enable_caching:
            # TODO(Chen): add comments
            ordered_blocks = self._merge_blocks_by_length_reversed(blocks)
        else:
            ordered_blocks = []
            for block_list in blocks.values():
                ordered_blocks.extend(block_list)

        for block in ordered_blocks:
            block.decr_ref()
            # TODO(Chen): add comments: never free the null_block, so do not
            # need to track its ref_cnt carefully.
            if block.ref_cnt == 0 and block != self._null_block:
                self.free_block_queue.append(block)

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

    def _touch(self, blocks: KVCacheBlocks) -> None:
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

    def _cache_full_blocks(self, request: Request, blk_start_idx: int,
                           full_blocks: List[KVCacheBlock],
                           prev_block: Optional[KVCacheBlock],
                           group_id: int) -> None:
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
            group_name: TODO(Chen): add comments
        """
        kv_block_hashes = request.kv_block_hashes[group_id]
        num_cached_block_hashes = len(kv_block_hashes)
        block_size = self.managers[group_id].block_size

        # Update the new blocks with the block hashes through the chain.
        prev_block_hash_value = None
        if prev_block is not None:
            # Previous block must have a block hash because it must be
            # a full, cached block.
            assert prev_block.block_hash is not None
            prev_block_hash_value = prev_block.block_hash.hash_value

        for i, blk in enumerate(full_blocks):
            blk_idx = blk_start_idx + i

            if blk_idx < num_cached_block_hashes:
                # The block hash may already be computed in
                # "get_computed_blocks" if the tokens are not generated by
                # this request (either the prompt tokens or the previously
                # generated tokens with preemption). In this case we simply
                # reuse the block hash.
                block_hash = kv_block_hashes[blk_idx]
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

                extra_keys = (group_id, )

                # Generate extra keys for multi-modal inputs. Note that since
                # we reach to this branch only when the block is completed with
                # generated tokens, we only need to consider the last mm input.
                extra_mm_keys, _ = generate_block_hash_extra_keys(
                    request, start_token_idx, end_token_idx, -1)
                if extra_mm_keys is not None:
                    extra_keys += extra_mm_keys

                # Compute the hash of the current block.
                block_hash = hash_block_tokens(prev_block_hash_value,
                                               block_tokens, extra_keys)
                request.append_kv_block_hashes(group_id, block_hash)

            # Update and added the full block to the cache.
            blk.block_hash = block_hash
            self.cached_block_hash_to_block[block_hash][blk.block_id] = blk
            prev_block_hash_value = block_hash.hash_value

    def _merge_blocks_by_length_reversed(
            self, blocks: KVCacheBlocks) -> List[KVCacheBlock]:
        # merge blocks from different groups based on the block size
        block_size_set = set(manager.block_size for manager in self.managers)
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
            ordered_blocks = [block for _, block in ordered_blocks_with_key]

        return ordered_blocks

    def get_common_computed_tokens(self,
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

    def get_null_block(self) -> KVCacheBlock:
        return self._null_block
