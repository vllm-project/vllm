from collections import defaultdict
from typing import Dict, List, Optional

from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.core.kv_cache_utils import (BlockHashType, FreeKVCacheBlockQueue,
                                         KVCacheBlock, hash_block_tokens,
                                         hash_request_tokens)
from vllm.v1.request import Request

logger = init_logger(__name__)


class KVCacheManager:

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        sliding_window: Optional[int] = None,
        enable_caching: bool = True,
        num_preallocate_tokens: int = 64,
    ) -> None:
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.sliding_window = sliding_window
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
        self.num_preallocate_tokens = num_preallocate_tokens
        self.num_preallocate_blocks = cdiv(num_preallocate_tokens, block_size)

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
        self.req_to_blocks: Dict[str, List[KVCacheBlock]] = {}

    def get_computed_blocks(self, request: Request) -> List[KVCacheBlock]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A list of blocks that are computed for the request.
        """
        if not self.enable_caching:
            # Prefix caching is disabled.
            return []

        computed_blocks = []

        # TODO(rickyx): potentially we could cache this so we don't have to
        # recompute it every time.
        block_hashes = hash_request_tokens(self.block_size,
                                           request.all_token_ids)

        for block_hash in block_hashes:
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := self._get_cached_block(block_hash):
                computed_blocks.append(cached_block)
            else:
                break

        return computed_blocks

    def append_slots(
        self,
        request: Request,
        num_tokens: int,
    ) -> Optional[List[KVCacheBlock]]:
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
        num_required_blocks = cdiv(request.num_computed_tokens + num_tokens,
                                   self.block_size)
        req_blocks = self.req_to_blocks[request.request_id]

        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks > self.free_block_queue.num_free_blocks:
            # Need to allocate new blocks due to insufficient pre-allocated
            # slots, but we cannot allocate new blocks due to the limit.
            return None

        # No new block is needed.
        if num_required_blocks <= len(req_blocks):
            return []

        # Get new blocks from the free block pool.
        num_new_blocks = min(
            num_new_blocks + self.num_preallocate_blocks,
            self.free_block_queue.num_free_blocks,
        )

        new_blocks = self._get_new_blocks(num_new_blocks)
        req_blocks.extend(new_blocks)

        if not self.enable_caching:
            return new_blocks

        num_computed_full_blocks = (request.num_computed_tokens //
                                    self.block_size)

        # NOTE(rickyx): We are assuming the `num_tokens` are actual
        # tokens rather than placeholders for lookahead slots. If not
        # we will need to differentiate between them so that we can
        # know how many blocks are full after appending the actual tokens.
        num_full_blocks_after_append = (request.num_computed_tokens +
                                        num_tokens) // self.block_size
        assert num_full_blocks_after_append <= len(req_blocks)

        new_full_blocks = req_blocks[
            num_computed_full_blocks:num_full_blocks_after_append]
        self._cache_full_blocks(
            request=request,
            blk_start_idx=num_computed_full_blocks,
            full_blocks=new_full_blocks,
            prev_block=req_blocks[num_computed_full_blocks - 1]
            if num_computed_full_blocks >= 1 else None,
            cached_block_hash_to_block=self.cached_block_hash_to_block,
            block_size=self.block_size,
        )

        return new_blocks

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        computed_blocks: List[KVCacheBlock],
    ) -> Optional[List[KVCacheBlock]]:
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

        # If a computed block of a request is an eviction candidate (in the
        # free queue and ref_cnt == 0), it cannot be counted as a free block
        # when allocating this request.
        num_evictable_computed_blocks = len(
            [blk for blk in computed_blocks if blk.ref_cnt == 0])

        num_required_blocks = cdiv(num_tokens, self.block_size)
        if (num_required_blocks > self.free_block_queue.num_free_blocks -
                num_evictable_computed_blocks):
            # Cannot allocate new blocks.
            return None

        # Determine the number of new blocks to allocate considering
        # preallocated blocks.
        num_new_blocks = min(
            num_required_blocks + self.num_preallocate_blocks,
            self.free_block_queue.num_free_blocks -
            num_evictable_computed_blocks)

        # Concatenate the computed block IDs and the new block IDs.
        new_blocks = self._get_new_blocks(num_new_blocks)
        self.req_to_blocks[request.request_id] = computed_blocks + new_blocks
        self._touch(computed_blocks)

        if not self.enable_caching:
            return new_blocks

        num_computed_tokens = len(computed_blocks) * self.block_size
        num_full_blocks = (num_computed_tokens + num_tokens) // self.block_size

        self._cache_full_blocks(
            request=request,
            blk_start_idx=len(computed_blocks),
            # The new full blocks are the full blocks that are not computed.
            full_blocks=self.req_to_blocks[request.request_id]
            [len(computed_blocks):num_full_blocks],
            prev_block=computed_blocks[-1] if computed_blocks else None,
            cached_block_hash_to_block=self.cached_block_hash_to_block,
            block_size=self.block_size,
        )

        return new_blocks

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        When caching is enabled, we free the blocks in reverse order so that
        the tail blocks are evicted first.

        Args:
            request: The request to free the blocks.
        """
        # Default to [] in case a request is freed (aborted) before alloc.
        blocks = self.req_to_blocks.pop(request.request_id, [])
        if self.enable_caching:
            # Free blocks in reverse order so that the tail blocks are
            # freed first.
            blocks = reversed(blocks)

        for block in blocks:
            block.decr_ref()
            if block.ref_cnt == 0:
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

        # First allocate blocks.
        ret: List[KVCacheBlock] = []
        idx = 0
        while idx < num_blocks:
            curr_block = self.free_block_queue.popleft()
            assert curr_block.ref_cnt == 0
            curr_block.incr_ref()
            ret.append(curr_block)
            idx += 1

        return ret

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

    def _touch(self, blocks: List[KVCacheBlock]) -> None:
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
                self.free_block_queue.remove(block)
            block.incr_ref()

    @staticmethod
    def _cache_full_blocks(
        request: Request,
        blk_start_idx: int,  # TODO: use prev_block's info instead.
        full_blocks: List[KVCacheBlock],
        prev_block: Optional[KVCacheBlock],
        cached_block_hash_to_block: Dict[BlockHashType, Dict[int,
                                                             KVCacheBlock]],
        block_size: int,
    ) -> None:
        # Some of the full blocks may be computed blocks cached previously.
        # Remove them from the cache.
        for blk in full_blocks:
            blk.reset_hash_metadata()

            if blk.block_hash in cached_block_hash_to_block:
                del cached_block_hash_to_block[blk.block_hash][blk.block_id]

                if len(cached_block_hash_to_block[blk.block_hash]) == 0:
                    del cached_block_hash_to_block[blk.block_hash]

        # Update the new blocks with the block hashes through the chain.
        prev_block_hash = (prev_block.block_hash
                           if prev_block is not None else None)
        for i, blk in enumerate(full_blocks):
            blk_idx = blk_start_idx + i

            block_tokens = request.all_token_ids[blk_idx *
                                                 block_size:(blk_idx + 1) *
                                                 block_size]
            assert len(block_tokens) == block_size, (
                f"Expected {block_size} tokens, got {len(block_tokens)} at "
                f"{blk_idx}th block for request {request.request_id}({request})"
            )

            # Compute the hash of the current block.
            # TODO(rickyx): we will add more metadata to the block hash
            # from the request later on.
            block_hash = hash_block_tokens(prev_block_hash,
                                           tuple(block_tokens))

            # Update and added the full block to the cache.
            blk.update_hash_metadata(
                block_hash=block_hash,
                num_hashed_tokens=(blk_idx + 1) * block_size,
            )
            cached_block_hash_to_block[block_hash][blk.block_id] = blk

            prev_block_hash = block_hash
