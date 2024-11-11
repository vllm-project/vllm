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

        # When caching is enabled, assign token IDs to already allocated blocks.
        new_token_ids = None
        parent_block = None
        if self.enable_caching:
            # Figure out the token IDs to add to the blocks.
            new_token_ids = request.all_token_ids[
                request.num_computed_tokens:request.num_computed_tokens +
                num_tokens]

            # Find the last full block index.
            # TODO: This may be optimized by calculating the computed tokens.
            last_full_block_idx = len(req_blocks) - 1
            while (last_full_block_idx >= 0
                   and req_blocks[last_full_block_idx].block_hash is None):
                last_full_block_idx -= 1

            parent_block = (req_blocks[last_full_block_idx]
                            if last_full_block_idx >= 0 else None)
            token_id_idx = self._add_token_ids_to_blocks(
                blocks=req_blocks[last_full_block_idx + 1:],
                token_ids=new_token_ids,
                parent_block=parent_block)

            new_token_ids = new_token_ids[token_id_idx:]
            parent_block = req_blocks[-1]

        # No new block is needed. When caching is enabled, we make sure
        # token_id_idx is equal to len(new_token_ids), meaning that all tokens
        # are added to allocated blocks.
        if num_required_blocks <= len(req_blocks):
            assert not self.enable_caching or token_id_idx == num_tokens, \
                    f"{token_id_idx=} != {num_tokens=}"
            return []

        # Allocate new blocks considering preallocated blocks, and
        # add token IDs to them if caching is enabled.
        num_new_blocks = min(num_new_blocks + self.num_preallocate_blocks,
                             self.free_block_queue.num_free_blocks)
        new_blocks = self._get_new_blocks(num_new_blocks, new_token_ids,
                                          parent_block)
        req_blocks.extend(new_blocks)
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

        num_computed_tokens = len(computed_blocks) * self.block_size

        # When caching is enabled, get the new token IDs and the parent block
        # ID to generate cache keys.
        new_token_ids = None
        parent_block = None
        if self.enable_caching:
            # Touch the computed blocks to make sure they won't be evicted.
            self._touch(computed_blocks)

            # Get the token IDs for the blocks being allocated for hashing.
            new_token_ids = request.all_token_ids[
                num_computed_tokens:num_computed_tokens + num_tokens]
            if not new_token_ids:
                raise RuntimeError(
                    "Failed to infer the token IDs for allocation. "
                    f"#all_tokens={len(request.all_token_ids)} < "
                    f"#computed_tokens={num_computed_tokens}")

            # Get the parent block ID to construct the block chain.
            parent_block = computed_blocks[-1] if computed_blocks else None

        new_blocks = self._get_new_blocks(num_new_blocks, new_token_ids,
                                          parent_block)

        # Concatenate the computed block IDs and the new block IDs.
        self.req_to_blocks[request.request_id] = computed_blocks + new_blocks
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
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self.free_block_queue.append(block)

    def _get_new_blocks(
            self,
            num_blocks: int,
            token_ids: Optional[List[int]] = None,
            parent_block: Optional[int] = None) -> List[KVCacheBlock]:
        """Get new blocks from the free block pool, and add token IDs to
        allocated blocks if caching is enabled.
        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.
            token_ids: The token IDs in the blocks. None if caching is disabled.
            parent_block: The parent block. Used to include block chain
                in the block hash.

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

            # Evict blocks from the cache.
            if self.enable_caching:
                block_hash = curr_block.block_hash
                if (block_hash is not None
                        and block_hash in self.cached_block_hash_to_block):
                    if len(self.cached_block_hash_to_block[block_hash]) == 1:
                        del self.cached_block_hash_to_block[block_hash]
                    else:
                        del self.cached_block_hash_to_block[block_hash][
                            curr_block.block_id]
                curr_block.reset()

            curr_block.ref_cnt = 1
            ret.append(curr_block)
            idx += 1

        # Then assign token IDs to the allocated blocks.
        if self.enable_caching:
            assert token_ids is not None
            token_id_idx = self._add_token_ids_to_blocks(
                blocks=ret, token_ids=token_ids, parent_block=parent_block)
            assert token_id_idx == len(token_ids)

        return ret

    def _cache_full_block(self,
                          block: KVCacheBlock,
                          parent_block: Optional[KVCacheBlock] = None) -> None:
        """Cache a full block for prefix caching.

        Args:
            block: The block to cache.
            parent_block: The parent block. None if this is the first block.
        """
        parent_block_hash = (parent_block.block_hash
                             if parent_block is not None else None)
        assert len(block.token_ids) == self.block_size
        block.token_ids = tuple(block.token_ids)
        block_hash = hash_block_tokens(parent_block_hash, block.token_ids)
        block.block_hash = block_hash
        block.num_hashed_tokens = self.block_size + (
            parent_block.num_hashed_tokens if parent_block is not None else 0)
        self.cached_block_hash_to_block[block_hash][block.block_id] = block

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
            block.ref_cnt += 1

    def _add_token_ids_to_blocks(
            self,
            blocks: List[KVCacheBlock],
            token_ids: List[int],
            parent_block: Optional[KVCacheBlock] = None) -> int:
        """Add token IDs to a list of allocated blocks.
        If a block becomes full after adding token IDs, cache it.
        Return the token ID index that has not been added to the blocks
        if the blocks are not enough to hold all the token IDs.

        Args:
            blocks: A list of blocks to add token IDs.
            token_ids: A list of token IDs to add.
            parent_block: The parent block. None if this is the
                first block.

        Returns:
            The starting token ID index that has not been added to the blocks
            due to insufficient given blocks.
        """
        token_id_start = 0
        for curr_block in blocks:
            # If all token IDs are added, then the rest of the blocks are
            # preallocated blocks, so we only need to update the
            # parent_block_id. FIXME
            if token_id_start == len(token_ids):
                continue

            # Add token IDs to the empty slots in the block.
            empty_slots = self.block_size - len(curr_block.token_ids)
            token_id_end = min(token_id_start + empty_slots, len(token_ids))
            curr_block.token_ids.extend(token_ids[token_id_start:token_id_end])
            # Cache the block if it becomes full.
            if len(curr_block.token_ids) == self.block_size:
                self._cache_full_block(curr_block, parent_block)
            parent_block = curr_block
            token_id_start = token_id_end
        return token_id_start
