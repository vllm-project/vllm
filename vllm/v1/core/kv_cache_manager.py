from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class KVCacheBlock:
    """KV-cache block metadata."""
    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    block_id: int
    # Previous block ID. Used to include block chain in the block hash.
    prev_block_id: Optional[int] = None
    # Reference count.
    ref_cnt: int = 0
    # Token IDs in the block.
    token_ids: List[int] = field(default_factory=list)
    # The hash of the block. It is only available when the block is full.
    block_hash: Optional[int] = None
    # The number of hashed tokens. More hashed tokens means the block
    # is closer to the end of a prompt and more likely to be evicted.
    num_hashed_tokens: int = 0

    def reset(self):
        self.prev_block_id = None
        self.ref_cnt = 0
        self.token_ids.clear()
        self.block_hash = None
        self.num_hashed_tokens = 0


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

        self.block_pool: List[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # [Prefix caching] The free block list ordered by block ID in the
        # beginning. However, when a block is allocated and then freed, it
        # will be added back with the eviction order:
        # 1. The least recently used block is at the front
        # 2. If two blocks have the same last accessed time (allocated by the
        #    same sequence), the one with more hash tokens (the tail of a block
        #    chain) is at the front.
        # We maintain this order by reversing the block order when free
        # blocks of a request.
        #
        # Note that the block in this list is NOT guaranteed to be free
        # due to prefix caching. If a block in free block list is touched
        # by a request, we do not remove it immediately from free_block_list
        # due to O(n) removal cost. Instead, we remove ref_cnt>0 blocks when
        # allocating new blocks. That's why we need to maintain
        # lazy_remove_block_ids and num_free_blocks counter separately.
        #
        # [No prefix caching] The free block list is simply in the order
        # of last accessed time.
        self.free_block_queue = deque(self.block_pool)
        self.lazy_remove_block_ids = set()
        self.num_free_blocks = num_gpu_blocks

        # {block_hash: {block ID: block}}. A cached block is
        # a full block with a block hash that can be used for prefix caching.
        # The cached block may be used by running requests or in the
        # free_block_queue that could potentially be evicted.
        # NOTE: We currently don't de-duplicate the blocks in the cache,
        # meaning that if a block becomes full and is cached, we don't check
        # if there is already an identical block in the cache. This is because
        # we want to make sure the allocated block IDs won't change so that
        # block IDs are append-only.
        self.cached_block_hash_to_block: Dict[int, Dict[
            int, KVCacheBlock]] = defaultdict(dict)

        # Mapping from request ID to block IDs to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_block_ids: Dict[str, List[int]] = {}

    def get_computed_blocks(self, request: Request) -> List[int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.
        
        Returns:
            A list of block IDs that are computed for the request.
        """
        if not self.enable_caching:
            # No prefix caching.
            return []

        computed_block_ids = []
        block_hashes = self.hash_prompt_tokens(request.prompt_token_ids)

        for block_hash in block_hashes:
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := self._get_cached_block(block_hash):
                computed_block_ids.append(cached_block.block_id)
            else:
                break

        return computed_block_ids

    def append_slots(
        self,
        request: Request,
        num_tokens: int,
    ) -> Optional[List[int]]:
        """Append slots to the block table of the request.
        We first append slots to already allocated blocks. If the allocated
        blocks are not enough, we allocate new blocks.

        Args:
            request: The request to append slots.
            num_tokens: The number of tokens to append.
        
        Returns:
            A list of new block IDs if new blocks are allocated, or None
            if new blocks are required but cannot be allocated.
        """
        new_token_ids = None
        if self.enable_caching:
            if request.num_computed_tokens < request.num_prompt_tokens:
                # (Chunked) Prefill.
                new_token_ids = request.prompt_token_ids[
                    request.num_computed_tokens:request.num_computed_tokens +
                    num_tokens]
            else:
                # Decode.
                num_computed_output_tokens = (request.num_computed_tokens -
                                              request.num_prompt_tokens)
                new_token_ids = request.output_token_ids[
                    num_computed_output_tokens:num_computed_output_tokens +
                    num_tokens]

        num_required_blocks = cdiv(request.num_computed_tokens + num_tokens,
                                   self.block_size)
        req_block_ids = self.req_to_block_ids[request.request_id]

        # Assign token IDs to already allocated blocks.
        if self.enable_caching:
            last_full_block_idx = len(req_block_ids) - 1
            while (last_full_block_idx >= 0 and self.block_pool[
                    req_block_ids[last_full_block_idx]].block_hash is None):
                last_full_block_idx -= 1

            prev_block_id = (last_full_block_idx
                             if last_full_block_idx >= 0 else None)
            token_id_idx = self._add_token_ids_to_blocks(
                block_ids=req_block_ids[last_full_block_idx + 1:],
                token_ids=new_token_ids,
                prev_block_id=prev_block_id)

        if num_required_blocks <= len(req_block_ids):
            # No new block is needed. We caching is enabled,
            # then token_id_idx must be equal to len(new_token_ids),
            # meaning that all tokens are added to allocated blocks.
            assert not self.enable_caching or token_id_idx == num_tokens, \
                    f"{token_id_idx=} != {num_tokens=}"
            return []

        num_new_blocks = num_required_blocks - len(req_block_ids)
        if num_new_blocks > self.num_free_blocks:
            # Cannot allocate new blocks.
            return None

        # Allocate new blocks and add token IDs to them if caching is enabled.
        num_new_blocks = min(num_new_blocks + self.num_preallocate_blocks,
                             self.num_free_blocks)
        if self.enable_caching:
            new_token_ids = new_token_ids[token_id_idx:]
            prev_block_id = req_block_ids[-1]
        else:
            new_token_ids = None
            prev_block_id = None
        new_blocks = self._get_new_blocks(num_new_blocks, new_token_ids,
                                          prev_block_id)
        new_block_ids = [blk.block_id for blk in new_blocks]
        req_block_ids.extend(new_block_ids)
        return new_block_ids

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        computed_block_ids: List[int],
    ) -> Optional[List[int]]:
        """Allocate slots for a new request.

        Args:
            request: The request to allocate slots.
            num_tokens: The number of tokens to allocate. Note that this does
                not include the tokens that have already been computed.
            computed_block_ids: The block IDs that have already been computed.
        
        Returns:
            A list of new allocated block IDs.
        """
        if num_tokens == 0:
            raise ValueError(
                f"num_tokens must be greater than 0, got {num_tokens}")

        num_required_blocks = cdiv(num_tokens, self.block_size)
        if num_required_blocks > self.num_free_blocks:
            # Cannot allocate new blocks.
            return None

        # Determine the number of new blocks to allocate considering
        # preallocated blocks.
        num_new_blocks = min(num_required_blocks + self.num_preallocate_blocks,
                             self.num_free_blocks)
        # Get the token IDs for the blocks being allocated for hashing.
        # Note that we expect this function to be called only once per
        # request, so we must have all new token IDs in the prompt.
        num_computed_tokens = len(computed_block_ids) * self.block_size
        if self.enable_caching:
            new_token_ids = request.prompt_token_ids[
                num_computed_tokens:num_computed_tokens + num_tokens]
            if not new_token_ids:
                raise RuntimeError(
                    "Failed to infer the token IDs for allocation. "
                    f"#prompt_tokens={len(request.prompt_token_ids)} < "
                    f"#computed_tokens={num_computed_tokens}")

            # Touch the computed blocks to make sure they are not evicted.
            for block_id in computed_block_ids:
                self._touch(block_id)

            # Get the previous block ID to construct the block chain.
            prev_block_id = computed_block_ids[
                -1] if computed_block_ids else None
        else:
            new_token_ids = None
            prev_block_id = None
        new_blocks = self._get_new_blocks(num_new_blocks, new_token_ids,
                                          prev_block_id)
        new_block_ids = [blk.block_id for blk in new_blocks]

        # Concatenate the computed block IDs and the new block IDs.
        block_ids = computed_block_ids + new_block_ids
        self.req_to_block_ids[request.request_id] = block_ids
        return new_block_ids

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        When caching is enabled, we free the blocks in reverse order so that
        the tail blocks are evicted first.

        Args:
            request: The request to free the blocks.
        """
        block_ids = self.req_to_block_ids.pop(request.request_id)
        if self.enable_caching:
            # Free blocks in reverse order so that the tail blocks are
            # freed first.
            for block_id in reversed(block_ids):
                self.block_pool[block_id].ref_cnt -= 1
                if self.block_pool[block_id].ref_cnt == 0:
                    if block_id in self.lazy_remove_block_ids:
                        # This happens when a block is touched gets freed before
                        # being lazily removed from free_block_list yet. In this
                        # case we have to pay O(n) cost to move the block to the
                        # end of the free_block_list to maintain theeviction
                        # order.
                        self.free_block_queue.remove(self.block_pool[block_id])
                        self.lazy_remove_block_ids.remove(block_id)
                    self.free_block_queue.append(self.block_pool[block_id])
                    self.num_free_blocks += 1
        else:
            for block_id in block_ids:
                self.block_pool[block_id].ref_cnt -= 1
                if self.block_pool[block_id].ref_cnt == 0:
                    self.free_block_queue.append(self.block_pool[block_id])
                    self.num_free_blocks += 1

    def _get_new_blocks(
            self,
            num_blocks: int,
            token_ids: Optional[List[int]] = None,
            prev_block_id: Optional[int] = None) -> List[KVCacheBlock]:
        """Get new blocks from the free block pool, and add token IDs to
        allocated blocks if caching is enabled.
        Note that we do not check block cache in this function.
        
        Args:
            num_blocks: The number of blocks to allocate.
            token_ids: The token IDs in the blocks. None if caching is disabled.
            prev_block_id: The previous block ID. Used to include block chain
                in the block hash.
        
        Returns:
            A list of new block.
        """
        assert num_blocks <= self.num_free_blocks
        if num_blocks > self.num_free_blocks:
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the pool")

        # First allocate blocks.
        ret = []
        idx = 0
        while idx < num_blocks:
            curr_block = self.free_block_queue.popleft()
            # The block has been allocated by another request. This happens
            # when another request touches (cache hit) the block before it
            # is evicted.
            if curr_block.ref_cnt > 0:
                self.lazy_remove_block_ids.remove(curr_block.block_id)
                continue

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
                block_ids=[blk.block_id for blk in ret],
                token_ids=token_ids,
                prev_block_id=prev_block_id)
            assert token_id_idx == len(token_ids)

        self.num_free_blocks -= num_blocks
        return ret

    def _cache_full_block(self,
                          block: KVCacheBlock,
                          prev_block: Optional[KVCacheBlock] = None) -> None:
        """Cache a full block for prefix caching.

        Args:
            block: The block to cache.
            prev_block: The previous block. None if this is the first block.
        """
        prev_block_hash = (prev_block.block_hash
                           if prev_block is not None else None)
        block_hash = hash_block_tokens(prev_block_hash, tuple(block.token_ids))
        block.block_hash = block_hash
        block.num_hashed_tokens = self.block_size + (
            prev_block.num_hashed_tokens if prev_block is not None else 0)
        self.cached_block_hash_to_block[block_hash][block.block_id] = block

    def _get_cached_block(self, block_hash: int) -> Optional[KVCacheBlock]:
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

    def _touch(self, block_id: int) -> None:
        """Touch a block manes to remove it from the free block list
        so that it will not be evicted. This happens when the block is
        freed but has not been evicted yet, and then it can be reused
        by another request.

        Args:
            block_id: The ID of the block to touch.
        """
        curr_block = self.block_pool[block_id]
        # The block has no reference yet, meaning that it is in
        # the free list, so we reduce the number of free blocks by 1,
        # but not remove it from the free list now to avoid O(n) cost.
        if curr_block.ref_cnt == 0:
            self.num_free_blocks -= 1
            self.lazy_remove_block_ids.add(block_id)
        curr_block.ref_cnt += 1

    def _add_token_ids_to_blocks(self,
                                 block_ids: List[int],
                                 token_ids: List[int],
                                 prev_block_id: Optional[int] = None) -> int:
        """Add token IDs to a list of allocated blocks.
        If a block becomes full after adding token IDs, cache it.
        Return the token ID index that has not been added to the blocks
        if the blocks are not enough to hold all the token IDs.

        Args:
            block_ids: A list of block IDs to add token IDs.
            token_ids: A list of token IDs to add.
            prev_block_id: The previous block ID. None if this is the
                first block.

        Returns:
            The starting token ID index that has not been added to the blocks
            due to insufficient given blocks.
        """
        prev_block = self.block_pool[
            prev_block_id] if prev_block_id is not None else None
        token_id_start = 0
        for block_id in block_ids:
            curr_block = self.block_pool[block_id]
            curr_block.prev_block_id = prev_block_id

            # If all token IDs are added, the rest of the blocks are
            # preallocated blocks, so we only need to update the prev_block_id.
            if token_id_start == len(token_ids):
                continue

            # Add token IDs to the empty slots in the block.
            empty_slots = self.block_size - len(curr_block.token_ids)
            token_id_end = min(token_id_start + empty_slots, len(token_ids))
            curr_block.token_ids.extend(token_ids[token_id_start:token_id_end])
            # Cache the block if it becomes full.
            if len(curr_block.token_ids) == self.block_size:
                self._cache_full_block(curr_block, prev_block)
            prev_block = curr_block
            prev_block_id = prev_block.block_id
            token_id_start = token_id_end
        return token_id_start

    def hash_prompt_tokens(self, token_ids: List[int]) -> List[int]:
        """Computes hash values of a chain of blocks given a sequence of
        token IDs. The hash value is used for prefix caching.

        Args:
            token_ids: A sequence of token ids in the prompt.

        Returns:
            The list of computed hash values.
        """
        ret = []
        prev_block_hash = None
        for start in range(0, len(token_ids), self.block_size):
            end = start + self.block_size
            block_token_ids = tuple(token_ids[start:end])
            # Do not hash the block if it is not full.
            if len(block_token_ids) < self.block_size:
                break
            block_hash = hash_block_tokens(prev_block_hash, block_token_ids)
            ret.append(block_hash)
            prev_block_hash = block_hash
        return ret


@lru_cache(maxsize=1024)
def hash_block_tokens(prev_block_hash: Optional[int],
                      cur_block_token_ids: Tuple[int]) -> int:
    """Computes a hash value corresponding to the contents of a block and
    the contents of the preceding block(s). The hash value is used for
    prefix caching. We use LRU cache for this function to avoid recomputing
    hash values for the same block contents.

    TODO: Support arbitrary metadata so that we could support more
    features such as LoRA adapter.

    Args:
        prev_block_hash: The hash of the previous block. None
            if this is the first block.
        cur_block_token_ids: A tuple of token ids in the current
            block. The current block is assumed to be full.

    Returns:
        The computed hash value for the block.
    """
    return hash((prev_block_hash, *cur_block_token_ids))
