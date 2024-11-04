from collections import defaultdict
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
    # Parent block ID. Used to include block chain in the block hash.
    parent_block_id: Optional[int] = None
    # Reference count.
    ref_cnt: int = 0
    # Token IDs in the block.
    token_ids: List[int] = field(default_factory=list)
    # The hash of the block. It is only available when the block is full.
    block_hash: Optional[int] = None
    # The number of hashed tokens. More hashed tokens means the block
    # is closer to the end of a prompt and more likely to be evicted.
    num_hashed_tokens: int = 0

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: Optional["KVCacheBlock"] = None
    next_free_block: Optional["KVCacheBlock"] = None

    def reset(self):
        """Reset the block metadata."""
        self.parent_block_id = None
        self.ref_cnt = 0
        self.token_ids.clear()
        self.block_hash = None
        self.num_hashed_tokens = 0


class FreeKVCacheBlockQueue:
    """This class organizes a list of KVCacheBlock objects to a doubly linked
    list of free blocks. We implement this class instead of using Python
    builtin deque to support removing a block in the middle of the queue
    in O(1) time. To close the performance gap to the builtin deque which is
    implemented in C++, this class does not allocate any Python objects when
    manipulating the linked list. Instead, this class manipulates the 
    prev_free_block and next_free_block attributes of the given blocks.

    The queue is ordered by block ID in the beginning. When a block is allocated
    and then freed, it will be appended back with the eviction order:
    1. The least recent used block is at the front (LRU).
    2. If two blocks have the same last accessed time (allocated by the
       same sequence), the one with more hash tokens (the tail of a block
       chain) is at the front.
    Note that we maintain this order by reversing the block order when free
    blocks of a request. This operation is outside of this class.

    Args:
        blocks: A list of KVCacheBlock objects.
    """

    def __init__(self, blocks: List[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)

        # Initialize the doubly linked list of free blocks.
        self.free_list_head = blocks[0]
        self.free_list_tail = blocks[-1]
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

    def popleft(self) -> KVCacheBlock:
        """Pop the first free block and reduce num_free_blocks by 1.
        
        Returns:
            The first free block.
        """
        if not self.free_list_head:
            raise ValueError("No free blocks available")

        block = self.free_list_head
        self.remove(block)
        return block

    def remove(self, block: KVCacheBlock) -> None:
        """Remove a block in the free list and reduce num_free_blocks by 1.
        
        Args:
            block: The block to remove.
        """
        if block.prev_free_block is not None:
            # Link the previous block to the next block.
            block.prev_free_block.next_free_block = block.next_free_block
        if block.next_free_block is not None:
            # Link the next block to the previous block.
            block.next_free_block.prev_free_block = block.prev_free_block

        if block == self.free_list_head:
            # Update the head if the block is the head.
            self.free_list_head = block.next_free_block
        if block == self.free_list_tail:
            # Update the tail if the block is the tail.
            self.free_list_tail = block.prev_free_block

        # Remove the block from the linked list.
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        """Put a block back into the free list and increase
        num_free_blocks by 1.

        Args:
            block: The block to append.
        """
        if self.free_list_tail is not None:
            # Link the last block to the new block.
            self.free_list_tail.next_free_block = block
            block.prev_free_block = self.free_list_tail
            self.free_list_tail = block
        else:
            # The free list is empty.
            self.free_list_head = self.free_list_tail = block

        block.next_free_block = None
        self.num_free_blocks += 1

    def get_all_free_blocks(self) -> List[KVCacheBlock]:
        """Get all free blocks in the free list. Mainly used for testing.
        
        Returns:
            A list of free blocks.
        """
        ret = []
        curr_block = self.free_list_head
        while curr_block is not None:
            ret.append(curr_block)
            curr_block = curr_block.next_free_block
        return ret


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
        self.free_block_queue = FreeKVCacheBlockQueue(self.block_pool)

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
            # Prefix caching is disabled.
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
        num_required_blocks = cdiv(request.num_computed_tokens + num_tokens,
                                   self.block_size)
        req_block_ids = self.req_to_block_ids[request.request_id]

        num_new_blocks = num_required_blocks - len(req_block_ids)
        if num_new_blocks > self.free_block_queue.num_free_blocks:
            # Need to allocate new blocks due to insufficient pre-allocated
            # slots, but we cannot allocate new blocks due to the limit.
            return None

        # Assign token IDs to already allocated blocks.
        new_token_ids = None
        parent_block_id = None
        if self.enable_caching:
            # Figure out the token IDs to add to the blocks.
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

            # Find the last full block index.
            # TODO: This may be optimized by calculating the computed tokens.
            last_full_block_idx = len(req_block_ids) - 1
            while (last_full_block_idx >= 0 and self.block_pool[
                    req_block_ids[last_full_block_idx]].block_hash is None):
                last_full_block_idx -= 1

            parent_block_id = (last_full_block_idx
                               if last_full_block_idx >= 0 else None)
            token_id_idx = self._add_token_ids_to_blocks(
                block_ids=req_block_ids[last_full_block_idx + 1:],
                token_ids=new_token_ids,
                parent_block_id=parent_block_id)

            new_token_ids = new_token_ids[token_id_idx:]
            parent_block_id = req_block_ids[-1]

        # No new block is needed. When caching is enabled, we make sure
        # token_id_idx is equal to len(new_token_ids), meaning that all tokens
        # are added to allocated blocks.
        if num_required_blocks <= len(req_block_ids):
            assert not self.enable_caching or token_id_idx == num_tokens, \
                    f"{token_id_idx=} != {num_tokens=}"
            return []

        # Allocate new blocks considering preallocated blocks, and
        # add token IDs to them if caching is enabled.
        num_new_blocks = min(num_new_blocks + self.num_preallocate_blocks,
                             self.free_block_queue.num_free_blocks)
        new_blocks = self._get_new_blocks(num_new_blocks, new_token_ids,
                                          parent_block_id)
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

        # If a computed block of a request is an eviction candidate (in the
        # free queue and ref_cnt == 0), it cannot be counted as a free block
        # when allocating this request.
        num_evictable_computed_blocks = len([
            bid for bid in computed_block_ids
            if self.block_pool[bid].ref_cnt == 0
        ])

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

            # Touch the computed blocks to make sure they won't be evicted.
            self._touch(computed_block_ids)

            # Get the parent block ID to construct the block chain.
            parent_block_id = computed_block_ids[
                -1] if computed_block_ids else None
        else:
            new_token_ids = None
            parent_block_id = None
        new_blocks = self._get_new_blocks(num_new_blocks, new_token_ids,
                                          parent_block_id)
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
            block_ids = reversed(block_ids)

        for block_id in block_ids:
            self.block_pool[block_id].ref_cnt -= 1
            if self.block_pool[block_id].ref_cnt == 0:
                self.free_block_queue.append(self.block_pool[block_id])

    def _get_new_blocks(
            self,
            num_blocks: int,
            token_ids: Optional[List[int]] = None,
            parent_block_id: Optional[int] = None) -> List[KVCacheBlock]:
        """Get new blocks from the free block pool, and add token IDs to
        allocated blocks if caching is enabled.
        Note that we do not check block cache in this function.
        
        Args:
            num_blocks: The number of blocks to allocate.
            token_ids: The token IDs in the blocks. None if caching is disabled.
            parent_block_id: The parent block ID. Used to include block chain
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
                block_ids=[blk.block_id for blk in ret],
                token_ids=token_ids,
                parent_block_id=parent_block_id)
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
        block_hash = hash_block_tokens(parent_block_hash,
                                       tuple(block.token_ids))
        block.block_hash = block_hash
        block.num_hashed_tokens = self.block_size + (
            parent_block.num_hashed_tokens if parent_block is not None else 0)
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

    def _touch(self, block_ids: List[int]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            block_id: The ID of the block to touch.
        """
        for block_id in block_ids:
            curr_block = self.block_pool[block_id]
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if curr_block.ref_cnt == 0:
                self.free_block_queue.remove(curr_block)
            curr_block.ref_cnt += 1

    def _add_token_ids_to_blocks(self,
                                 block_ids: List[int],
                                 token_ids: List[int],
                                 parent_block_id: Optional[int] = None) -> int:
        """Add token IDs to a list of allocated blocks.
        If a block becomes full after adding token IDs, cache it.
        Return the token ID index that has not been added to the blocks
        if the blocks are not enough to hold all the token IDs.

        Args:
            block_ids: A list of block IDs to add token IDs.
            token_ids: A list of token IDs to add.
            parent_block_id: The parent block ID. None if this is the
                first block.

        Returns:
            The starting token ID index that has not been added to the blocks
            due to insufficient given blocks.
        """
        parent_block = self.block_pool[
            parent_block_id] if parent_block_id is not None else None
        token_id_start = 0
        for block_id in block_ids:
            curr_block = self.block_pool[block_id]
            curr_block.parent_block_id = parent_block_id

            # If all token IDs are added, then the rest of the blocks are
            # preallocated blocks, so we only need to update the
            # parent_block_id.
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
            parent_block_id = parent_block.block_id
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
        parent_block_hash = None
        for start in range(0, len(token_ids), self.block_size):
            end = start + self.block_size
            block_token_ids = tuple(token_ids[start:end])
            # Do not hash the block if it is not full.
            if len(block_token_ids) < self.block_size:
                break
            block_hash = hash_block_tokens(parent_block_hash, block_token_ids)
            ret.append(block_hash)
            parent_block_hash = block_hash
        return ret


@lru_cache(maxsize=1024)
def hash_block_tokens(parent_block_hash: Optional[int],
                      cur_block_token_ids: Tuple[int]) -> int:
    """Computes a hash value corresponding to the contents of a block and
    the contents of the preceding block(s). The hash value is used for
    prefix caching. We use LRU cache for this function to avoid recomputing
    hash values for the same block contents.

    TODO: Support arbitrary metadata so that we could support more
    features such as LoRA adapter.

    Args:
        parent_block_hash: The hash of the parent block. None
            if this is the first block.
        cur_block_token_ids: A tuple of token ids in the current
            block. The current block is assumed to be full.

    Returns:
        The computed hash value for the block.
    """
    return hash((parent_block_hash, *cur_block_token_ids))
