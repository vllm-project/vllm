from collections import deque
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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


class KVCacheBlockPool:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.block_pool: List[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_blocks)
        ]

        # The free block list orderd by block ID in the beginning. However,
        # when a block is allocated and then freed, it will be added back
        # with the eviction order:
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
        # allocating new blocks. That's why we need to maintain a separate
        # num_free_blocks counter.
        self.free_block_queue = deque(self.block_pool)
        self.num_free_blocks = num_blocks

        # Mapping of cached block hashes to block ID. A cached block is
        # a full block with a block hash that can be used for prefix caching.
        # Full blocks in the free_block_ids and being used by running requests
        # are considered as cached blocks. When a block is evicted, meaning that
        # it is not used and being removed from free_block_ids due to out of
        # blocks, it will be removed from cached blocks.
        self.cached_block_hash_to_block = {}

    def __getitem__(self, block_id: int) -> KVCacheBlock:
        return self.block_pool[block_id]
    
    def get_free_blocks(self, num: int, token_ids: List[int], prev_block: Optional[KVCacheBlock] = None) -> List[KVCacheBlock]:
        if num > self.num_free_blocks:
            raise ValueError(f"Cannot get {num} free blocks from the pool")
        
        ret = []
        if prev_block is None:
            num_hashed_tokens = 0
        else:
            num_hashed_tokens = prev_block.num_hashed_tokens

        idx = 0
        while idx < num:
            curr_block = self.free_block_queue.popleft()
            # The block has been allocated by another request. This happens
            # when another request touches (cache hit) the block before it
            # is evicted.
            if curr_block.ref_cnt > 0:
                continue

            # Evict blocks from the cache.
            block_hash = curr_block.block_hash
            if block_hash is not None and block_hash in self.cached_block_hash_to_block:
                del self.cached_block_hash_to_block[block_hash]

            curr_block.ref_cnt = 1
            curr_block.token_ids = token_ids[idx * self.block_size:(idx + 1) * self.block_size]
            # If the block is full, compute its hash and add to the cache.
            num_block_tokens = len(curr_block.token_ids)
            if num_block_tokens == self.block_size:
                num_hashed_tokens += self.block_size
                block_hash = hash_block_tokens(prev_block.block_hash, tuple(curr_block.token_ids))
                curr_block.block_hash = block_hash
                curr_block.num_hashed_tokens = num_hashed_tokens
                self.cached_block_hash_to_block[block_hash] = curr_block
            prev_block = curr_block

            ret.append(curr_block)
            idx += 1

        self.num_free_blocks -= num
        return ret
    
    def get_cached_block(self, block_hash: int) -> Optional[KVCacheBlock]:
        if block_hash in self.cached_block_hash_to_block:
            return self.cached_block_hash_to_block[block_hash]
        return None


    def touch(self, block_id: int):
        """Touch a block manes to remove it from the free block list
        so that it will not be evicted. This happens when the block is
        freed but has not been evicted yet, and then it can be reused
        by another request.
        """
        curr_block = self.block_pool[block_id]
        # The block has no reference yet, meaning that it is in
        # the free list, so we reduce the number of free blocks by 1,
        # but not remove it from the free list now to avoid O(n) cost.
        if curr_block.ref_cnt == 0:
            self.num_free_blocks -= 1
        curr_block.ref_cnt += 1


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

        self.block_pool = KVCacheBlockPool(num_gpu_blocks, block_size)

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
            if cached_block := self.block_pool.get_cached_block(block_hash):
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
        If the last block of the request is not full, we append slots to it
        first and then allocate new blocks if needed.

        Args:
            request: The request to append slots.
            num_tokens: The number of tokens to append.
        
        Returns:
            A list of new block IDs if new blocks are allocated, or None
            if new blocks are required but cannot be allocated.
        """
        if request.num_computed_tokens < request.num_prompt_tokens:
            # (Chunked) Prefill.
            new_token_ids = request.prompt_token_ids[request.num_computed_tokens:request.num_computed_tokens + num_tokens]
        else:
            # Decode.
            num_computed_output_tokens = request.num_computed_tokens - request.num_prompt_tokens
            new_token_ids = request.output_token_ids[num_computed_output_tokens:num_computed_output_tokens + num_tokens]

        num_required_blocks = cdiv(request.num_computed_tokens + num_tokens,
                                   self.block_size)
        req_block_ids = self.req_to_block_ids[request.request_id]
        last_block_id = req_block_ids[-1]
        if num_required_blocks <= len(req_block_ids):
            # No new block is needed. Update the token IDs in the last block.
            self.block_pool[last_block_id].token_ids.extend(new_token_ids)
            # TODO: Promote the block to the cached block if it is full.
            return []

        num_new_blocks = num_required_blocks - len(req_block_ids)
        if num_new_blocks > self.block_pool.num_free_blocks:
            # Cannot allocate new blocks.
            return None

        # Allocate new blocks.
        num_new_blocks = min(num_new_blocks + self.num_preallocate_blocks,
                             self.block_pool.num_free_blocks)
        new_block_ids = self._get_new_blocks(num_new_blocks, new_token_ids, last_block_id)
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
            A list of block IDs. If computed block IDs are given, the list
            is composed of the computed block IDs followed by the new block IDs.
        """
        if num_tokens == 0:
            raise ValueError(
                f"num_tokens must be greater than 0, got {num_tokens}")

        num_required_blocks = cdiv(num_tokens, self.block_size)
        if num_required_blocks > self.block_pool.num_free_blocks:
            # Cannot allocate new blocks.
            return None

        # Determine the number of new blocks to allocate considering
        # preallocated blocks.
        num_new_blocks = min(num_required_blocks + self.num_preallocate_blocks,
                             self.block_pool.num_free_blocks)
        # Get the token IDs for the blocks being allocated for hashing.
        # Note that we expect this function to be called only once for a
        # request, so we must have new token IDs in the prompt.
        new_token_ids = request.prompt_token_ids[request.num_computed_tokens:]
        if not new_token_ids:
            raise RuntimeError(
                "Failed to infer the token IDs for allocation. "
                f"#prompt_tokens={len(request.prompt_token_ids)} < "
                f"#computed_tokens={request.num_computed_tokens}")
        # Get the previous block ID to construct the block chain.
        prev_block_id = computed_block_ids[-1] if computed_block_ids else None
        new_blocks = self._get_new_blocks(num_new_blocks, num_new_blocks, prev_block_id)
        new_block_ids = [blk.block_id for blk in new_blocks]

        # Touch the computed blocks to make sure they are not evicted.
        for block_id in computed_block_ids:
            self.block_pool.touch(block_id)

        # Concatenate the computed block IDs and the new block IDs.
        block_ids = computed_block_ids + new_block_ids
        self.req_to_block_ids[request.request_id] = block_ids
        return new_block_ids


    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request."""
        block_ids = self.req_to_block_ids.pop(request.request_id)
        # Free blocks in reverse order so that the tail blocks are freed first.
        for block_id in reversed(block_ids):
            self.block_pool[block_id].ref_cnt -= 1
            if self.block_pool[block_id].ref_cnt == 0:
                self.block_pool.free_block_queue.append(self.block_pool[block_id])


    def _get_new_blocks(self, num_blocks: int, token_ids: List[int], prev_block_id: Optional[int] = None) -> List[KVCacheBlock]:
        """Get new blocks from the free block pool.
        Note that we do not check block cache in this function.
        
        Args:
            num_blocks: The number of blocks to allocate.
            token_ids: The token IDs in the blocks.
            prev_block_id: The previous block ID. Used to include block chain
                in the block hash.
        
        Returns:
            A list of new block IDs.
        """
        assert num_blocks <= len(self.block_pool.num_free_blocks)
        prev_block = self.block_pool[prev_block_id] if prev_block_id is not None else None
        return self.block_pool.get_free_blocks(num_blocks, token_ids, prev_block)


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
    prefix caching.

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
