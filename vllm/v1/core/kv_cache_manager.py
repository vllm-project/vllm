from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

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
    # Token IDs in the block.
    token_ids: List[int] = field(default_factory=list)
    # The hash of the block. It is only available when the block is full.
    block_hash: Optional[int] = None

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

        self.block_pool: List[KVCacheBlock] = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]
        self.free_block_ids = deque(list(range(num_gpu_blocks)))
        self.allocated_block_ids = set([])
        self.req_to_block_ids: Dict[str, List[int]] = {}


    def get_computed_blocks(self, request: Request) -> List[int]:
        if not self.enable_caching:
            # No prefix caching.
            return []
        # TODO(woosuk): Implement hash-based caching.
        return []

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
        num_required_blocks = cdiv(request.num_computed_tokens + num_tokens,
                                   self.block_size)
        req_block_ids = self.req_to_block_ids[request.request_id]
        if num_required_blocks <= len(req_block_ids):
            # No new block is needed.
            return []

        num_new_blocks = num_required_blocks - len(req_block_ids)
        num_free_blocks = len(self.free_block_ids)
        if num_new_blocks > num_free_blocks:
            # Cannot allocate new blocks.
            return None

        # Allocate new blocks.
        # TODO: If we need to allocate more than one block, it is also
        # possible that the block is already computed.
        num_new_blocks = min(num_new_blocks + self.num_preallocate_blocks,
                             num_free_blocks)
        new_block_ids = self._get_new_blocks(num_new_blocks)
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
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_free_blocks = len(self.free_block_ids)
        if num_required_blocks > num_free_blocks:
            # Cannot allocate new blocks.
            return None

        # Allocate new blocks.
        num_new_blocks = min(num_required_blocks + self.num_preallocate_blocks,
                             num_free_blocks)
        new_block_ids = self._get_new_blocks(num_new_blocks)

        # Increase the reference count of the computed blocks.
        for block_id in computed_block_ids:
            self.block_pool[block_id].ref_cnt += 1

        # Concatenate the computed block IDs and the new block IDs.
        block_ids = computed_block_ids + new_block_ids
        self.req_to_block_ids[request.request_id] = block_ids
        return new_block_ids


    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request."""
        block_ids = self.req_to_block_ids.pop(request.request_id)
        for block_id in block_ids:
            self.block_pool[block_id].ref_cnt -= 1
            if self.block_pool[block_id].ref_cnt == 0:
                self.allocated_block_ids.remove(block_id)
                self.free_block_ids.append(block_id)
                # TODO: Add to evictor.


    def _get_new_blocks(self, num_blocks: int, prev_block_id: Optional[int] = None) -> List[int]:
        """Get new blocks from the free block pool.
        Specifically, we move block IDs from free_block_ids to
        allocated_block_ids, and set their reference count to 1.

        Note that we do not reuse existing blocks in this function.
        
        Args:
            num_blocks: The number of blocks to allocate.
            prev_block_id: The previous block ID. Used to include block chain
                in the block hash.
        
        Returns:
            A list of new block IDs.
        """
        assert num_blocks <= len(self.free_block_ids)
        new_block_ids = self.free_block_ids.popleft(num_blocks)
        for block_id in new_block_ids:
            self.block_pool[block_id].prev_block_id = prev_block_id
            self.block_pool[block_id].ref_cnt = 1
            prev_block_id = block_id

        self.allocated_block_ids.update(new_block_ids)
        return new_block_ids
