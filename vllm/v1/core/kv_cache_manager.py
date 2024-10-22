from typing import Dict, List, Optional

import numpy as np

from vllm.logger import init_logger
from vllm.utils import cdiv
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

        self.free_block_ids = list(range(num_gpu_blocks))
        self.req_to_block_ids: Dict[str, List[int]] = {}
        self.ref_cnts = np.zeros(num_gpu_blocks, dtype=np.int32)

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
        num_new_blocks = min(num_new_blocks + self.num_preallocate_blocks,
                             num_free_blocks)
        new_block_ids = self._get_new_blocks(num_new_blocks)
        req_block_ids.extend(new_block_ids)
        self.ref_cnts[new_block_ids] += 1
        return new_block_ids

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        computed_block_ids: List[int],
    ) -> Optional[List[int]]:
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_free_blocks = len(self.free_block_ids)
        if num_required_blocks > num_free_blocks:
            # Cannot allocate new blocks.
            return None

        num_new_blocks = min(num_required_blocks + self.num_preallocate_blocks,
                             num_free_blocks)
        new_block_ids = self._get_new_blocks(num_new_blocks)
        block_ids = computed_block_ids + new_block_ids
        self.req_to_block_ids[request.request_id] = block_ids
        self.ref_cnts[block_ids] += 1
        return new_block_ids

    def free(self, request: Request) -> None:
        block_ids = self.req_to_block_ids.pop(request.request_id)
        self.ref_cnts[block_ids] -= 1
        for block_id in block_ids:
            ref_cnt = self.ref_cnts[block_id]
            if ref_cnt == 0:
                self.free_block_ids.append(block_id)

    def _get_new_blocks(self, num_blocks: int) -> List[int]:
        assert num_blocks <= len(self.free_block_ids)
        new_block_ids = self.free_block_ids[-num_blocks:]
        self.free_block_ids = self.free_block_ids[:-num_blocks]
        return new_block_ids
