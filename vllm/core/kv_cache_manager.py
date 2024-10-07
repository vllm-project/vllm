from typing import Dict, List, Optional, Set, Tuple

from vllm.request import Request
from vllm.logger import init_logger
from vllm.utils import cdiv

logger = init_logger(__name__)


class KVCacheManager:

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        sliding_window: Optional[int] = None,
        enable_caching: bool = True,
        watermark: float = 0.01,
    ) -> None:
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.sliding_window = sliding_window
        self.enable_caching = enable_caching
        self.watermark = watermark

        # Reserve block id 0 for padding.
        self.free_block_ids = list(range(num_gpu_blocks))
        self.req_to_block_ids: Dict[str, List[int]] = {}
        self.block_id_to_reqs: List[Set[str]] = [
            set() for _ in range(num_gpu_blocks)
        ]

    def get_computed_blocks(self, request: Request) -> List[int]:
        return []

    def append_slots(
        self,
        request: Request,
        num_tokens: int,
    ) -> Optional[List[int]]:
        num_blocks = cdiv(request.num_computed_tokens + num_tokens,
                          self.block_size)
        req_block_ids = self.req_to_block_ids[request.request_id]
        num_new_blocks = num_blocks - len(req_block_ids)
        if num_new_blocks > len(self.free_block_ids):
            # Cannot allocate new blocks.
            return None
        if num_new_blocks == 0:
            # No new block is needed.
            return []
        # Allocate new blocks.
        new_block_ids = self._get_new_blocks(num_new_blocks)
        req_block_ids.extend(new_block_ids)
        for block_id in new_block_ids:
            self.block_id_to_reqs[block_id].add(request.request_id)
        return new_block_ids

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        computed_block_ids: List[int],
    ) -> Optional[List[int]]:
        num_new_blocks = cdiv(num_tokens, self.block_size)
        if (len(self.free_block_ids) - num_new_blocks <
                self.watermark * self.num_gpu_blocks):
            # Cannot allocate new blocks.
            return None

        new_block_ids = self._get_new_blocks(num_new_blocks)
        self.req_to_block_ids[request.request_id] = (computed_block_ids +
                                                     new_block_ids)
        for block_id in new_block_ids:
            self.block_id_to_reqs[block_id].add(request.request_id)
        return new_block_ids

    def free(self, request: Request) -> None:
        block_ids = self.req_to_block_ids.pop(request.request_id)
        for block_id in block_ids:
            reqs = self.block_id_to_reqs[block_id]
            reqs.remove(request.request_id)
            if not reqs:
                self.free_block_ids.append(block_id)

    def _get_new_blocks(self, num_blocks: int) -> List[int]:
        assert num_blocks <= len(self.free_block_ids)
        new_block_ids = self.free_block_ids[-num_blocks:]
        self.free_block_ids = self.free_block_ids[:-num_blocks]
        return new_block_ids
