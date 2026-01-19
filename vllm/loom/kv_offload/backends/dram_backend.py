from __future__ import annotations

import ctypes
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import LoadStoreSpec
from vllm.v1.kv_offload.backend import Backend, BlockStatus
from vllm.v1.kv_offload.mediums import DRAMLoadStoreSpec
from collections import deque


class LoomDRAMBlockStatus(BlockStatus):
    _fields_ = BlockStatus._fields_ + [("block_id", ctypes.c_int64)]  # type: ignore

    def __init__(self, block_id: int):
        super().__init__()
        self.block_id = block_id


class LoomDRAMBackend(Backend):
    def __init__(self, block_size: int, num_blocks: int):
        super().__init__(block_size=block_size, medium=DRAMLoadStoreSpec.medium())

        self.num_blocks: int = num_blocks
        self.allocated_blocks_free_list: deque[int] = deque()
        self.allocated_blocks_free_list.extend(range(num_blocks))

    def get_num_free_blocks(self) -> int:
        return len(self.allocated_blocks_free_list)

    def allocate_blocks(self, block_hashes: list[BlockHash]) -> list[BlockStatus]:
        assert len(self.allocated_blocks_free_list) >= len(block_hashes)

        blocks: list[BlockStatus] = []
        for _ in range(len(block_hashes)):
            block_id = self.allocated_blocks_free_list.pop()
            blocks.append(LoomDRAMBlockStatus(block_id))

        return blocks

    def free(self, block: BlockStatus):
        assert isinstance(block, LoomDRAMBlockStatus)
        self.allocated_blocks_free_list.append(block.block_id)

    def get_load_store_spec(
        self, block_hashes: Iterable[BlockHash], blocks: Iterable[BlockStatus]
    ) -> LoadStoreSpec:
        block_ids = [block.block_id for block in blocks]
        return DRAMLoadStoreSpec(block_ids)
