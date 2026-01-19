from __future__ import annotations

import ctypes
from collections.abc import Iterable
from collections import deque

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import LoadStoreSpec
from vllm.v1.kv_offload.backend import Backend, BlockStatus
from vllm.v1.kv_offload.mediums import CXLLoadStoreSpec


class LoomCXLBlockStatus(BlockStatus):
    _fields_ = BlockStatus._fields_ + [("block_id", ctypes.c_int64)]  # type: ignore

    def __init__(self, block_id: int):
        super().__init__()
        self.block_id = block_id


class LoomCXLBackend(Backend):
    def __init__(
        self,
        block_size: int,
        num_blocks: int,
        *,
        numa_node: int | None = None,
        is_remote: bool = False,
        remote_numa_node: int | None = None,
    ):
        super().__init__(block_size=block_size, medium=CXLLoadStoreSpec.medium())

        self.num_blocks: int = num_blocks
        self.allocated_blocks_free_list: deque[int] = deque()
        self.allocated_blocks_free_list.extend(range(num_blocks))

        self.numa_node = numa_node
        self.is_remote = is_remote
        self.remote_numa_node = remote_numa_node

    def get_num_free_blocks(self):
        return len(self.allocated_blocks_free_list)

    def allocate_blocks(self, block_hashes: list[BlockHash]) -> list[BlockStatus]:
        assert len(self.allocated_blocks_free_list) >= len(block_hashes)

        blocks: list[BlockStatus] = []
        for _ in range(len(block_hashes)):
            block_id = self.allocated_blocks_free_list.pop()
            blocks.append(LoomCXLBlockStatus(block_id))

        return blocks

    def free(self, block: BlockStatus):
        assert isinstance(block, LoomCXLBlockStatus)
        self.allocated_blocks_free_list.append(block.block_id)

    def get_load_store_spec(
        self, block_hashes: Iterable[BlockHash], blocks: Iterable[BlockStatus]
    ) -> LoadStoreSpec:
        block_ids = [block.block_id for block in blocks]
        return CXLLoadStoreSpec(block_ids)
