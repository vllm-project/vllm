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

    def allocate_extent(self, num_blocks: int) -> tuple[int, int]:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
        if len(self.allocated_blocks_free_list) < num_blocks:
            raise ValueError(
                "Not enough free blocks for extent allocation: "
                f"need={num_blocks} have={len(self.allocated_blocks_free_list)}"
            )

        base_block_id = self.allocated_blocks_free_list.popleft()
        last_block_id = base_block_id
        for _ in range(num_blocks - 1):
            last_block_id = self.allocated_blocks_free_list.popleft()

        if last_block_id != base_block_id + num_blocks - 1:
            raise RuntimeError(
                "Non-contiguous block ids encountered during extent allocation: "
                f"base={base_block_id} last={last_block_id} num_blocks={num_blocks}"
            )

        return base_block_id, num_blocks

    def reserve_extent(self, base_block_id: int, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
        start = int(base_block_id)
        end = start + int(num_blocks)
        reserved = set(range(start, end))
        if not reserved:
            return
        free = self.allocated_blocks_free_list
        new_free = deque([bid for bid in free if bid not in reserved])
        if len(new_free) + len(reserved) != len(free):
            missing = [bid for bid in reserved if bid not in free]
            raise RuntimeError(
                "Attempted to reserve blocks that are not free in LoomCXLBackend: "
                f"missing={missing[:16]} (n_missing={len(missing)})"
            )
        self.allocated_blocks_free_list = new_free

    def free(self, block: BlockStatus):
        assert isinstance(block, LoomCXLBlockStatus)
        self.allocated_blocks_free_list.append(block.block_id)

    def get_load_store_spec(
        self, block_hashes: Iterable[BlockHash], blocks: Iterable[BlockStatus]
    ) -> LoadStoreSpec:
        block_ids = [block.block_id for block in blocks]
        return CXLLoadStoreSpec(block_ids)
