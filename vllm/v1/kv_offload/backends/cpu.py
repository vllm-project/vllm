# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import LoadStoreSpec
from vllm.v1.kv_offload.backend import Backend, BlockStatus
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec


class CPUBlockStatus(BlockStatus):
    _fields_ = BlockStatus._fields_ + [("block_id", ctypes.c_int64)
                                       ]  # type: ignore

    def __init__(self, block_id: int):
        super().__init__()
        self.block_id = block_id


class CPUBackend(Backend):

    def __init__(self, block_size: int, num_blocks: int):
        super().__init__(block_size=block_size,
                         medium=CPULoadStoreSpec.medium())

        self.num_blocks: int = num_blocks
        self.num_allocated_blocks: int = 0
        self.allocated_blocks_free_list: list[int] = []

    def get_num_free_blocks(self):
        return (len(self.allocated_blocks_free_list) + self.num_blocks -
                self.num_allocated_blocks)

    def allocate_blocks(self,
                        block_hashes: list[BlockHash]) -> list[BlockStatus]:
        num_fresh_blocks = min(len(block_hashes),
                               self.num_blocks - self.num_allocated_blocks)
        num_reused_blocks = len(block_hashes) - num_fresh_blocks
        assert len(self.allocated_blocks_free_list) >= num_reused_blocks

        # allocate fresh blocks
        blocks: list[BlockStatus] = []
        for _ in range(num_fresh_blocks):
            blocks.append(CPUBlockStatus(self.num_allocated_blocks))
            self.num_allocated_blocks += 1

        # allocate reused blocks
        for _ in range(num_reused_blocks):
            block_id = self.allocated_blocks_free_list.pop()
            blocks.append(CPUBlockStatus(block_id))

        return blocks

    def free(self, block: BlockStatus):
        assert isinstance(block, CPUBlockStatus)
        self.allocated_blocks_free_list.append(block.block_id)

    def get_load_store_spec(self, block_hashes: Iterable[BlockHash],
                            blocks: Iterable[BlockStatus]) -> LoadStoreSpec:
        return CPULoadStoreSpec([block.block_id for block in blocks])
