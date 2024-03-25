"""A block manager that manages token blocks."""
import enum
from itertools import count
from os.path import commonprefix
from typing import Dict, List, Optional, Set, Tuple

from vllm.block import BlockTable, PhysicalTokenBlock
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device
from vllm.core.evictor import Evictor, EvictionPolicy, make_evictor
from vllm.core.block.naive_block import NaiveBlockAllocator, NaiveBlock
from vllm.core.block.interfaces import DeviceAwareBlockAllocator, Block
from vllm.utils import chunk_list, cdiv


class BlockTable:
    """The goal of this class is to map sequences to blocks.
    Upon construction, it is bound to a sequence ID.

    it is basically a list of blocks. 
    """

    def __init__(
        self,
        block_size: int,
        block_allocator: DeviceAwareBlockAllocator,
        _blocks: Optional[List[Block]] = None,
    ):
        self._block_size = block_size
        self._allocator = block_allocator
        self._blocks: Optional[List[Block]] = _blocks
        self._num_full_slots = len(self._get_all_token_ids())

    @staticmethod
    def get_num_required_blocks(token_ids: List[int], block_size: int) -> int:
        return cdiv(len(token_ids), block_size)

    def can_allocate(self,
                     token_ids: List[int],
                     device: Device = Device.GPU) -> bool:
        pass

    def allocate(self,
                 token_ids: List[int],
                 device: Device = Device.GPU) -> None:
        assert not self._is_allocated
        assert token_ids
        self._blocks = self._allocate_blocks_for_token_ids(prev_block=None,
                                                           token_ids=token_ids,
                                                           device=device)
        self._num_full_slots = len(token_ids)

    def append_token_ids(self, token_ids: List[int]) -> None:
        assert self._is_allocated

        self.ensure_num_empty_slots(num_empty_slots=len(token_ids))

        blocks = self._blocks[self._num_full_slots // self._block_size:]
        first_chunk_size = self._block_size - self._num_full_slots % self._block_size
        token_blocks = [token_ids[:first_chunk_size]] + chunk_list(
            token_ids[first_chunk_size:], self._block_size)

        for block, token_block in zip(blocks, token_blocks):
            block.append_token_ids(token_block)

        self._num_full_slots += len(token_ids)

    def ensure_num_empty_slots(self, num_empty_slots: int) -> None:
        # Currently the block table only supports
        # appending tokens to GPU blocks.
        device = Device.GPU
        assert self._is_allocated

        if self._num_empty_slots >= num_empty_slots:
            return

        slots_to_allocate = num_empty_slots - self._num_empty_slots
        blocks_to_allocate = cdiv(slots_to_allocate, self._block_size)

        for _ in range(blocks_to_allocate):
            self._blocks.append(
                self._allocator.allocate_mutable(prev_block=self._blocks[-1],
                                                 device=device))

    def fork(self) -> "BlockTable":
        assert self._is_allocated
        forked_blocks = self._allocator.fork(self._blocks[-1])
        return BlockTable(
            block_size=self._block_size,
            block_allocator=self._allocator,
            _blocks=forked_blocks,
        )

    def free(self) -> None:
        assert self._is_allocated
        for block in self._blocks:
            self._allocator.free(block)
        self._blocks = None

    @property
    def physical_block_ids(self) -> List[int]:
        assert self._is_allocated
        return [block.physical_block_index for block in self._blocks]

    def _allocate_blocks_for_token_ids(self, prev_block: Optional[Block],
                                       token_ids: List[int],
                                       device: Device) -> List[Block]:
        blocks = []
        for block_token_ids in chunk_list(token_ids, self._block_size):
            if len(block_token_ids) == self._block_size:
                # If the block is full, create an immutable block.
                prev_block = self._allocator.allocate_immutable(
                    prev_block, token_ids=block_token_ids, device=device)
            else:
                # Else, partially fill a mutable block with token ids.
                prev_block = self._allocator.allocate_mutable(
                    prev_block=prev_block, device=device)
                prev_block.append_token_ids(block_token_ids)
            blocks.append(prev_block)

        return blocks

    def _get_all_token_ids(self) -> List[int]:
        # NOTE: This function is O(seq_len); use sparingly.
        token_ids = []

        if not self._is_allocated:
            return token_ids

        for block in self._blocks:
            token_ids.extend(block.token_ids)

        return token_ids

    @property
    def _is_allocated(self) -> bool:
        return self._blocks is not None

    @property
    def _num_empty_slots(self) -> int:
        assert self._is_allocated
        return len(self._blocks) * self._block_size - self._num_full_slots

    @property
    def num_full_slots(self) -> int:
        return self._num_full_slots
