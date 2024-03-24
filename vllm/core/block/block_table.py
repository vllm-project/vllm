


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
        token_ids: List[int],
        block_size: int,
        block_allocator: DeviceAwareBlockAllocator,
    ):
        assert token_ids
        self._token_ids = token_ids
        self._block_size = block_size
        self._allocator = block_allocator
        self._blocks: Optional[List[Block]] = None
    
    def allocate(self, device: Device = Device.GPU) -> None:
        assert not self._is_allocated
        self._blocks = self._allocate_blocks_for_token_ids(prev_block=None, token_ids=self._token_ids, device=device)

    def append_token_ids(self, token_ids: List[int]) -> None:
        """Track first mutable block.
        Append tokens to it.
            the block will manage CoW itself.
        """
        assert self._is_allocated

        # Currently the block table only supports
        # appending tokens to GPU blocks.
        device = Device.GPU

        # TODO optimize O(seq_len)
        for block in self._blocks:
            if block.is_full:
                continue

            num_empty_slots = block.num_empty_slots
            token_ids_to_append = token_ids[:num_empty_slots]
            token_ids = token_ids[num_empty_slots:]

            block.append_token_ids(token_ids_to_append)

            if not token_ids:
                break

        # If not enough blocks to store all tokens, allocate new blocks.
        if token_ids:
            assert self._blocks
            last_block = self._blocks[-1]

            new_blocks = self._allocate_blocks_for_token_ids(prev_block=last_block, token_ids=token_ids, device=device)
            self._blocks.extend(new_blocks)

    def ensure_num_empty_slots(self, num_empty_slots: int) -> None:
        # Currently the block table only supports
        # appending tokens to GPU blocks.
        device = Device.GPU
        assert self._is_allocated

        # TODO optimize O(seq_len)
        cur_num_empty_slots = sum(block.num_empty_slots for block in self._blocks)

        if cur_num_empty_slots >= num_empty_slots:
            return

        slots_to_allocate = num_empty_slots - cur_num_empty_slots
        blocks_to_allocate = cdiv(slots_to_allocate, self._block_size)
        
        for _ in range(blocks_to_allocate):
            self._blocks.append(self._allocator.allocate_mutable(prev_block=self._blocks[-1], device=device))


    def free(self) -> None:
        assert self._is_allocated
        for block in self._blocks:
            self._allocator.free(block)
        self._blocks = None

    @property
    def physical_block_ids(self) -> List[int]:
        assert self._is_allocated
        return [block.physical_block_index for block in self._blocks]

    def _allocate_blocks_for_token_ids(self, prev_block: Optional[Block], token_ids: List[int], device: Device) -> List[Block]:
        blocks = []
        for block_token_ids in chunk_list(token_ids, self._block_size):
            if len(block_token_ids) == self._block_size:
                # If the block is full, create an immutable block.
                prev_block = self._allocator.allocate_immutable(prev_block, token_ids=block_token_ids, device=device)
            else:
                # Else, partially fill a mutable block with token ids.
                prev_block = self._allocator.allocate_mutable(prev_block=prev_block, device=device)
                prev_block.append_token_ids(block_token_ids)
            blocks.append(prev_block)

        return blocks

    @property
    def _is_allocated(self) -> bool:
        return self._blocks is not None
