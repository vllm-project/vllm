


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
from vllm.utils import chunk_list


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
        assert self._blocks is None

        blocks = []
        prev_block = None
        for block_token_ids in chunk_list(self._token_ids, self._block_size):
            if len(block_token_ids) == self._block_size:
                # If the block is full, create an immutable block.
                prev_block = self._allocator.allocate_immutable(prev_block, token_ids=block_token_ids, device=device)
            else:
                # Else, partially fill a mutable block with token ids.
                prev_block = self._allocator.allocate_mutable(prev_block=prev_block, device=device)
                prev_block.append_token_ids(block_token_ids)

            blocks.append(prev_block)

        self._blocks = blocks

    """
    Update token ids
    Ensure lookahead
    """
    def append_token_ids(self, token_ids: List[int]):
        pass

    def free(self) -> None:
        assert self._blocks is not None
        for block in self._blocks:
            self._allocator.free(block)
        self._blocks = None
