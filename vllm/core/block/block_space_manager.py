"""Token blocks."""
from typing import List, Optional, Set, Iterable, Tuple, Dict
from abc import ABC, abstractmethod, abstractproperty

from vllm.utils import Device

_BLANK_TOKEN_ID = -1

DEFAULT_LAST_ACCESSED_TIME = -1

"""
Missing pieces:
- CoW
- Compose NaiveBlock within prefix caching block
- Separate out into files
- Integrate into BlockSpaceManager
    - CoW
    - Swap
    - append_slots logistics (who allocates)
"""


class BlockSpaceManager:

    def __init__(self):
        pass

    def can_allocate(self, seq_group) -> bool:
        """
        Assume each block in seq will consume a new block
            (sliding window is less)

        some notion of watermark
        """
        pass

    def allocate(self, seq_group) -> None:
        """
        For each logical block, allocate a block.
            sliding window rewrites old
            store in block table

        duplicate the block table of each sequence to others in seq
            group
        """

        """
        Have scheduler loop over waiting sequences.
        """
        pass

    def can_append_slot(self, seq_group) -> None:
        """
        Assume each running sequence in a group will require a new block
        Can we allocate that many blocks ?
        """
        pass

    def append_slot(self, seq) -> Optional[Tuple[int, int]]:
        """
        if block table is smaller than logical blocks
            allocate a new one
                if sliding window use an old one
                else if block is full, try to get a cached block
                else if block is not full, get any block
            check if the last one is "appendable"
                if refcount == 1, maybe promote the last block
                if refcount > 1, allocate a new one (maybe via prefix caching)
            return any CoW
        """
        pass

    def fork(self, parent_seq, child_seq) -> None:
        # called by scheduler::fork_seq
        """
        Copy the block table
        increment refcount of each.
        """
        pass

    def can_swap_in(self, seq_group) -> bool:
        pass

    def swap_in(self, seq_group) -> Dict[int, int]:
        """
        for each sequence in the group that is swapped
            for each cpu block in the block table
                if the cpu block is scheduled to be copied
                    increase the refcount
                    use the destination gpu block
                else schedule a copy by allocating a gpu block
            free the cpu block

        return the mapping of cpu block number to gpu block number
        """
        pass

    def can_swap_out(self, seq_group) -> bool:
        pass

    def swap_out(self, seq_group) -> Dict[int, int]:
        pass

    def free(self, seq) -> None:
        # called by scheduler::free_seq
        pass

        """
        if seq in block tables
            for each block in the block table
                free the block (using the appropriate device allocator)
        """

    def reset(self) -> None:
        # unused?
        pass

    def get_block_table(self, seq) -> List[int]:
        # used to get physical mappings of seq blocks, in scheduler
        pass

    def get_num_free_gpu_blocks(self) -> int:
        # used to print stats
        pass

    def get_num_free_cpu_blocks(self) -> int:
        # used to print stats
        pass




