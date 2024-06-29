import enum
from abc import ABC, abstractmethod
from typing import OrderedDict

from vllm.block import PhysicalTokenBlock


class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()


class Evictor(ABC):
    """The Evictor subclasses should be used by the BlockAllocator class to
    handle eviction of freed PhysicalTokenBlocks.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __contains__(self, block_hash: int) -> bool:
        pass

    @abstractmethod
    def evict(self) -> PhysicalTokenBlock:
        """Runs the eviction algorithm and returns the evicted block"""
        pass

    @abstractmethod
    def add(self, block: PhysicalTokenBlock):
        """Adds block to the evictor, making it a candidate for eviction"""
        pass

    @abstractmethod
    def remove(self, block_hash: int) -> PhysicalTokenBlock:
        """Simply removes the block with the hash value block_hash from the
        evictor. Caller is responsible for making sure that block_hash is
        contained in the evictor before calling remove. Should be used to
        "bring back" blocks that have been freed but not evicted yet.
        """
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        pass


class LRUEvictor(Evictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the PhysicalTokenBlock. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted. If two blocks each have the lowest last_accessed time and
    highest num_hashed_tokens value, then one will be chose arbitrarily
    """

    def __init__(self):
        self.free_table: OrderedDict[int, PhysicalTokenBlock] = OrderedDict()

    def __contains__(self, block_hash: int) -> bool:
        return block_hash in self.free_table

    def evict(self) -> PhysicalTokenBlock:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        evicted_block = next(iter(self.free_table.values()))
        # The blocks with the lowest timestamps should be placed consecutively
        # at the start of OrderedDict. Loop through all these blocks to
        # find the one with maximum number of hashed tokens.
        for _, block in self.free_table.items():
            if evicted_block.last_accessed < block.last_accessed:
                break
            if evicted_block.num_hashed_tokens < block.num_hashed_tokens:
                evicted_block = block

        self.free_table.pop(evicted_block.block_hash)

        evicted_block.computed = False
        return evicted_block

    def add(self, block: PhysicalTokenBlock):
        self.free_table[block.block_hash] = block

    def remove(self, block_hash: int) -> PhysicalTokenBlock:
        if block_hash not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        block: PhysicalTokenBlock = self.free_table[block_hash]
        self.free_table.pop(block_hash)
        return block

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


def make_evictor(eviction_policy: EvictionPolicy) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        return LRUEvictor()
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")
