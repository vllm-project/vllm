import enum
from typing import Dict, List, Optional
from abc import ABC, abstractmethod, abstractproperty

from vllm.block import PhysicalTokenBlock


class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()
    FIFO = enum.auto()


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

    @abstractproperty
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
        self.free_table: Dict[int, PhysicalTokenBlock] = {}

    def __contains__(self, block_hash: int) -> bool:
        return block_hash in self.free_table

    # TODO: The performance of this evict function can be optimized further.
    def evict(self) -> PhysicalTokenBlock:
        free_blocks: List[PhysicalTokenBlock] = list(self.free_table.values())
        if len(free_blocks) == 0:
            raise ValueError("No usable cache memory left")

        # Find lowest timestamp
        lowest_timestamp = free_blocks[0].last_accessed
        for block in free_blocks:
            if block.last_accessed < lowest_timestamp:
                lowest_timestamp = block.last_accessed

        # Find all blocks with the lowest timestamp
        least_recent: List[PhysicalTokenBlock] = []
        for block in free_blocks:
            if block.last_accessed == lowest_timestamp:
                least_recent.append(block)

        # Find highest prefix count per block
        highest_num_hashed_tokens = 0
        for block in least_recent:
            if block.num_hashed_tokens > highest_num_hashed_tokens:
                highest_num_hashed_tokens = block.num_hashed_tokens

        evicted_block: Optional[PhysicalTokenBlock] = None

        # Find the first block with the lowest timestamp
        for block in least_recent:
            if block.num_hashed_tokens == highest_num_hashed_tokens:
                evicted_block = block
                break

        assert evicted_block is not None

        del self.free_table[evicted_block.block_hash]

        evicted_block.computed = False
        return evicted_block

    def add(self, block: PhysicalTokenBlock):
        self.free_table[block.block_hash] = block

    def remove(self, block_hash: int) -> PhysicalTokenBlock:
        if block_hash not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        block: PhysicalTokenBlock = self.free_table[block_hash]
        del self.free_table[block_hash]
        return block

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


class RandomEvictor(Evictor):
    """Evicts in a first-in-first-out order"""

    def __init__(self):
        self.free_table: Dict[int, PhysicalTokenBlock] = {}

    def __contains__(self, block_hash: int) -> bool:
        return block_hash in self.free_table

    def evict(self) -> PhysicalTokenBlock:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")
        evicted_block = next(iter(self.free_table.values()))
        evicted_block.computed = False
        del self.free_table[evicted_block.block_hash]
        return evicted_block

    def add(self, block: PhysicalTokenBlock):
        self.free_table[block.block_hash] = block

    def remove(self, block_hash: int) -> PhysicalTokenBlock:
        if block_hash not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        block: PhysicalTokenBlock = self.free_table[block_hash]
        del self.free_table[block_hash]
        return block

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


def make_evictor(eviction_policy: EvictionPolicy) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        return LRUEvictor()
    elif eviction_policy == EvictionPolicy.FIFO:
        return RandomEvictor()
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")
