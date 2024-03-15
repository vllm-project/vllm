import enum
from typing import Dict, List, Optional
from abc import ABC, abstractmethod, abstractproperty
from sortedcontainers import SortedList

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


class BlockMetaInfo:
    """PhysicalTokenBlock's block_hash & num_hashed_tokens & last_accessed
    are stored in class BlockMetaInfo and sorted by SortedList
    """

    def __init__(
        self,
        block_hash,
        num_hashed_tokens,
        last_accessed,
    ) -> None:
        self.block_hash = block_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed

    def __lt__(self, other):
        if self.last_accessed == other.last_accessed:
            return self.num_hashed_tokens > other.num_hashed_tokens
        return self.last_accessed < other.last_accessed

    def __eq__(self, other):
        return self.block_hash == other.block_hash

    def __repr__(self) -> str:
        return (f'BlockMetaInfo(block_hash={self.block_hash}, '
                f'num_hashed_tokens={self.num_hashed_tokens}, '
                f'last_accessed={self.last_accessed}')


class LRUEvictor(Evictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the PhysicalTokenBlock. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted. If two blocks each have the lowest last_accessed time and
    highest num_hashed_tokens value, then one will be chose arbitrarily
    """

    def __init__(self):
        self.free_table: Dict[int, PhysicalTokenBlock] = {}
        self.sorted_list: SortedList[BlockMetaInfo] = SortedList()

    def __contains__(self, block_hash: int) -> bool:
        return block_hash in self.free_table

    def evict(self) -> PhysicalTokenBlock:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        evicted_block_metainfo = self.sorted_list.pop(0)
        evicted_block = self.free_table[evicted_block_metainfo.block_hash]

        assert evicted_block is not None

        del self.free_table[evicted_block.block_hash]

        evicted_block.computed = False
        return evicted_block

    def add(self, block: PhysicalTokenBlock):
        self.free_table[block.block_hash] = block
        self.sorted_list.add(
            BlockMetaInfo(block_hash=block.block_hash,
                          num_hashed_tokens=block.num_hashed_tokens,
                          last_accessed=block.last_accessed))

    def remove(self, block_hash: int) -> PhysicalTokenBlock:
        if block_hash not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        block: PhysicalTokenBlock = self.free_table[block_hash]
        del self.free_table[block_hash]
        self.sorted_list.remove(
            BlockMetaInfo(block_hash=block.block_hash,
                          num_hashed_tokens=block.num_hashed_tokens,
                          last_accessed=block.last_accessed))
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
