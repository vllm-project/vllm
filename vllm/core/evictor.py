import enum
from typing import Dict
# from typing import List, Optional
from abc import ABC, abstractmethod, abstractproperty
from vllm.utils import Device

from vllm.block import PhysicalTokenBlock, BlockTable


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
        evictor. Caller is responsible for making sure that block_hash is contained
        in the evictor before calling remove. Should be used to "bring back" blocks
        that have been freed but not evicted yet.
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
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")
        free_blocks = self.free_table.values()

        # # Find lowest timestamp
        # lowest_timestamp = next(iter(free_blocks)).last_accessed
        # # Find highest prefix count per block
        # highest_num_hashed_tokens = 0
        # Get evicted block
        evicted_block: PhysicalTokenBlock = next(iter(free_blocks))

        for block in free_blocks:
            if block.last_accessed < evicted_block.last_accessed or block.last_accessed == evicted_block.last_accessed and block.num_hashed_tokens > evicted_block.num_hashed_tokens:
                evicted_block = block

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

    def __init__(self, device: Device, block_size: int, num_blocks: int):
        self.free_table: BlockTable = []
        # reserve(self.free_table, num_blocks)
        for i in range(num_blocks):
            block = PhysicalTokenBlock(device=device,
                                       block_number=i,
                                       block_size=block_size,
                                       block_hash=-1,
                                       num_hashed_tokens=0)
            self.free_table.append(block)

    def __contains__(self, block_hash: int) -> bool:
        return any(b.block_hash == block_hash for b in self.free_table)

    def evict(self) -> PhysicalTokenBlock:
        if not self.free_table:
            raise ValueError("No usable cache memory left")
        evicted_block = self.free_table.pop()
        evicted_block.computed = False
        return evicted_block

    def add(self, block: PhysicalTokenBlock):
        self.free_table.append(block)

    def remove(self, block_hash: int) -> PhysicalTokenBlock:
        new_table = [b for b in self.free_table if b.block_hash != block_hash]
        self.free_table = new_table

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


def make_evictor(eviction_policy: EvictionPolicy, device: Device,
                 block_size: int, num_blocks: int) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        return LRUEvictor()
    elif eviction_policy == EvictionPolicy.FIFO:
        return RandomEvictor(device, block_size, num_blocks)
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")
