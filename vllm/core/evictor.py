import enum
from typing import Dict, List, Optional
from abc import ABC, abstractmethod, abstractproperty

from vllm.block import PhysicalTokenBlock, DEFAULT_LAST_ACCESSED_TIME

class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()
    FIFO = enum.auto()


class Evictor(ABC):
    """
    """

    @abstractmethod
    def evict(self) -> PhysicalTokenBlock:
        pass

    @abstractmethod
    def __contains__(self, block_hash: int) -> bool:
        pass

    @abstractmethod
    def append(self, block: PhysicalTokenBlock):
        pass

    @abstractmethod
    def remove(self, block_hash: int) -> PhysicalTokenBlock:
        pass

    @abstractproperty
    def num_blocks(self) -> int:
        pass


class LRUEvictor(Evictor):
    def __init__(self):
        self.free_table: Dict[int, PhysicalTokenBlock] = {}

    def __contains__(self, block_hash: int) -> bool:
        return block_hash in self.free_table

    def evict(self) -> PhysicalTokenBlock:
        free_blocks: List[PhysicalTokenBlock] = list(self.free_table.values())
        if len(free_blocks) == 0:
            raise ValueError("No usable cache memory left")

        # Find lowest timestamp
        lowest_timestamp = DEFAULT_LAST_ACCESSED_TIME
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

    def append(self, block: PhysicalTokenBlock):
        self.free_table[block.block_hash] = block

    def remove(self, block_hash: int) -> PhysicalTokenBlock:
        if not block_hash in self.free_table:
            raise AssertionError(
            "Attempting to remove block that's not in the evictor") 
        block: PhysicalTokenBlock = self.free_table[block_hash]
        del self.free_table[block_hash]
        return block

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


class FIFOEvictor(Evictor):
    """Evicts in a first-in-first-out order"""
    
    def __init__(self):
        self.free_list: List[PhysicalTokenBlock] = []

    def __contains__(self, block_hash: int) -> bool:
        return any(block_hash == free_block.block_hash
                   for free_block in self.free_list)

    def evict(self) -> PhysicalTokenBlock:
        if len(self.free_list) == 0:
            raise ValueError("No usable cache memory left")
        return self.free_list.popleft()

    def append(self, block: PhysicalTokenBlock):
        self.free_list.append(block)

    def remove(self, block_hash: int) -> PhysicalTokenBlock:
        for free_block in self.free_list:
            if block_hash == free_block.block_hash:
                self.free_list.remove(free_block)
                return free_block
        raise AssertionError(
            "Attempting to remove block that's not in the evictor")
    
    @property
    def num_blocks(self) -> int:
        return len(self.free_list)


def make_evictor(eviction_policy: EvictionPolicy) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        return LRUEvictor()
    elif eviction_policy == EvictionPolicy.FIFO:
        return FIFOEvictor()
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")