import enum
import heapq
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

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

    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, PhysicalTokenBlock] = {}
        self.priority_queue = []

    def __contains__(self, block_hash: int) -> bool:
        return block_hash in self.free_table

    def evict(self) -> PhysicalTokenBlock:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            # Lazy deletion algorithm is applied.
            last_accessed, _, block_hash = heapq.heappop(self.priority_queue)
            if (block_hash in self.free_table
                    and self.free_table[block_hash].last_accessed
                    == last_accessed):
                evicted_block = self.free_table.pop(block_hash)
                evicted_block.computed = False
                return evicted_block

        raise ValueError("No usable cache memory left")

    def add(self, block: PhysicalTokenBlock):
        self.free_table[block.block_hash] = block
        heapq.heappush(
            self.priority_queue,
            (block.last_accessed, -block.num_hashed_tokens, block.block_hash))
        self._cleanup_if_necessary()

    def remove(self, block_hash: int) -> PhysicalTokenBlock:
        if block_hash not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        block: PhysicalTokenBlock = self.free_table.pop(block_hash)
        return block

    def _cleanup_if_necessary(self):
        if len(self.priority_queue) > LRUEvictor.CLEANUP_THRESHOLD * len(
                self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[float, int, int]] = []

        for block_hash, block in self.free_table.items():
            new_priority_queue.append(
                (block.last_accessed, -block.num_hashed_tokens,
                 block.block_hash))
        heapq.heapify(new_priority_queue)

        self.priority_queue = new_priority_queue

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


def make_evictor(eviction_policy: EvictionPolicy) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        return LRUEvictor()
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")
