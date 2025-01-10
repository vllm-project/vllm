from collections import deque, defaultdict
import copy
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Protocol, Tuple, DefaultDict

from vllm.core.block.interfaces import Block, BlockAllocator

BlockId = int
RefCount = int
SlotMappings = List[int]
TokenMappings = DefaultDict[int, List[int]]
EVICTED_SLOT_ID = -2


class RefCounterProtocol(Protocol):

    def incr(self, block_id: BlockId) -> RefCount:
        raise NotImplementedError

    def decr(self, block_id: BlockId) -> RefCount:
        raise NotImplementedError

    def get(self, block_id: BlockId) -> RefCount:
        raise NotImplementedError


class RefCounter(RefCounterProtocol):
    """A class for managing reference counts for a set of block indices.

    The RefCounter class maintains a dictionary that maps block indices to their
    corresponding reference counts. It provides methods to increment, decrement,
    and retrieve the reference count for a given block index.

    Args:
        all_block_indices (Iterable[BlockId]): An iterable of block indices
            to initialize the reference counter with.
    """

    def __init__(self, all_block_indices: Iterable[BlockId]):
        deduped = set(all_block_indices)
        self._refcounts: Dict[BlockId,
                              RefCount] = {index: 0
                                           for index in deduped}

    def incr(self, block_id: BlockId) -> RefCount:
        assert block_id in self._refcounts
        pre_incr_refcount = self._refcounts[block_id]

        assert pre_incr_refcount >= 0

        post_incr_refcount = pre_incr_refcount + 1
        self._refcounts[block_id] = post_incr_refcount
        return post_incr_refcount

    def decr(self, block_id: BlockId) -> RefCount:
        assert block_id in self._refcounts
        refcount = self._refcounts[block_id]

        assert refcount > 0
        refcount -= 1

        self._refcounts[block_id] = refcount

        return refcount

    def get(self, block_id: BlockId) -> RefCount:
        assert block_id in self._refcounts
        return self._refcounts[block_id]

    def as_readonly(self) -> "ReadOnlyRefCounter":
        return ReadOnlyRefCounter(self)


class ReadOnlyRefCounter(RefCounterProtocol):
    """A read-only view of the RefCounter class.

    The ReadOnlyRefCounter class provides a read-only interface to access the
    reference counts maintained by a RefCounter instance. It does not allow
    modifications to the reference counts.

    Args:
        refcounter (RefCounter): The RefCounter instance to create a read-only
            view for.
    """

    def __init__(self, refcounter: RefCounter):
        self._refcounter = refcounter

    def incr(self, block_id: BlockId) -> RefCount:
        raise ValueError("Incr not allowed")

    def decr(self, block_id: BlockId) -> RefCount:
        raise ValueError("Decr not allowed")

    def get(self, block_id: BlockId) -> RefCount:
        return self._refcounter.get(block_id)


class CopyOnWriteTracker:
    """A class for tracking and managing copy-on-write operations for blocks.

    The CopyOnWriteTracker class maintains a mapping of source block indices to
        their corresponding copy-on-write destination block indices. It works in
        conjunction with a RefCounter.

    Args:
        refcounter (RefCounter): The reference counter used to track block
            reference counts.
    """

    def __init__(self, refcounter: RefCounterProtocol):
        self._copy_on_writes: List[Tuple[BlockId, BlockId]] = []
        self._refcounter = refcounter

    def is_appendable(self, block: Block) -> bool:
        """Checks if the block is shared or not. If shared, then it cannot
        be appended and needs to be duplicated via copy-on-write
        """
        block_id = block.block_id
        if block_id is None:
            return True

        refcount = self._refcounter.get(block_id)
        return refcount <= 1

    def record_cow(self, src_block_id: Optional[BlockId],
                   trg_block_id: Optional[BlockId]) -> None:
        """Records a copy-on-write operation from source to target block id
        Args:
            src_block_id (BlockId): The source block id from which to copy 
                the data
            trg_block_id (BlockId): The target block id to which the data
                is copied
        """
        assert src_block_id is not None
        assert trg_block_id is not None
        self._copy_on_writes.append((src_block_id, trg_block_id))

    def clear_cows(self) -> List[Tuple[BlockId, BlockId]]:
        """Clears the copy-on-write tracking information and returns the current
        state.

        This method returns a list mapping source block indices to
         destination block indices for the current copy-on-write operations.
        It then clears the internal tracking information.

        Returns:
            List[Tuple[BlockId, BlockId]]: A list mapping source
                block indices to destination block indices for the
                current copy-on-write operations.
        """
        cows = self._copy_on_writes
        self._copy_on_writes = []
        return cows


class BlockPool:
    """Used to pre-allocate block objects, in order to avoid excessive python
    object allocations/deallocations.
    The pool starts from "pool_size" objects and will increase to more objects
    if necessary

    Note that multiple block objects may point to the same physical block id,
    which is why this pool is needed, so that it will be easier to support
    prefix caching and more complicated sharing of physical blocks.
    """

    def __init__(self, block_size: int, create_block: Block.Factory,
                 allocator: BlockAllocator, pool_size: int):
        self._block_size = block_size
        self._create_block = create_block
        self._allocator = allocator
        self._pool_size = pool_size
        assert self._pool_size >= 0

        self._free_ids: Deque[int] = deque(range(self._pool_size))
        self._pool = []
        for i in range(self._pool_size):
            self._pool.append(
                self._create_block(prev_block=None,
                                   token_ids=[],
                                   block_size=self._block_size,
                                   allocator=self._allocator,
                                   block_id=None,
                                   extra_hash=None))

    def increase_pool(self):
        """Doubles the internal pool size
        """
        cur_pool_size = self._pool_size
        new_pool_size = cur_pool_size * 2
        self._pool_size = new_pool_size

        self._free_ids += deque(range(cur_pool_size, new_pool_size))

        for i in range(cur_pool_size, new_pool_size):
            self._pool.append(
                self._create_block(prev_block=None,
                                   token_ids=[],
                                   block_size=self._block_size,
                                   allocator=self._allocator,
                                   block_id=None,
                                   extra_hash=None))

    def init_block(self,
                   prev_block: Optional[Block],
                   token_ids: List[int],
                   block_size: int,
                   physical_block_id: Optional[int],
                   extra_hash: Optional[int] = None) -> Block:
        if len(self._free_ids) == 0:
            self.increase_pool()
            assert len(self._free_ids) > 0

        pool_id = self._free_ids.popleft()

        block = self._pool[pool_id]
        block.__init__(  # type: ignore[misc]
            prev_block=prev_block,
            token_ids=token_ids,
            block_size=block_size,
            allocator=block._allocator,  # type: ignore[attr-defined] 
            block_id=physical_block_id,
            extra_hash=extra_hash)
        block.pool_id = pool_id  # type: ignore[attr-defined]
        return block

    def free_block(self, block: Block) -> None:
        self._free_ids.appendleft(block.pool_id)  # type: ignore[attr-defined]


class VirtualBlockTable:
    """VirtualBlockTable maintains the mappings between tokens and physical
    memory blocks. Both the token mappings and slot mappings are tracked
    respectively, one maps physical blocks to tokens, and the other maps
    tokens to their memory slots. The slot mappings are updated with the tokens
    added and this avoids the need to reconstruct the mappings on every
    iteration for kv copy to the cache.
    """

    def __init__(
        self,
        block_size: int,
        slot_mappings: Optional[SlotMappings] = None,
        token_mappings: Optional[TokenMappings] = None
    ):
        self._block_size = block_size
        if slot_mappings is None:
            slot_mappings = []
        self._slot_mappings: SlotMappings = slot_mappings
        if token_mappings is None:
            token_mappings = (
                defaultdict(lambda: [EVICTED_SLOT_ID] * self._block_size))
        self._token_mappings: TokenMappings = token_mappings

    def append_tokens(
        self,
        blocks: List[Block],
        num_new_tokens: int,
        evicted: bool = False
    ) -> None:
        first_chunk_size = self._block_size - (self.num_tokens %
                                               self._block_size)
        if first_chunk_size < num_new_tokens:
            last_chunk_size = (self.num_tokens + num_new_tokens) % self._block_size
            num_middle_chunks = ((num_new_tokens - first_chunk_size) //
                                 self._block_size)
            middle_chunk_sizes = [(0, self._block_size)] * num_middle_chunks
            chunk_list = [(self._block_size - first_chunk_size, first_chunk_size)]
            chunk_list.extend(middle_chunk_sizes)
            if last_chunk_size > 0:
                chunk_list.append((0, last_chunk_size))
        else:
            chunk_list = [(self._block_size - first_chunk_size, num_new_tokens)]
        assert len(chunk_list) == len(blocks)

        for (slot_offset, chunk_size), block in zip(chunk_list, blocks):
            if not evicted:
                block_token_mappings = self._token_mappings[block.block_id]
                block_token_mappings[slot_offset:slot_offset+chunk_size] = (
                    range(self.num_tokens, self.num_tokens+chunk_size))

                slot_start = block.block_id * self._block_size + slot_offset
                slot_end = slot_start + chunk_size
                slots = range(slot_start, slot_end)
            else:
                slots = [EVICTED_SLOT_ID] * chunk_size
            self._slot_mappings.extend(slots)

    def insert_tokens(
        self,
        block: Block,
        slot_offset: int,
        num_new_tokens: int
    ) -> None:
        # If evicting previous tokens from physical blocks, replace them with
        # new tokens, and update the slot_mappings
        block_slot_start = slot_offset
        block_slot_end = slot_offset + num_new_tokens
        block_token_mappings = self._token_mappings[block.block_id]
        for token_idx in block_token_mappings[block_slot_start:block_slot_end]:
            if token_idx != EVICTED_SLOT_ID:
                self._slot_mappings[token_idx] = EVICTED_SLOT_ID

        # Populate new tokens in the physical blocks, and update both
        # token_mappings and slot_mappings
        slot_start = (block.block_id * self._block_size) + slot_offset
        slot_end = slot_start + num_new_tokens
        self._slot_mappings.extend(range(slot_start, slot_end))

        token_start = self.num_tokens - num_new_tokens
        block_token_mappings[block_slot_start:block_slot_end] = range(
            token_start, self.num_tokens)

    def reset(self):
        self._slot_mappings = []
        self._token_mappings = defaultdict(
            lambda: [EVICTED_SLOT_ID] * self._block_size)

    @property
    def num_tokens(self) -> int:
        return len(self._slot_mappings)

    @property
    def slot_mappings(self) -> SlotMappings:
        return self._slot_mappings

    @property
    def token_mappings(self) -> TokenMappings:
        return self._token_mappings

    def fork(self) -> "VirtualBlockTable":
        return VirtualBlockTable(self._block_size,
                                 self._slot_mappings.copy(),
                                 copy.deepcopy(self._token_mappings))


class PhysicalBlockTable:
    """PhysicalBlockTable (formerly BlockList) keeps track of the allocated
    cache blocks. It is also an optimization to allow fast-access to physical
    block ids. It maintains a block id list that is updated with the block
    list and this avoids the need to reconstruct the block id list on every
    iteration of the block manager.
    """

    def __init__(self, blocks: Optional[List[Block]] = None):
        if blocks is None:
            blocks = []
        self._blocks: List[Block] = []
        self._block_ids: List[int] = []

        self.update(blocks)

    def update(self, blocks: List[Block]):
        self._blocks = blocks.copy()
        # Cache block ids for fast query
        self._block_ids = []

        for block in self._blocks:
            self._add_block_id(block.block_id)

    def append_tokens(self, block_index: int, token_ids: List[int]) -> Block:
        block = self._blocks[block_index]
        prev_block_id = block.block_id

        block.append_token_ids(token_ids)

        # CoW or promotion may update the internal block_id
        if prev_block_id != block.block_id:
            self._update_block_id(block_index, block.block_id)

        return block

    def insert_tokens(
        self,
        block_index: int,
        slot_offset: int,
        token_ids: List[int]
    ) -> Block:
        block = self._blocks[block_index]
        prev_block_id = block.block_id

        block.insert_token_ids(slot_offset, token_ids)

        # CoW or promotion may update the internal block_id
        if prev_block_id != block.block_id:
            self._update_block_id(block_index, block.block_id)

        return block

    def append(self, new_block: Block):
        self._blocks.append(new_block)
        self._add_block_id(new_block.block_id)

    def reset(self):
        self._blocks = []
        self._block_ids = []

    def list(self) -> List[Block]:
        return self._blocks

    def ids(self) -> List[int]:
        return self._block_ids

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, block_index: int) -> Block:
        return self._blocks[block_index]

    def __setitem__(self, block_index: int, new_block: Block) -> None:
        self._blocks[block_index] = new_block
        self._update_block_id(block_index, new_block.block_id)

    def _add_block_id(self, block_id: BlockId) -> None:
        assert block_id is not None
        self._block_ids.append(block_id)

    def _update_block_id(self, block_index: int,
                         new_block_id: Optional[BlockId]) -> None:
        assert new_block_id is not None
        self._block_ids[block_index] = new_block_id


@dataclass
class CacheMetricData:
    """A utility dataclass to maintain cache metric.
    To avoid overflow, we maintain the hit rate in block granularity, so that
    we can maintain a single hit rate for n_completed_block x block_size,
    and calculate the real time hit rate by the following:
    BS = The number of queries per block.
    nB = The number of completed blocks.
    HR = hit rate of (nB x BS) queries.
    Q = current number of queries (< BS).
    H = current number of hits (< BS).
    hit rate = ((HR x nB) + (H / Q) x (Q / BS)) / (nB + Q / BS)
    """
    num_completed_blocks: int = 0
    completed_block_cache_hit_rate: float = 0.0
    num_incompleted_block_queries: int = 0
    num_incompleted_block_hit: int = 0
    block_size: int = 1000

    def query(self, hit: bool):
        self.num_incompleted_block_queries += 1
        self.num_incompleted_block_hit += 1 if hit else 0

        # When a block is completed, update the cache hit rate
        # and reset the incomplete numbers.
        if self.num_incompleted_block_queries == self.block_size:
            hit_rate = (self.num_incompleted_block_hit /
                        self.num_incompleted_block_queries)
            self.completed_block_cache_hit_rate = (
                self.completed_block_cache_hit_rate * self.num_completed_blocks
                + hit_rate) / (self.num_completed_blocks + 1)
            self.num_incompleted_block_queries = 0
            self.num_incompleted_block_hit = 0
            self.num_completed_blocks += 1

    def get_hit_rate(self):
        incomplete_ratio = self.num_incompleted_block_queries / self.block_size
        total_blocks = self.num_completed_blocks + incomplete_ratio
        if total_blocks == 0:
            return 0.0

        completed_block_hit, incompleted_block_hit = 0.0, 0.0
        if self.num_completed_blocks > 0:
            completed_block_hit = (self.completed_block_cache_hit_rate *
                                   self.num_completed_blocks)
        if self.num_incompleted_block_queries > 0:
            incompleted_hit_rate = (self.num_incompleted_block_hit /
                                    self.num_incompleted_block_queries)
            incompleted_block_hit = (incompleted_hit_rate * incomplete_ratio)
        return (completed_block_hit + incompleted_block_hit) / total_blocks


def get_all_blocks_recursively(last_block: Block) -> List[Block]:
    """Retrieves all the blocks in a sequence starting from the last block.

    This function recursively traverses the sequence of blocks in reverse order,
    starting from the given last block, and returns a list of all the blocks in
    the sequence.

    Args:
        last_block (Block): The last block in the sequence.

    Returns:
        List[Block]: A list of all the blocks in the sequence, in the order they
            appear.
    """

    def recurse(block: Block, lst: List[Block]) -> None:
        if block.prev_block is not None:
            recurse(block.prev_block, lst)
        lst.append(block)

    all_blocks: List[Block] = []
    recurse(last_block, all_blocks)
    return all_blocks
