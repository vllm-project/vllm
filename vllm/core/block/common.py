from collections import deque
from typing import Dict, Deque, Iterable, List, Optional, Protocol, Tuple

from vllm.core.block.interfaces import Block, BlockAllocator

BlockId = int
RefCount = int


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

    def record_cow(self, src_block_id: BlockId, trg_block_id: BlockId) -> None:
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

    # TODO: Remove
    # def cow_block_if_not_appendable(self, block: Block) -> BlockId:
    #     """Performs a copy-on-write operation on the given block if it is not
    #     appendable.

    #     This method checks the reference count of the given block. If the
    #     reference count is greater than 1, indicating that the block is shared,
    #     a copy-on-write operation is performed. The original block id is freed,
    #     and a new block id is allocated (which will hold the same content)

    #     Args:
    #         block (Block): The block to check for copy-on-write.

    #     Returns:
    #         BlockId: The block index of the new block if a copy-on-write
    #             operation was performed, or the original block index if
    #             no copy-on-write was necessary.
    #     """
    #     if self._is_appendable(block):
    #         # No CoW needed
    #         return block.block_id

    #     # Perform CoW
    #     src_block_id = block.block_id

    #     # Decrement refcount of the source block id
    #     self._allocator.free_block_id(block)

    #     # Allocate a new target block id
    #     trg_block_id = self._allocator.allocate_block_id()

    #     # Track src => trg block id mapping (for the CoW GPU kernel)
    #     assert src_block_id is not None
    #     assert trg_block_id is not None
    #     self._copy_on_writes.append((src_block_id, trg_block_id))

    #     return trg_block_id

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
                                   token_ids=None,
                                   block_size=self._block_size,
                                   allocator=self._allocator,
                                   block_id=None))

    def increase_pool(self):
        cur_pool_size = self._pool_size
        new_pool_size = cur_pool_size * 2
        self._pool_size = new_pool_size

        self._free_ids += deque(range(cur_pool_size, new_pool_size))

        for i in range(cur_pool_size, new_pool_size):
            self._pool.append(
                self._create_block(prev_block=None,
                                   token_ids=None,
                                   block_size=self._block_size,
                                   allocator=self._allocator,
                                   block_id=None))

    def init_block(self, prev_block: Optional[Block],
                   token_ids: Optional[List[int]], block_size: int,
                   physical_block_id: Optional[int]) -> Block:
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
            block_id=physical_block_id)
        block.pool_id = pool_id  # type: ignore[attr-defined]
        return block

    def free_block(self, block: Block) -> None:
        self._free_ids.appendleft(block.pool_id)  # type: ignore[attr-defined]


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
