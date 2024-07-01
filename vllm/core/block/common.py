from typing import List, Optional, Protocol, Tuple

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
    """A class for managing reference counts for a range of block indices.

    The RefCounter class maintains a list of reference counts.
    It provides methods to increment, decrement,
    and retrieve the reference count for a given block index in that range.

    Args:
        block_index_start (BlockId): The starting block index for the reference
            counter.
        block_index_end (BlockId): The ending block index for the reference
            counter.
    """

    def __init__(self, block_index_start: BlockId, block_index_end: BlockId):
        self.block_index_start = block_index_start
        self.block_index_end = block_index_end
        self._refcounts = [0] * (block_index_end - block_index_start)

    def incr(self, block_id: BlockId) -> RefCount:
        assert self.block_index_start <= block_id < self.block_index_end
        idx = block_id - self.block_index_start
        self._refcounts[idx] += 1
        answer = self._refcounts[idx]
        assert answer > 0
        return answer

    def decr(self, block_id: BlockId) -> RefCount:
        assert self.block_index_start <= block_id < self.block_index_end
        idx = block_id - self.block_index_start
        self._refcounts[idx] -= 1
        answer = self._refcounts[idx]
        assert answer >= 0
        return answer

    def get(self, block_id: BlockId) -> RefCount:
        assert self.block_index_start <= block_id < self.block_index_end
        idx = block_id - self.block_index_start
        answer = self._refcounts[idx]
        return answer

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
        conjunction with a RefCounter and a BlockAllocator to handle reference
        counting and block allocation.

    Args:
        refcounter (RefCounter): The reference counter used to track block
            reference counts.
        allocator (BlockAllocator): The block allocator used to allocate and
            free blocks.
    """

    def __init__(
        self,
        refcounter: RefCounterProtocol,
        allocator: BlockAllocator,
    ):
        self._copy_on_writes: List[Tuple[BlockId, BlockId]] = []
        self._refcounter = refcounter
        self._allocator = allocator

    def cow_block_if_not_appendable(self, block: Block) -> Optional[BlockId]:
        """Performs a copy-on-write operation on the given block if it is not
        appendable.

        This method checks the reference count of the given block. If the
        reference count is greater than 1, indicating that the block is shared,
        a copy-on-write operation is performed. The original block is freed,
        and a new block is allocated with the same content. The new block index
        is returned.

        Args:
            block (Block): The block to check for copy-on-write.

        Returns:
            Optional[BlockId]: The block index of the new block if a copy-on
                -write operation was performed, or the original block index if
                no copy-on-write was necessary.
        """
        block_id = block.block_id
        if block_id is None:
            return block_id

        refcount = self._refcounter.get(block_id)
        assert refcount != 0
        if refcount > 1:
            src_block_id = block_id
            # Decrement refcount of the old block.
            self._allocator.free(block)

            # Allocate a fresh new block.
            block_id = self._allocator.allocate_mutable(
                prev_block=block.prev_block).block_id

            # Track src/dst copy.
            assert src_block_id is not None
            assert block_id is not None
            self._copy_on_writes.append((src_block_id, block_id))

        return block_id

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
