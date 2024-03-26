from typing import List, Iterable, Dict, Optional
from collections import defaultdict

from vllm.core.block.interfaces import Block, BlockAllocator

BlockIndex = int
RefCount = int


class RefCounter:

    def __init__(self, all_block_indices: Iterable[BlockIndex]):
        deduped = set(all_block_indices)
        self._refcounts: Dict[BlockIndex,
                              RefCount] = {index: 0
                                           for index in deduped}

    def incr(self, block_index: BlockIndex) -> RefCount:
        assert block_index in self._refcounts
        pre_incr_refcount = self._refcounts[block_index]

        assert pre_incr_refcount >= 0

        post_incr_refcount = pre_incr_refcount + 1
        self._refcounts[block_index] = post_incr_refcount
        return post_incr_refcount

    def decr(self, block_index: BlockIndex) -> RefCount:
        assert block_index in self._refcounts
        refcount = self._refcounts[block_index]

        assert refcount > 0
        refcount -= 1

        self._refcounts[block_index] = refcount

        return refcount

    def get(self, block_index: BlockIndex) -> RefCount:
        assert block_index in self._refcounts
        return self._refcounts[block_index]

    def as_readonly(self) -> "ReadOnlyRefCounter":
        return ReadOnlyRefCounter(self)


class ReadOnlyRefCounter:
    def __init__(self, refcounter: RefCounter):
        self._refcounter = refcounter

    def incr(self, block_index: BlockIndex) -> RefCount:
        raise ValueError("Incr not allowed")

    def decr(self, block_index: BlockIndex) -> RefCount:
        raise ValueError("Decr not allowed")

    def get(self, block_index: BlockIndex) -> RefCount:
        return self._refcounter.get(block_index)


class CopyOnWriteTracker:

    def __init__(
        self,
        refcounter: RefCounter,
        allocator: BlockAllocator,
    ):
        self._copy_on_writes = defaultdict(list)
        self._refcounter = refcounter
        self._allocator = allocator

    def cow_block_if_not_appendable(self,
                                    block: Block) -> Optional[BlockIndex]:
        block_index = block.physical_block_index
        if block_index is None:
            return block_index

        refcount = self._refcounter.get(block_index)
        assert refcount != 0
        if refcount > 1:
            block_index = self._copy_on_write(block, block_index)

        return block_index

    def _copy_on_write(self, block: Block,
                       src_block_index: BlockIndex) -> BlockIndex:
        # Decrement refcount of the old block.
        self._allocator.free(block)

        # Allocate a fresh new block.
        dst_block_index = self._allocator.allocate_mutable(
            prev_block=block.prev_block).physical_block_index

        # Track src/dst copy.
        self._copy_on_writes[src_block_index].append(dst_block_index)

        return dst_block_index


    def clear_cows(self) -> Dict[BlockIndex, List[BlockIndex]]:
        cows = dict(self._copy_on_writes)
        self._copy_on_writes.clear()
        return cows


def get_all_blocks_recursively(last_block: Block) -> List[Block]:

    def recurse(block: Block, lst: List[Block]) -> None:
        if block.prev_block is not None:
            recurse(block.prev_block, lst)
        lst.append(block)

    all_blocks = []
    recurse(last_block, all_blocks)
    return all_blocks
