from typing import FrozenSet, Iterable, List, Optional, Set, Tuple

from vllm.core.block.common import (CopyOnWriteTracker, RefCounter,
                                    get_all_blocks_recursively)
from vllm.core.block.interfaces import Block, BlockAllocator, BlockId, Device
from vllm.utils import cdiv

Refcount = int


class NaiveBlockAllocator(BlockAllocator):
    """A simple block allocator that manages blocks of memory without prefix
    caching.

    Args:
        create_block (Block.Factory): A factory function for creating new
            blocks. This is used when a NaiveBlockAllocator is composed within
            a prefix caching allocator -- the naive block allocator must
            construct prefix caching blocks (but shouldn't know anything else
            about them).
        num_blocks (int): The total number of blocks to manage.
        block_size (int): The size of each block in tokens.
        block_ids (Optional[Iterable[int]], optional): An optional iterable of
            block IDs. If not provided, block IDs will be assigned sequentially
            from 0 to num_blocks - 1.
    """

    def __init__(
        self,
        create_block: Block.Factory,
        num_blocks: int,
        block_size: int,
        block_ids: Optional[Iterable[int]] = None,
    ):
        if block_ids is None:
            block_ids = range(num_blocks)

        self._free_block_indices: Set[BlockId] = set(block_ids)
        self._all_block_indices = frozenset(block_ids)
        assert len(self._all_block_indices) == num_blocks

        self._refcounter = RefCounter(
            all_block_indices=self._free_block_indices)
        self._create_block = create_block
        self._block_size = block_size

        self._cow_tracker = CopyOnWriteTracker(
            refcounter=self._refcounter.as_readonly(),
            allocator=self,
        )

    def allocate_immutable(self,
                           prev_block: Optional[Block],
                           token_ids: List[int],
                           device: Optional[Device] = None) -> Block:
        """Allocates a new immutable block with the given token IDs, linked to
        the previous block.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence. If
                None, then the block to be allocated is the first block in the
                sequence.
            token_ids (List[int]): The token IDs to be stored in the new block.

        Returns:
            Block: The newly allocated immutable block.
        """
        assert device is None
        block = self.allocate_mutable(prev_block=prev_block)
        block.append_token_ids(token_ids)
        return block

    def allocate_mutable(self,
                         prev_block: Optional[Block],
                         device: Optional[Device] = None) -> Block:
        """Allocates a new mutable block, linked to the previous block.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence. If
                None, then the block to be allocated is the first block in the
                sequence.

        Returns:
            Block: The newly allocated mutable block.
        """
        assert device is None
        block_id = self._allocate_new_block_id()
        return self._create_block(
            prev_block=prev_block,
            token_ids=[],
            block_id=block_id,
            block_size=self._block_size,
            allocator=self,
        )

    def free(self, block: Block) -> None:
        assert block.block_id is not None
        self._free_block_id(block.block_id)
        block.block_id = None

    def fork(self, last_block: Block) -> List[Block]:
        """Creates a new sequence of blocks that shares the same underlying
        memory as the original sequence.

        Args:
            last_block (Block): The last block in the original sequence.

        Returns:
            List[Block]: The new sequence of blocks that shares the same memory
                as the original sequence.
        """
        source_blocks = get_all_blocks_recursively(last_block)

        forked_blocks = []
        prev_block = None
        for block in source_blocks:

            # Increment refcount for each block.
            assert block.block_id is not None
            refcount = self._refcounter.incr(block.block_id)
            assert refcount != 1, "can't fork free'd block"

            forked_blocks.append(
                self._create_block(
                    prev_block=prev_block,
                    token_ids=block.token_ids,
                    block_id=block.block_id,
                    block_size=self._block_size,
                    allocator=self,
                ))
            prev_block = forked_blocks[-1]

        return forked_blocks

    def get_num_free_blocks(self) -> int:
        return len(self._free_block_indices)

    def get_num_total_blocks(self) -> int:
        return len(self._all_block_indices)

    def _allocate_new_block_id(self) -> BlockId:
        if not self._free_block_indices:
            raise BlockAllocator.NoFreeBlocksError()

        block_id = next(iter(self._free_block_indices))
        self._refcounter.incr(block_id)
        self._free_block_indices.remove(block_id)
        return block_id

    def _free_block_id(self, block_id: BlockId) -> None:
        refcount = self._refcounter.decr(block_id)
        if refcount == 0:
            self._free_block_indices.add(block_id)

    def get_physical_block_id(self, absolute_id: int) -> int:
        """Returns the zero-offset block id on certain block allocator
        given the absolute block id.

        Args:
            absolute_id (int): The absolute block id for the block 
            in whole allocator.

        Returns:
            int: The zero-offset block id on certain device.
        """
        return sorted(self._all_block_indices).index(absolute_id)

    @property
    def refcounter(self):
        return self._refcounter

    @property
    def all_block_ids(self) -> FrozenSet[int]:
        return self._all_block_indices

    def cow_block_if_not_appendable(self, block: Block) -> Optional[BlockId]:
        """Performs a copy-on-write operation on the given block if it is not
        appendable.

        Args:
            block (Block): The block to check for copy-on-write.

        Returns:
            Optional[BlockId]: The block index of the new block if a copy-on
                -write operation was performed, or the original block index if
                no copy-on-write was necessary.
        """
        return self._cow_tracker.cow_block_if_not_appendable(block)

    def clear_copy_on_writes(self) -> List[Tuple[BlockId, BlockId]]:
        """Returns the copy-on-write source->destination mapping and clears it.

        Returns:
            List[Tuple[BlockId, BlockId]]: A list mapping source
                block indices to destination block indices.
        """
        return self._cow_tracker.clear_cows()

    def mark_blocks_as_accessed(self, block_ids: List[int],
                                now: float) -> None:
        """Mark blocks as accessed, used in prefix caching.

        Since the naive allocator does not implement prefix caching, we do
        nothing.
        """
        pass

    def mark_blocks_as_computed(self, block_ids: List[int]) -> None:
        """Mark blocks as computed, used in prefix caching.

        Since the naive allocator does not implement prefix caching, we do
        nothing.
        """
        pass

    def get_common_computed_block_ids(
            self, seq_block_ids: List[List[int]]) -> List[int]:
        """Determine blocks that can be skipped in prefill.

        Since the naive allocator does not support prefix caching, always return
        an empty list.
        """
        return []

    def promote_to_immutable_block(self, block: Block) -> BlockId:
        raise NotImplementedError

    def get_num_blocks_touched(self,
                               blocks: List[Block],
                               num_lookahead_slots: int = 0) -> int:
        """Determine the number of blocks that will be touched by
        swapping in/out the given blocks from certain sequence
        group with the provided num_lookahead_slots.

        Args:
            blocks (List[Block]): The potential blocks to swap.
            num_lookahead_slots (int): number of lookahead slots (0 for swap 
                out).
        
        Returns:
            int: the number of blocks that will be touched by
                swapping in/out the given blocks and num_lookahead_slots.
        """
        # NOTE: for naive block, we use set to eliminate common blocks among
        # seqs, also we compare the empty slots in the mutable blocks with
        # lookahead slots to get the number of unique new block that are
        # needed.
        old_block_set = set()
        new_block_count = 0
        # TODO(cade): make sure the logic is correct and clean it up.
        for block in blocks:
            if not block.is_full and num_lookahead_slots != 0:
                if block.num_empty_slots >= num_lookahead_slots:
                    new_block_count += 1
                else:
                    new_block_count += cdiv(
                        num_lookahead_slots - block.num_empty_slots,
                        self._block_size)
            else:
                old_block_set.add(block.block_id)
        num_touched_blocks = new_block_count + len(old_block_set)
        return num_touched_blocks

    def swap_out(self, blocks: List[Block]) -> None:
        for block in blocks:
            self.free(block)

    def swap_in(self, blocks: List[Block]) -> None:
        for block in blocks:
            if block.is_full:
                alloc = self.allocate_immutable(block.prev_block,
                                                block.token_ids)
            else:
                alloc = self.allocate_mutable(block.prev_block)
                alloc.append_token_ids(block.token_ids)
            block.block_id = alloc.block_id


class NaiveBlock(Block):
    """An implementation of the Block class that does not support prefix
    caching.

    The NaiveBlock class represents a block of token IDs with a fixed size. It
    provides methods for appending token IDs to the block and manages copy-on
    -write operations when necessary.

    Args:
        prev_block (Block): The previous block in the sequence.
        token_ids (List[int]): The initial token IDs to be stored in the block.
        block_size (int): The maximum number of token IDs that can be stored in
            the block.
        allocator (BlockAllocator): The block allocator associated with this
            block.
        block_id (Optional[int], optional): The physical block index
            of this block. Defaults to None, which means no allocation has been
            made.
        _cow_target (Optional[Block], optional): The copy-on-write target block.
            If not provided, it defaults to self.
    """

    def __init__(self,
                 prev_block: Optional[Block],
                 token_ids: List[int],
                 block_size: int,
                 allocator: BlockAllocator,
                 block_id: Optional[int] = None,
                 _cow_target: Optional[Block] = None):
        self._token_ids: List[int] = []
        self._block_size = block_size
        self._prev_block = prev_block
        self._block_id = block_id
        self._allocator = allocator
        self._cow_target = _cow_target if _cow_target is not None else self

        self._append_token_ids_no_cow(token_ids)

    def append_token_ids(self, token_ids: List[int]) -> None:
        """Appends the given token IDs to the block, instructing the allocator
        to perform a copy-on-write if necessary.

        Args:
            token_ids (List[int]): The token IDs to be appended to the block.
        """
        self._append_token_ids_no_cow(token_ids)

        if self._block_id is not None:
            self._block_id = (self._allocator.cow_block_if_not_appendable(
                self._cow_target))

    def _append_token_ids_no_cow(self, token_ids: List[int]) -> None:
        assert self.num_empty_slots >= len(token_ids)
        self._token_ids.extend(token_ids)

    @property
    def computed(self) -> bool:
        raise NotImplementedError

    @computed.setter
    def computed(self, value) -> None:
        raise NotImplementedError

    @property
    def last_accessed(self) -> float:
        raise NotImplementedError

    @last_accessed.setter
    def last_accessed(self, last_accessed_ts: float):
        raise NotImplementedError

    @property
    def block_id(self) -> Optional[int]:
        return self._block_id

    @block_id.setter
    def block_id(self, value: Optional[int]) -> None:
        self._block_id = value

    @property
    def is_full(self) -> bool:
        return self.num_empty_slots == 0

    @property
    def num_empty_slots(self) -> int:
        return self._block_size - len(self._token_ids)

    @property
    def token_ids(self) -> List[int]:
        return self._token_ids

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def prev_block(self) -> Optional["Block"]:
        return self._prev_block

    @property
    def content_hash(self) -> Optional[int]:
        return None
