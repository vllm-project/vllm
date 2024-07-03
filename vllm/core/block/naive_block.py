from collections import deque
from typing import Deque, FrozenSet, Iterable, List, Optional, Tuple

from vllm.core.block.common import (BlockPool, CopyOnWriteTracker, RefCounter,
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
        block_pool: Optional[BlockPool] = None,
    ):
        if block_ids is None:
            block_ids = range(num_blocks)

        self._free_block_indices: Deque[BlockId] = deque(block_ids)
        self._all_block_indices = frozenset(block_ids)
        assert len(self._all_block_indices) == num_blocks

        self._refcounter = RefCounter(
            all_block_indices=self._free_block_indices)
        self._block_size = block_size

        self._cow_tracker = CopyOnWriteTracker(
            refcounter=self._refcounter.as_readonly())

        if block_pool is None:
            extra_factor = 4
            # Pre-allocate "num_blocks * extra_factor" block objects.
            # The "* extra_factor" is a buffer to allow more block objects
            # than physical blocks
            self._block_pool = BlockPool(self._block_size, create_block, self,
                                         num_blocks * extra_factor)
        else:
            # In this case, the block pool is provided by the caller,
            # which means that there is most likely a need to share
            # a block pool between allocators
            self._block_pool = block_pool

    def allocate_immutable_block(self,
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
        block = self.allocate_mutable_block(prev_block=prev_block)
        block.append_token_ids(token_ids)
        return block

    def allocate_immutable_blocks(
            self,
            prev_block: Optional[Block],
            block_token_ids: List[List[int]],
            device: Optional[Device] = None) -> List[Block]:
        assert device is None
        num_blocks = len(block_token_ids)

        block_ids = []
        for i in range(num_blocks):
            block_ids.append(self._allocate_block_id())

        blocks = []
        for i in range(num_blocks):
            prev_block = self._block_pool.init_block(
                prev_block=prev_block,
                token_ids=block_token_ids[i],
                block_size=self._block_size,
                physical_block_id=block_ids[i])
            blocks.append(prev_block)

        return blocks

    def allocate_mutable_block(self,
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
        block_id = self._allocate_block_id()
        block = self._block_pool.init_block(prev_block=prev_block,
                                            token_ids=[],
                                            block_size=self._block_size,
                                            physical_block_id=block_id)
        return block

    def _allocate_block_id(self) -> BlockId:
        if not self._free_block_indices:
            raise BlockAllocator.NoFreeBlocksError()

        block_id = self._free_block_indices.popleft()
        self._refcounter.incr(block_id)
        return block_id

    def _free_block_id(self, block: Block) -> None:
        block_id = block.block_id
        assert block_id is not None

        refcount = self._refcounter.decr(block_id)
        if refcount == 0:
            self._free_block_indices.appendleft(block_id)

        block.block_id = None

    def free(self, block: Block, keep_block_object: bool = False) -> None:
        # Release the physical block id
        self._free_block_id(block)

        # Release the block object
        if not keep_block_object:
            self._block_pool.free_block(block)

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

        forked_blocks: List[Block] = []
        prev_block = None
        for block in source_blocks:

            # Increment refcount for each block.
            assert block.block_id is not None
            refcount = self._refcounter.incr(block.block_id)
            assert refcount != 1, "can't fork free'd block"

            forked_block = self._block_pool.init_block(
                prev_block=prev_block,
                token_ids=block.token_ids,
                block_size=self._block_size,
                physical_block_id=block.block_id)

            forked_blocks.append(forked_block)
            prev_block = forked_blocks[-1]

        return forked_blocks

    def get_num_free_blocks(self) -> int:
        return len(self._free_block_indices)

    def get_num_total_blocks(self) -> int:
        return len(self._all_block_indices)

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

    def cow_block_if_not_appendable(self, block: Block) -> BlockId:
        """Performs a copy-on-write operation on the given block if it is not
        appendable.

        Args:
            block (Block): The block to check for copy-on-write.

        Returns:
            BlockId: The block index of the new block if a copy-on-write 
                operation was performed, or the original block index if
                no copy-on-write was necessary.
        """
        src_block_id = block.block_id
        assert src_block_id is not None

        if self._cow_tracker.is_appendable(block):
            return src_block_id

        self._free_block_id(block)
        trg_block_id = self._allocate_block_id()

        self._cow_tracker.record_cow(src_block_id, trg_block_id)

        return trg_block_id

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

    def get_computed_block_ids(self, prev_computed_block_ids: List[int],
                               block_ids: List[int],
                               skip_last_block_id: bool) -> List[int]:
        """No prefix caching here => return empty list
        """
        return []

    def get_common_computed_block_ids(
            self, computed_seq_block_ids: List[List[int]]) -> List[int]:
        """Determine blocks that can be skipped in prefill.

        Since the naive allocator does not support prefix caching, always return
        an empty list.
        """
        return []

    def promote_to_immutable_block(self, block: Block) -> BlockId:
        raise NotImplementedError("There is no promotion for naive blocks")

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
            self._free_block_id(block)

    def swap_in(self, blocks: List[Block]) -> None:
        for block in blocks:
            # Here we allocate either immutable or mutable block and then
            # extract its block_id. Note that the block object is released
            # and the block_id is assigned to "block" to allow reusing the
            # existing "block" object
            if block.is_full:
                tmp_block = self.allocate_immutable_block(
                    prev_block=block.prev_block, token_ids=block.token_ids)
            else:
                tmp_block = self.allocate_mutable_block(
                    prev_block=block.prev_block)
                tmp_block.append_token_ids(block.token_ids)

            block_id = tmp_block.block_id
            tmp_block.block_id = None
            self._block_pool.free_block(tmp_block)

            block.block_id = block_id  # Assign block_id


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
        """Appends the given token IDs to the block and performs a 
        copy-on-write if necessary.

        Args:
            token_ids (Optional[List[int]]): The token IDs to be appended 
                to the block.
        """
        self._append_token_ids_no_cow(token_ids)

        if self._block_id is not None:
            self._block_id = (self._allocator.cow_block_if_not_appendable(
                self._cow_target))

    def _append_token_ids_no_cow(self, token_ids: List[int]) -> None:
        """Appends the given token IDs to the block

        Args:
            token_ids (List[int]): The token IDs to be appended to the block.
        """
        if len(token_ids) == 0:
            return

        assert len(token_ids) <= self.num_empty_slots

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
        return self._block_size - len(self.token_ids)

    @property
    def token_ids(self) -> List[int]:
        return self._token_ids

    @property
    def num_tokens_total(self) -> int:
        raise NotImplementedError(
            "num_tokens_total is not used for naive block")

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def prev_block(self) -> Optional["Block"]:
        return self._prev_block

    @property
    def content_hash(self) -> Optional[int]:
        return None
