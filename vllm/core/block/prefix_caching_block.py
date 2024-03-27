"""Token blocks."""
from itertools import takewhile
from os.path import commonprefix
from typing import Dict, Iterable, List, Optional

from vllm.core.block.common import (CopyOnWriteTracker,
                                    get_all_blocks_recursively)
from vllm.core.block.interfaces import Block, BlockAllocator
from vllm.core.block.naive_block import NaiveBlock, NaiveBlockAllocator

PrefixHash = int
BlockId = int


class PrefixCachingBlockAllocator(BlockAllocator):
    """A block allocator that implements prefix caching.

    The PrefixCachingBlockAllocator maintains a cache of blocks based on their
    content hash. It reuses blocks with the same content hash to avoid redundant
    memory allocation. The allocator also supports copy-on-write operations.

    Args:
        num_blocks (int): The total number of blocks to manage.
        block_size (int): The size of each block in tokens.
        block_ids(Optional[Iterable[int]], optional): An optional iterable of
            block IDs. If not provided, block IDs will be assigned sequentially
            from 0 to num_blocks - 1.
    """

    # TODO last access time / evictor integration

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        block_ids: Optional[Iterable[int]] = None,
    ):
        # A mapping of prefix hash to block index. All blocks which have a
        # prefix hash will be in this dict, even if they have refcount 0.
        self._cached_blocks: Dict[PrefixHash, BlockId] = {}

        # A mapping of prefix hash to block index. All blocks which have a
        # prefix hash AND refcount 0 will be in this dict. Thus, it is a subset
        # of self._cached_blocks.
        self._unused_cached_blocks: Dict[PrefixHash, BlockId] = {}

        # An allocator for blocks that do not have prefix hashes.
        self._hashless_allocator = NaiveBlockAllocator(
            create_block=self._create_block,
            num_blocks=num_blocks,
            block_size=block_size,
            block_ids=block_ids,
        )

        self._block_size = block_size

        # We share the refcounter between allocators. This allows us to promote
        # blocks originally allocated in the hashless allocator to immutable
        # blocks.
        self._refcounter = self._hashless_allocator.refcounter

        self._cow_tracker = CopyOnWriteTracker(
            refcounter=self._refcounter.as_readonly(),
            allocator=self,
        )

    # Implements Block.Factory.
    def _create_block(
        self,
        prev_block: Optional[Block],
        token_ids: List[int],
        block_size: int,
        allocator: BlockAllocator,
        block_id: Optional[int] = None,
    ) -> Block:
        # Bind block to self.
        allocator = self

        return PrefixCachingBlock(
            prev_block=prev_block,
            token_ids=token_ids,
            block_size=block_size,
            block_id=block_id,
            prefix_caching_allocator=allocator,
        )

    def allocate_immutable(self, prev_block: Optional[Block],
                           token_ids: List[int]) -> Block:
        """Allocates an immutable block with the given token IDs, reusing cached
        blocks if possible.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence.
            token_ids (List[int]): The token IDs to be stored in the block.

        Returns:
            Block: The allocated immutable block.
        """
        assert_prefix_caching_block_or_none(prev_block)

        block = self._create_block(
            prev_block=prev_block,
            token_ids=token_ids,
            block_size=self._block_size,
            allocator=self,
        )
        assert block.content_hash is not None

        cached_block_id = self._cached_blocks.get(block.content_hash, None)
        if cached_block_id is not None:
            block.block_id = cached_block_id
            self._incr_refcount_cached_block(block.content_hash,
                                             block.block_id)
            return block

        block = self.allocate_mutable(prev_block)
        block.append_token_ids(token_ids)
        assert block.content_hash is not None
        # TODO computed bit

        return block

    def allocate_mutable(self, prev_block: Block) -> Block:
        """Allocates a mutable block. If there are no free blocks, this will
        evict unused cached blocks.

        Args:
            prev_block (Block): The previous block in the sequence.

        Returns:
            Block: The allocated mutable block.
        """
        assert_prefix_caching_block_or_none(prev_block)

        try:
            return self._hashless_allocator.allocate_mutable(
                prev_block=prev_block)
        except BlockAllocator.NoFreeBlocksError:
            # We must check the unused cached blocks before raising OOM.
            pass

        if self._unused_cached_blocks:
            # TODO policy for selecting block to remove
            content_hash_to_evict = next(iter(self._unused_cached_blocks))

            # Clear content hash mapping; the block will be overwritten.
            del self._cached_blocks[content_hash_to_evict]

            block_id = self._unused_cached_blocks.pop(content_hash_to_evict)
            refcount = self._refcounter.incr(block_id)
            assert refcount == 1
            block = self._create_block(
                prev_block=prev_block,
                token_ids=[],
                block_size=self._block_size,
                allocator=self,
                block_id=block_id,
            )
            assert block.content_hash is None
            return block

        # No block available in hashless allocator, nor in unused cache blocks.
        raise BlockAllocator.NoFreeBlocksError()

    def _incr_refcount_cached_block(self, content_hash: int,
                                    block_id: BlockId) -> None:
        refcount = self._refcounter.incr(block_id)
        if refcount == 1:
            assert content_hash in self._unused_cached_blocks
            del self._unused_cached_blocks[content_hash]

    def free(self, block: Block) -> None:
        """Decrement the refcount of the block. If the decremented refcount is
        zero, store the block in the freelist.

        If the block has a content hash (meaning it is immutable), then we will
        keep the block around in case future allocations require it.
        """
        assert (block.block_id
                is not None), "freeing unallocated block is undefined"

        self._free_block_id_for_block(block.block_id, block)
        block.block_id = None

    def _free_block_id_for_block(self, block_id: BlockId,
                                 block: Block) -> None:
        assert isinstance(block, PrefixCachingBlock)

        if block.content_hash is None:
            return self._hashless_allocator.free(block)

        refcount = self._refcounter.decr(block_id)

        # If no longer used, add the block to the unused cached blocks.
        if refcount == 0:
            assert block.content_hash not in self._unused_cached_blocks
            assert block.content_hash in self._cached_blocks
            self._unused_cached_blocks[block.content_hash] = block_id

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
        # The number of free blocks is the number of hashless free blocks
        # plus the number of hashful blocks that are unused.
        return self._hashless_allocator.get_num_free_blocks() + len(
            self._unused_cached_blocks)

    @property
    def all_block_ids(self) -> frozenset[int]:
        return self._hashless_allocator.all_block_ids

    def promote_to_immutable_block(self,
                                   block: "PrefixCachingBlock") -> BlockId:
        """Once a mutable block is full, it can be promoted to an immutable
        block. This means that its content can be referenced by future blocks
        having the same prefix.

        Note that if we already have a cached block with the same content, we
        will replace the newly-promoted block's mapping with the existing cached
        block.

        Args:
            block (PrefixCachingBlock): The mutable block to be promoted.

        Returns:
            BlockId: Either the original block index, or the block index of
                the previously cached block matching the same content.
        """
        assert block.content_hash is not None
        assert block.block_id is not None
        assert self._refcounter.get(block.block_id) > 0

        # If the content hash does not have a corresponding cached block,
        # set this block as the cached block.
        if block.content_hash not in self._cached_blocks:
            self._cached_blocks[block.content_hash] = block.block_id
        else:
            self._free_block_id_for_block(block.block_id, block)
            self._incr_refcount_cached_block(
                block.content_hash, self._cached_blocks[block.content_hash])

        return self._cached_blocks[block.content_hash]

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

    def clear_copy_on_writes(self) -> Dict[BlockId, List[BlockId]]:
        """Returns the copy-on-write source->destination mapping and clears it.

        Returns:
            Dict[BlockId, List[BlockId]]: A dictionary mapping source
                block indices to lists of destination block indices.
        """
        return self._cow_tracker.clear_cows()

    def mark_blocks_as_computed(self) -> None:
        """Mark blocks as computed, used in prefix caching."""
        # TODO Track computed blocks.
        pass

    def get_common_computed_block_ids(
            self, seq_block_ids: List[List[int]]) -> List[int]:
        """Return the block ids that are common for a given sequence group.

        Used in prefill (can skip prefill of some blocks).
        """

        # TODO: Track computed blocks.
        computed = lambda block_id: False

        # NOTE We exclude the last block to avoid the case where the entire
        # prompt is cached. This would cause erroneous behavior in model
        # runner.
        ids_list = [
            takewhile(lambda block_id: computed(block_id), seq[:-1])
            for seq in seq_block_ids
        ]
        return commonprefix([ids for ids in ids_list if ids != []])


class PrefixCachingBlock(Block):
    """A block implementation that supports prefix caching.

    The PrefixCachingBlock class represents a block of token IDs with prefix
    caching capabilities. It wraps a NaiveBlock internally and provides
    additional functionality for content hashing and promoting immutable blocks
    with the prefix caching allocator.

    Args:
        prev_block (Optional[PrefixCachingBlock]): The previous block in the
            sequence.
        token_ids (List[int]): The initial token IDs to be stored in the block.
        block_size (int): The maximum number of token IDs that can be stored in
            the block.
        prefix_caching_allocator (PrefixCachingBlockAllocator): The prefix
            caching block allocator associated with this block.
        block_id (Optional[int], optional): The physical block index
            of this block. Defaults to None.
    """

    def __init__(
        self,
        prev_block: Optional["PrefixCachingBlock"],
        token_ids: List[int],
        block_size: int,
        prefix_caching_allocator: PrefixCachingBlockAllocator,
        block_id: Optional[int] = None,
    ):
        assert_prefix_caching_block_or_none(prev_block)

        self._prev_block = prev_block
        self._cached_content_hash: Optional[int] = None
        self._prefix_caching_allocator = prefix_caching_allocator

        self._block = NaiveBlock(
            prev_block=prev_block,
            token_ids=token_ids,
            block_size=block_size,
            block_id=block_id,
            allocator=prefix_caching_allocator,
            _cow_target=self,
        )

    def append_token_ids(self, token_ids: List[int]) -> None:
        """Appends the given token IDs to the block and registers the block as
        immutable if the block becomes full.

        Internally, the naive block handles CoW.

        Args:
            token_ids (List[int]): The token IDs to be appended to the block.
        """
        assert token_ids

        # naive block handles CoW.
        self._block.append_token_ids(token_ids)

        # If the content hash is present, then the block can be made immutable.
        # Register ourselves with the allocator, potentially replacing the
        # physical block index.
        if self.content_hash is not None:
            self.block_id = (self._prefix_caching_allocator.
                             promote_to_immutable_block(self))

    @property
    def block_id(self) -> Optional[int]:
        return self._block.block_id

    @block_id.setter
    def block_id(self, value) -> None:
        self._block.block_id = value

    @property
    def is_full(self) -> bool:
        return self._block.is_full

    @property
    def num_empty_slots(self) -> int:
        return self._block.num_empty_slots

    @property
    def block_size(self) -> int:
        return self._block.block_size

    @property
    def token_ids(self) -> List[int]:
        return self._block.token_ids

    @property
    def prev_block(self) -> Optional[Block]:
        return self._prev_block

    @property
    def content_hash(self) -> Optional[int]:
        """Return the content-based hash of the current block, or None if it is
        not yet defined.

        For the content-based hash to be defined, the current block must be
        full.
        """

        # If the hash is already computed, return it.
        if self._cached_content_hash is not None:
            return self._cached_content_hash

        # We cannot compute a hash for the current block because it is not full.
        if not self.is_full:
            return None

        is_first_block = self._prev_block is None
        prev_block_hash = (None if is_first_block else
                           self._prev_block.content_hash)

        # Previous block exists but does not yet have a hash.
        # Return no hash in this case.
        if prev_block_hash is None and not is_first_block:
            return None

        self._cached_content_hash = PrefixCachingBlock.hash_block_tokens(
            is_first_block,
            prev_block_hash,
            cur_block_token_ids=self.token_ids)
        return self._cached_content_hash

    @staticmethod
    def hash_block_tokens(is_first_block: bool, prev_block_hash: Optional[int],
                          cur_block_token_ids: List[int]) -> int:
        """Computes a hash value corresponding to the contents of a block and
        the contents of the preceding block(s). The hash value is used for
        prefix caching.

        NOTE: Content-based hashing does not yet support LoRA.

        Parameters:
        - is_first_block (bool): A flag indicating if the block is the first in
            the sequence.
        - prev_block_hash (Optional[int]): The hash of the previous block. None
            if this is the first block.
        - cur_block_token_ids (List[int]): A list of token ids in the current
            block. The current block is assumed to be full.

        Returns:
        - int: The computed hash value for the block.
        """
        assert (prev_block_hash is None) == is_first_block
        return hash((is_first_block, prev_block_hash, *cur_block_token_ids))


def assert_prefix_caching_block_or_none(block: Optional[Block]):
    if block is None:
        return
    assert isinstance(block, PrefixCachingBlock)
