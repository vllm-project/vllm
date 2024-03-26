"""Token blocks."""
from typing import List, Optional, Iterable, Dict

from vllm.core.block.interfaces import Block, BlockAllocator
from vllm.core.block.naive_block import NaiveBlockAllocator, NaiveBlock
from vllm.core.block.common import get_all_blocks_recursively, CopyOnWriteTracker

PrefixHash = int
BlockIndex = int


class PrefixCachingBlockAllocator(BlockAllocator):

    # TODO last access time / evictor integration

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        block_ids: Optional[Iterable[int]] = None,
    ):

        self._cached_blocks: Dict[PrefixHash, BlockIndex] = {}
        self._unused_cached_blocks: Dict[PrefixHash, BlockIndex] = {}

        self._hashless_allocator = NaiveBlockAllocator(
            create_block=self._create_block,
            num_blocks=num_blocks,
            block_size=block_size,
            block_ids=block_ids,
        )

        self._block_size = block_size
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
        physical_block_index: Optional[int] = None,
    ) -> Block:
        # Bind block to self.
        allocator = self

        return PrefixCachingBlock(
            prev_block=prev_block,
            token_ids=token_ids,
            block_size=block_size,
            physical_block_index=physical_block_index,
            prefix_caching_allocator=allocator,
        )

    def allocate_immutable(self, prev_block: Optional[Block],
                           token_ids: List[int]) -> Block:
        assert_prefix_caching_block_or_none(prev_block)

        block = self._create_block(
            prev_block=prev_block,
            token_ids=token_ids,
            block_size=self._block_size,
            allocator=self,
        )
        assert block.content_hash is not None

        cached_block_index = self._cached_blocks.get(block.content_hash, None)
        if cached_block_index is not None:
            block.physical_block_index = cached_block_index
            refcount = self._refcounter.incr(block.physical_block_index)
            if refcount == 1:
                assert block.content_hash in self._unused_cached_blocks
                del self._unused_cached_blocks[block.content_hash]
            return block

        block = self.allocate_mutable(prev_block)
        block.append_token_ids(token_ids)
        assert block.content_hash is not None
        # TODO computed bit

        return block

    def _allocate_block_index_for_block(self, block: Block) -> BlockIndex:
        # TODO
        pass

    def allocate_mutable(self, prev_block: Block) -> Block:
        """Look in freelist. If found, return.
        Else, look in cachelist (refcount==0). If found, return.

        Otherwise, raise :(
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

            physical_block_index = self._unused_cached_blocks.pop(
                content_hash_to_evict)
            refcount = self._refcounter.incr(physical_block_index)
            assert refcount == 1
            block = self._create_block(
                prev_block=prev_block,
                token_ids=[],
                block_size=self._block_size,
                allocator=self,
                physical_block_index=physical_block_index,
            )
            assert block.content_hash is None
            return block

        # No block available in hashless allocator, nor in unused cache blocks.
        raise BlockAllocator.NoFreeBlocksError()

    def free(self, block: Block) -> None:
        """Free a block.
        Check if it has a hash. If so, decr refcount ourselves. If zero, add to
        special list. If it does not have a hash, let the hashless allocator
        figure it out.
        """
        # TODO remove this assertion ?
        assert block.physical_block_index is not None

        self._free_block_index_for_block(block.physical_block_index, block)
        block.physical_block_index = None

    def _free_block_index_for_block(self, block_index: BlockIndex,
                                    block: Block) -> None:
        assert isinstance(block, PrefixCachingBlock)

        if block.content_hash is None:
            return self._hashless_allocator.free(block)

        refcount = self._refcounter.decr(block_index)

        # If no longer used, add the block to the unused cached blocks.
        if refcount == 0:
            assert block.content_hash not in self._unused_cached_blocks
            self._unused_cached_blocks[block.content_hash] = block_index

    def fork(self, last_block: Block) -> List[Block]:
        source_blocks = get_all_blocks_recursively(last_block)

        forked_blocks = []
        prev_block = None
        for block in source_blocks:
            refcount = self._refcounter.incr(block.physical_block_index)
            assert refcount != 1, "can't fork free'd block"

            forked_blocks.append(
                self._create_block(
                    prev_block=prev_block,
                    token_ids=block.token_ids,
                    physical_block_index=block.physical_block_index,
                    block_size=self._block_size,
                    allocator=self,
                ))
            prev_block = forked_blocks[-1]

        return forked_blocks

    def get_num_free_blocks(self) -> int:
        return self._hashless_allocator.get_num_free_blocks() + len(
            self._unused_cached_blocks)

    @property
    def all_block_ids(self) -> frozenset[int]:
        return self._hashless_allocator.all_block_ids

    def register_immutable_block(self,
                                 block: "PrefixCachingBlock") -> BlockIndex:
        assert block.content_hash is not None
        assert block.physical_block_index is not None

        # If the content hash does not have a corresponding cached block,
        # set this block as the cached block.
        if block.content_hash not in self._cached_blocks:
            self._cached_blocks[
                block.content_hash] = block.physical_block_index
        else:
            self._free_block_index_for_block(block.physical_block_index, block)
            # TODO need to call a function instead of refcount
            # as the block could transition from unused_cached_blocks
            # is it possible to use a NaiveAllocator for this, with the freelist
            # the uncached?
            self._refcounter.incr(self._cached_blocks[block.content_hash])

        return self._cached_blocks[block.content_hash]

    def cow_block_if_not_appendable(self,
                                    block: Block) -> Optional[BlockIndex]:
        return self._cow_tracker.cow_block_if_not_appendable(block)


    def clear_copy_on_writes(self) -> Dict[BlockIndex, List[BlockIndex]]:
        return self._cow_tracker.clear_cows()


class PrefixCachingBlock(Block):

    def __init__(
        self,
        prev_block: Optional["PrefixCachingBlock"],
        token_ids: List[int],
        block_size: int,
        prefix_caching_allocator: PrefixCachingBlockAllocator,
        physical_block_index: Optional[int] = None,
    ):
        assert_prefix_caching_block_or_none(prev_block)

        self._prev_block = prev_block
        self._cached_content_hash: Optional[int] = None
        self._prefix_caching_allocator = prefix_caching_allocator

        self._block = NaiveBlock(
            prev_block=prev_block,
            token_ids=token_ids,
            block_size=block_size,
            physical_block_index=physical_block_index,
            allocator=prefix_caching_allocator,
            _cow_target=self,
        )

    def append_token_ids(self, token_ids: List[int]) -> None:
        assert token_ids

        self._block.append_token_ids(token_ids)

        # If the content hash is present, then the block can be made immutable.
        # Register ourselves with the allocator, potentially replacing the
        # physical block index.
        if self.content_hash is not None:
            self.physical_block_index = (
                self._prefix_caching_allocator.register_immutable_block(self))

    @property
    def physical_block_index(self) -> Optional[int]:
        return self._block.physical_block_index

    @physical_block_index.setter
    def physical_block_index(self, value) -> None:
        self._block.physical_block_index = value

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
            #cur_block_token_ids=self._block.token_ids)
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
