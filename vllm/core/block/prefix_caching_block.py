"""Token blocks."""
from typing import List, Optional, Set, Iterable, Tuple, Dict
from abc import ABC, abstractmethod, abstractproperty

from vllm.core.block.interfaces import Block, BlockAllocator
from vllm.core.block.naive_block import NaiveBlockAllocator
from vllm.core.block.common import RefCounter

from vllm.utils import Device

_BLANK_TOKEN_ID = -1

DEFAULT_LAST_ACCESSED_TIME = -1

class PrefixCachingBlockAllocator(BlockAllocator):
    PrefixHash = int
    BlockIndex = int
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

    # Implements Block.Factory.
    def _create_block(
        self,
        prev_block: Optional[Block],
        token_ids: List[int],
        block_size: int,
        physical_block_index: Optional[int] = None,
    ) -> Block:
        # Bind block to self.
        return PrefixCachingBlock(
            prev_block=prev_block,
            token_ids=token_ids,
            block_size=self._block_size,
            prefix_caching_allocator=self,
            physical_block_index=physical_block_index,
        )


    def allocate_immutable(self, prev_block: Optional[Block], token_ids: List[int]) -> Block:
        assert_prefix_caching_block_or_none(prev_block)

        block = self._create_block(
            prev_block=prev_block,
            token_ids=token_ids,
            block_size=self._block_size,
        )
        assert block.content_hash is not None

        cached_block_index = self._cached_blocks.get(block.content_hash, None)
        if cached_block_index is not None:
            block.physical_block_index = cached_block_index
            self._refcounter.incr(block.physical_block_index)
            return block

        block = self.allocate_mutable(prev_block)
        block.append_token_ids(token_ids)
        assert block.content_hash is not None
        # TODO computed bit

        return block
 
    
    def allocate_mutable(self, prev_block: Block) -> Block:
        """Look in freelist. If found, return.
        Else, look in cachelist (refcount==0). If found, return.

        Otherwise, raise :(
        """
        assert_prefix_caching_block_or_none(prev_block)

        try:
            return self._hashless_allocator.allocate_mutable(prev_block=prev_block)
        except BlockAllocator.NoFreeBlocksError:
            # We must check the unused cached blocks before raising OOM.
            pass
        
        if self._unused_cached_blocks:
            # TODO policy for selecting block to remove
            content_hash_to_evict = next(iter(self._unused_cached_blocks))
            physical_block_index = self._unused_cached_blocks.pop(content_hash_to_evict)
            refcount = self._refcounter.incr(physical_block_index)
            block = self._create_block(
                prev_block=prev_block,
                token_ids=[],
                block_size=self._block_size,
                physical_block_index=physical_block_index,
            )
            assert block.content_hash is None
            return block

        # No block available in hashless allocator, nor in unused cache blocks.
        raise BlockAllocator.NoFreeBlocksError()

    def free(self, block: Block) -> None:
        """Free a block.
        Check if it has a hash. If so, decr refcount ourselves. If zero, add to special list.
        If it does not have a hash, let the hashless allocator figure it out.
        """
        assert isinstance(block, PrefixCachingBlock)
        assert block.physical_block_index is not None

        if block.content_hash is None:
            return self._hashless_allocator.free(block)
        
        physical_block_index = block.physical_block_index
        block.physical_block_index = None
        refcount = self._refcounter.decr(physical_block_index)
        
        # If no longer used, add the block to the unused cached blocks.
        if refcount == 0:
            assert block.content_hash not in self._unused_cached_blocks
            self._unused_cached_blocks[block.content_hash] = physical_block_index

    def get_num_free_blocks(self) -> int:
        return self._hashless_allocator.get_num_free_blocks() + len(self._unused_cached_blocks)

    @property
    def all_block_ids(self) -> frozenset[int]:
        return self._hashless_allocator.all_block_ids

    # TODO name: upsert_
    # promote
    # replace
    def register_immutable_block(self, block: "PrefixCachingBlock") -> BlockIndex:
        assert block.content_hash is not None
        assert block.physical_block_index is not None

        # If the content hash does not have a corresponding cached block,
        # set this block as the cached block.
        if block.content_hash not in self._cached_blocks:
            self._cached_blocks[block.content_hash] = block.physical_block_index

        return self._cached_blocks[block.content_hash]

class PrefixCachingBlock(Block):
    def __init__(
        self,
        prev_block: Optional["PrefixCachingBlock"],
        token_ids: List[int],
        block_size: int,
        prefix_caching_allocator: PrefixCachingBlockAllocator,
        physical_block_index: Optional[int] = None,
    ):
        self._prev_block = prev_block
        self._token_ids = token_ids[:]
        self._block_size = block_size
        self._cached_content_hash: Optional[int] = None
        self._physical_block_index = physical_block_index
        self._prefix_caching_allocator = prefix_caching_allocator

        assert_prefix_caching_block_or_none(prev_block)

    def append_token_ids(self, token_ids: List[int]) -> None:
        assert token_ids
        assert len(self._token_ids) + len(token_ids) <= self._block_size

        self._token_ids.extend(token_ids)

        # If the content hash is present, then the block can be made immutable.
        # Register ourselves with the allocator, potentially replacing the physical block index.
        if self.content_hash is not None:
            self.physical_block_index = self._prefix_caching_allocator.register_immutable_block(self)

    @property
    def physical_block_index(self) -> Optional[int]:
        return self._physical_block_index

    @physical_block_index.setter
    def physical_block_index(self, value) -> None:
        self._physical_block_index = value

    def is_full(self) -> bool:
        return len(self._token_ids) == self._block_size

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
        if not self.is_full():
            return None

        is_first_block = self._prev_block is None
        prev_block_hash = (None if is_first_block else self._prev_block.content_hash)

        # Previous block exists but does not yet have a hash.
        # Return no hash in this case.
        if prev_block_hash is None and not is_first_block:
            return None

        self._cached_content_hash = PrefixCachingBlock.hash_block_tokens(
            is_first_block,
            prev_block_hash,
            cur_block_token_ids=self._token_ids)
        return self._cached_content_hash

    @staticmethod
    def hash_block_tokens(is_first_block: bool, prev_block_hash: Optional[int], cur_block_token_ids) -> int:
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
