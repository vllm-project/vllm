"""Token blocks."""
from typing import List, Optional, Set, Iterable, Callable
from abc import ABC, abstractmethod, abstractproperty

from vllm.utils import Device

_BLANK_TOKEN_ID = -1

DEFAULT_LAST_ACCESSED_TIME = -1


class Block(ABC):

    @abstractmethod
    def append_token_ids(self, token_ids: List[int]) -> None:
        pass

    @abstractproperty
    def physical_block_index(self) -> Optional[int]:
        pass

class BlockAllocator(ABC):
    @abstractmethod
    def allocate_mutable(self, prev_block: Optional[Block]) -> Block:
        pass

    @abstractmethod
    def allocate_immutable(self, prev_block: Optional[Block], token_ids: List[int]) -> Block:
        pass
 
    @abstractmethod
    def free(self, block: Block) -> None:
        pass

    class NoFreeBlocksError(ValueError):
        pass

    #@abstractmethod
    #def get_operations(self):
    #    pass

class NaiveBlock(Block):
    def __init__(self, prev_block: Block, token_ids: List[int], block_size: int, physical_block_index: Optional[int] = None):
        self._token_ids = token_ids[:]
        self._prev_block = prev_block
        self._physical_block_index = physical_block_index

    def append_token_ids(self, token_ids: List[int]) -> None:
        pass

    @property
    def physical_block_index(self) -> Optional[int]:
        return self._physical_block_index

    @physical_block_index.setter
    def physical_block_index(self, value: Optional[int]) -> None:
        # TODO only allow call from allocator?
        self._physical_block_index = value
    

from typing import Type, TypeVar, T

class RefCounter:
    BlockIndex = int
    RefCount = int

    def __init__(self, all_block_indices: Iterable[BlockIndex]):
        deduped = set(all_block_indices)
        self._refcounts: Dict[BlockIndex, RefCount] = {index: 0 for index in deduped}

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

from typing import Protocol

class BlockCreator(Protocol):

    @abstractmethod
    def __call__(
        self,
        prev_block: Optional[Block],
        token_ids: List[int],
        block_size: int,
        physical_block_index: Optional[int] = None,
    ) -> Block:
        pass

class NaiveBlockAllocator(BlockAllocator):
    T = TypeVar('T', bound=Block)
    BlockIndex = int
    Refcount = int

    def __init__(self, create_block: BlockCreator, num_blocks: int, block_size: int):
        self._free_block_indices: Set[BlockIndex] = set(range(num_blocks))
        self._refcounter = RefCounter(all_block_indices=self._free_block_indices)
        self._create_block = create_block
        self._block_size = block_size

    def allocate_immutable(self, prev_block: Optional[Block], token_ids: List[int]) -> Block:
        block = self.allocate_mutable(prev_block=prev_block)
        block.append_token_ids(token_ids)
        return block

    def allocate_mutable(self, prev_block: Optional[Block]) -> Block:
        block_index = self._allocate_new_block()
        return self._create_block(
            prev_block=prev_block,
            token_ids=[],
            physical_block_index=block_index,
            block_size=self._block_size,
        )

    def free(self, block: Block) -> None:
        block_index = block.physical_block_index
        block.physical_block_index = None

        refcount = self._refcounter.decr(block_index)
        if refcount == 0:
            self._free_block_indices.add(block_index)

    def _allocate_new_block(self):
        if not self._free_block_indices:
            raise BlockAllocator.NoFreeBlocksError()

        block_index = next(iter(self._free_block_indices))
        refcount = self._refcounter.incr(block_index)
        self._free_block_indices.remove(block_index)
        return block_index

    @property
    def refcounter(self):
        return self._refcounter
    

class PrefixCachingBlock(Block):
    def __init__(
        self,
        prev_block: Optional["PrefixCachingBlock"],
        token_ids: List[int],
        block_size: int,
        prefix_caching_allocator: "PrefixCachingBlockAllocator",
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
        assert len(self._token_ids) + len(token_ids) <= self._block_size
        self._token_ids.extend(token_ids)

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


class PrefixCachingBlockAllocator(BlockAllocator):
    PrefixHash = int
    BlockIndex = int
    # TODO last access time / evictor integration
    
    def __init__(self, num_blocks: int, block_size: int):
        self._cached_blocks: Dict[PrefixHash, BlockIndex] = {}

        self._hashless_allocator = NaiveBlockAllocator(
            create_block=self._create_block,
            num_blocks=num_blocks,
            block_size=block_size,
        )

        self._block_size = block_size
        self._refcounter = self._hashless_allocator.refcounter

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
        
    
    def allocate_mutable(self, prev_block: Block) -> Block:
        """Look in freelist. If found, return.
        Else, look in cachelist (refcount==0). If found, return.

        Otherwise, raise :(
        """
        try:
            block = self._hashless_allocator.allocate_mutable(prev_block=prev_block)
            return block
        except BlockAllocator.NoFreeBlocksError:
            pass

        # TODO: weakref
        raise BlockAllocator.NoFreeBlocksError()

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
 
    def free(self, block: Block) -> None:
        pass

def assert_prefix_caching_block_or_none(block: Optional[Block]):
    if block is None:
        return
    assert isinstance(block, PrefixCachingBlock)
