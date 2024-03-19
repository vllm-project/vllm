"""Token blocks."""
from typing import List, Optional, Set
from abc import ABC, abstractmethod, abstractproperty

from vllm.utils import Device

_BLANK_TOKEN_ID = -1

DEFAULT_LAST_ACCESSED_TIME = -1

"""
PrefixCachingBlock:
	init(prev_block_hash: int, token_ids: List[int])

	Append_token_ids
		If full: raise error

                # if refcount > 1, do cow and get new block
                self.physical_block = cow.maybe_cow(physical_block)
                
		append()
		if full:
                    generate hash
                
                self.physical_block = prefix_cacher.maybe_replace_cached_block(hash, physical_block)

        Get_phys_block_num -> int
	    Raise if not defined

BlockAllocator
	allocate_mutable() -> logical_block
	allocate_immutable(token ids) -> logical_block

	allocate() -> logical block
	free(logical block)

	_Register_immutable_block # only prefix caching

	Get_cow_operations -> Dict[int, List[int]]
	Get_swap_operations -> Dict[int, List[int]]
	Get_compute_operations -> Dict[int, List[int]]
		(cow, swap, compute(?))

NOTE:
    a block can have no physical mapping if it is newly allocated or it
    is preempted (by recompute)
    so we should have optional physical block num
"""

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


class PrefixCachingBlock(Block):
    def __init__(self, prev_block: Block, token_ids: List[int]):
        self._token_ids = token_ids[:]
        self._prev_block = prev_block

    def append_token_ids(self, token_ids: List[int]) -> None:
        pass

    @property
    def physical_block_index(self) -> Optional[int]:
        pass

    @physical_block_index.setter
    def physical_block_index(self) -> None:
        pass

    @property
    def content_hash(self) -> Optional[int]:
        pass

class NaiveBlock(Block):
    def __init__(self, prev_block: Block, token_ids: List[int], physical_block_index: Optional[int] = None):
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

class NaiveBlockAllocator(BlockAllocator):
    T = TypeVar('T', bound=Block)
    BlockIndex = int
    Refcount = int

    def __init__(self, block_cls: Type[T], num_blocks: int, block_size: int):
        self._free_block_indices: Set[BlockIndex] = set(range(num_blocks))
        self._block_refcounts: Dict[BlockIndex, Refcount] = {block_index: 0 for block_index in self._free_block_indices}
        self._block_cls = block_cls
        #self._block_size = block_size

    def allocate_immutable(self, prev_block: Optional[Block], token_ids: List[int]) -> Block:
        block = self.allocate_mutable(prev_block=prev_block)
        block.append_token_ids(token_ids)
        return block

    def allocate_mutable(self, prev_block: Optional[Block]) -> Block:
        block_index = self._allocate_new_block()
        return self._block_cls(prev_block=prev_block, token_ids=[], physical_block_index=block_index)

    def free(self, block: Block) -> None:
        block_index = block.physical_block_index
        block.physical_block_index = None
        self._decr_refcount(block_index)

    def _allocate_new_block(self):
        if not self._free_block_indices:
            raise BlockAllocator.NoFreeBlocksError()

        block_index = next(iter(self._free_block_indices))
        self._incr_refcount(block_index, allow_allocate=True)
        return block_index

    def _incr_refcount(self, block_index: BlockIndex, allow_allocate: bool) -> None:
        assert block_index in self._block_refcounts
        pre_incr_refcount = self._block_refcounts[block_index]

        assert pre_incr_refcount >= 0

        if pre_incr_refcount == 0:
            assert allow_allocate
            assert block_index in self._free_block_indices
            self._free_block_indices.remove(block_index)
        else:
            assert block_index not in self._free_block_indices

        self._block_refcounts[block_index] = pre_incr_refcount + 1

    def _decr_refcount(self, block_index: BlockIndex) -> None:
        assert block_index in self._block_refcounts
        refcount = self._block_refcounts[block_index]

        assert refcount > 0
        refcount -= 1

        self._block_refcounts[block_index] = refcount

        if refcount == 0:
            self._free_block_indices.add(block_index)
    

class PrefixCachingBlockAllocator(BlockAllocator):
    
    def __init__(self):
        #self._mutable_block_allocator = NaiveBlockAllocator()
        #self._cached_blocks: Dict[int, Block]
        pass
    
    def allocate_mutable(self, prev_block: Block) -> Block:
        """Look in freelist. If found, return.
        Else, look in cachelist (refcount==0). If found, return.

        Otherwise, raise :(
        """
        pass

    def allocate_immutable(self, prev_block: Block, token_ids: List[int]) -> Block:
        assert isinstance(prev_block, PrefixCachingBlock)

        block = PrefixCachingBlock(prev_block=prev_block, token_ids=token_ids)
        assert block.content_hash is not None

        if block.content_hash in self._cache_list:
            # incr refcount
            return block
        
        # Do same logic as allocate_mutable; look in freelist, else look in weakref freelist.
 
    def free(self, block: Block) -> None:
        pass

