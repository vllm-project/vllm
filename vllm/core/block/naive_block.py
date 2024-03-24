from typing import List, Optional, Set, Iterable, Tuple, Dict, Type, TypeVar, T
from abc import ABC, abstractmethod, abstractproperty

from vllm.core.block.interfaces import BlockAllocator, Block
from vllm.core.block.common import RefCounter

from vllm.utils import Device

_BLANK_TOKEN_ID = -1

DEFAULT_LAST_ACCESSED_TIME = -1
    

class NaiveBlockAllocator(BlockAllocator):
    T = TypeVar('T', bound=Block)
    BlockIndex = int
    Refcount = int

    def __init__(
        self,
        create_block: Block.Factory,
        num_blocks: int,
        block_size: int,
        block_ids: Optional[Iterable[int]] = None,
    ):
        if block_ids is None:
            block_ids = range(num_blocks)
        
        self._free_block_indices: Set[BlockIndex] = set(block_ids)
        self._all_block_indices = frozenset(block_ids)

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

    def get_num_free_blocks(self) -> int:
        return len(self._free_block_indices)

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

    @property
    def all_block_ids(self):
        return self._all_block_indices

class NaiveBlock(Block):
    def __init__(self, prev_block: Block, token_ids: List[int], block_size: int, physical_block_index: Optional[int] = None):
        self._token_ids = []
        self._block_size = block_size
        self._prev_block = prev_block
        self._physical_block_index = physical_block_index

        self.append_token_ids(token_ids)

    def append_token_ids(self, token_ids: List[int]) -> None:
        assert self.num_empty_slots >= len(token_ids)
        self._token_ids.extend(token_ids)

    @property
    def physical_block_index(self) -> Optional[int]:
        return self._physical_block_index

    @physical_block_index.setter
    def physical_block_index(self, value: Optional[int]) -> None:
        # TODO only allow call from allocator?
        self._physical_block_index = value

    @property
    def is_full(self) -> bool:
        return self.num_empty_slots == 0

    @property
    def num_empty_slots(self) -> int:
        return self._block_size - len(self._token_ids)

    @property
    def token_ids(self) -> List[int]:
        return self._token_ids

    def block_size(self) -> int:
        return self._block_size
