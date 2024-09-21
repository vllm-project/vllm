"""Token blocks."""
from typing import TYPE_CHECKING, Iterator, List, Optional

from vllm.utils import Device

DEFAULT_LAST_ACCESSED_TIME: float = -1


class PhysicalTokenBlock:
    """Represents the state of a block in the KV cache."""

    def __init__(self,
                 device: Device,
                 block_number: int,
                 block_size: int,
                 block_hash: int,
                 num_hashed_tokens: int,
                 device_id: Optional[int] = None) -> None:
        self.device = device
        self.block_number = block_number
        self.block_size = block_size
        self.block_hash = block_hash
        self.prev_block_hash = block_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.prev_num_hashed_tokens = num_hashed_tokens

        self.ref_count = 0
        self.last_accessed = DEFAULT_LAST_ACCESSED_TIME

        self.prev_computed = False
        self.computed = False

        self.is_evicted = False

        self.device_id = int(self.device) if device_id is None else device_id

    def __repr__(self) -> str:
        return (f'PhysicalTokenBlock(device={self.device}, '
                f'block_number={self.block_number}, '
                f'block_hash={self.block_hash},'
                f'num_hashed_tokens={self.num_hashed_tokens}, '
                f'prev_block_hash={self.prev_block_hash},'
                f'prev_num_hashed_tokens={self.prev_num_hashed_tokens}, '
                f'ref_count={self.ref_count}, '
                f'last_accessed={self.last_accessed}, '
                f'prev_computed={self.prev_computed}, '
                f'computed={self.computed}, '
                f'is_evicted={self.is_evicted})')


class BlockTable:
    """Holds a list of blocks with caching of their associated block_ids 
    """

    def __init__(self, blocks: Optional[List[PhysicalTokenBlock]] = None):
        self._blocks: List[PhysicalTokenBlock] = []
        self._block_ids: List[int] = []

        if blocks is not None:
            for block in blocks:
                self.append(block)

    def append(self, block: PhysicalTokenBlock):
        self._blocks.append(block)
        self._block_ids.append(block.block_number)

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, key):
        return self._blocks[key]

    if TYPE_CHECKING:

        def __iter__(self) -> Iterator[PhysicalTokenBlock]:
            raise RuntimeError("Method should be automatically generated")

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            blocks = value
            self._blocks[key] = blocks
            self._block_ids[key] = [b.block_number for b in blocks]
        else:
            block = value
            self._blocks[key] = block
            self._block_ids[key] = block.block_number

    def __add__(self, other: "BlockTable") -> "BlockTable":
        return BlockTable(self._blocks + other._blocks)

    def reset(self):
        self._blocks = []
        self._block_ids = []

    def copy(self) -> "BlockTable":
        return BlockTable(self._blocks)

    def list(self) -> List[PhysicalTokenBlock]:
        return self._blocks

    def ids(self) -> List[int]:
        return self._block_ids

    def extend(self, block_table: "BlockTable"):
        self._blocks.extend(block_table._blocks)
        self._block_ids.extend(block_table._block_ids)

    def pop(self) -> PhysicalTokenBlock:
        block = self._blocks.pop()
        self._block_ids.pop()
        return block

    def clear(self):
        self._blocks.clear()
        self._block_ids.clear()