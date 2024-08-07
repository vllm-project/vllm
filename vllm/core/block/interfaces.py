from abc import ABC, abstractmethod
from typing import Dict, FrozenSet, List, Optional, Protocol, Tuple

from vllm.utils import Device

BlockId = int


class Block(ABC):

    @abstractmethod
    def append_token_ids(self, token_ids: List[int]) -> None:
        pass

    @property
    @abstractmethod
    def block_id(self) -> Optional[int]:
        pass

    @block_id.setter
    @abstractmethod
    def block_id(self, value: Optional[int]) -> None:
        """NOTE: Do not use this API outside Block."""
        self._block_id = value

    @property
    @abstractmethod
    def token_ids(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def num_tokens_total(self) -> int:
        """The number of tokens till the current block (inclusive)
        """
        pass

    @property
    @abstractmethod
    def num_empty_slots(self) -> int:
        pass

    @property
    @abstractmethod
    def is_full(self) -> bool:
        pass

    @property
    @abstractmethod
    def prev_block(self) -> Optional["Block"]:
        pass

    @property
    @abstractmethod
    def computed(self) -> bool:
        raise NotImplementedError

    @computed.setter
    @abstractmethod
    def computed(self, value) -> bool:
        """Should be only used by PrefixCacingAllocator"""
        raise NotImplementedError

    @property
    @abstractmethod
    def last_accessed(self) -> float:
        raise NotImplementedError

    @last_accessed.setter
    @abstractmethod
    def last_accessed(self, last_accessed_ts: float):
        raise NotImplementedError

    class Factory(Protocol):

        @abstractmethod
        def __call__(
            self,
            prev_block: Optional["Block"],
            token_ids: List[int],
            block_size: int,
            allocator: "BlockAllocator",
            block_id: Optional[int] = None,
        ) -> "Block":
            pass

    @property
    @abstractmethod
    def content_hash(self) -> Optional[int]:
        """Return the content-based hash of the current block, or None if it is
        not yet defined or not supported.

        For the content-based hash to be defined, the current block must be
        full.
        """
        return None


class BlockAllocator(ABC):

    @abstractmethod
    def allocate_mutable_block(self, prev_block: Optional[Block]) -> Block:
        pass

    @abstractmethod
    def allocate_immutable_block(self, prev_block: Optional[Block],
                                 token_ids: List[int]) -> Block:
        pass

    @abstractmethod
    def allocate_immutable_blocks(
            self, prev_block: Optional[Block],
            block_token_ids: List[List[int]]) -> List[Block]:
        pass

    @abstractmethod
    def free(self, block: Block) -> None:
        pass

    @abstractmethod
    def fork(self, last_block: Block) -> List[Block]:
        pass

    @abstractmethod
    def get_num_total_blocks(self) -> int:
        pass

    @abstractmethod
    def get_num_free_blocks(self) -> int:
        pass

    @abstractmethod
    def get_physical_block_id(self, absolute_id: int) -> int:
        pass

    @abstractmethod
    def swap_out(self, blocks: List[Block]) -> None:
        pass

    @abstractmethod
    def swap_in(self, blocks: List[Block]) -> None:
        pass

    @property
    @abstractmethod
    def all_block_ids(self) -> FrozenSet[int]:
        pass

    @abstractmethod
    def clear_copy_on_writes(self) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def mark_blocks_as_accessed(self, block_ids: List[int],
                                now: float) -> None:
        pass

    @abstractmethod
    def mark_blocks_as_computed(self, block_ids: List[int]) -> None:
        pass

    @abstractmethod
    def get_computed_block_ids(self, prev_computed_block_ids: List[int],
                               block_ids: List[int],
                               skip_last_block_id: bool) -> List[int]:
        pass

    @abstractmethod
    def get_common_computed_block_ids(
            self, computed_seq_block_ids: List[List[int]]) -> List[int]:
        pass

    @abstractmethod
    def cow_block_if_not_appendable(self, block: Block) -> BlockId:
        """NOTE: This should not be used besides Block"""
        pass

    @abstractmethod
    def promote_to_immutable_block(self, block: Block) -> BlockId:
        """NOTE: This should not be used besides Block"""
        pass

    @abstractmethod
    def get_num_blocks_touched(self,
                               blocks: List[Block],
                               num_lookahead_slots: int = 0) -> int:
        pass

    class NoFreeBlocksError(ValueError):
        pass


class DeviceAwareBlockAllocator(ABC):

    @abstractmethod
    def allocate_mutable_block(self, prev_block: Optional[Block],
                               device: Device) -> Block:
        pass

    @abstractmethod
    def allocate_immutable_block(self, prev_block: Optional[Block],
                                 token_ids: List[int],
                                 device: Device) -> Block:
        pass

    @abstractmethod
    def allocate_immutable_blocks(self, prev_block: Optional[Block],
                                  block_token_ids: List[List[int]],
                                  device: Device) -> List[Block]:
        pass

    @abstractmethod
    def get_num_free_blocks(self, device: Device) -> int:
        pass

    @abstractmethod
    def get_num_total_blocks(self, device: Device) -> int:
        pass

    @abstractmethod
    def free(self, block: Block) -> None:
        pass

    @abstractmethod
    def fork(self, last_block: Block) -> List[Block]:
        pass

    @property
    @abstractmethod
    def all_block_ids(self) -> FrozenSet[int]:
        pass

    @abstractmethod
    def clear_copy_on_writes(self) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def mark_blocks_as_accessed(self, block_ids: List[int],
                                now: float) -> None:
        pass

    @abstractmethod
    def mark_blocks_as_computed(self, block_ids: List[int]) -> None:
        pass

    @abstractmethod
    def get_computed_block_ids(self, prev_computed_block_ids: List[int],
                               block_ids: List[int],
                               skip_last_block_id: bool) -> List[int]:
        pass

    @abstractmethod
    def get_common_computed_block_ids(
            self, computed_seq_block_ids: List[List[int]]) -> List[int]:
        pass

    @abstractmethod
    def get_num_blocks_touched(self,
                               blocks: List[Block],
                               device: Device,
                               num_lookahead_slots: int = 0) -> int:
        pass

    @abstractmethod
    def swap(self, blocks: List[Block], src_device: Device,
             dst_device: Device) -> Dict[int, int]:
        pass

    @abstractmethod
    def get_physical_block_id(self, device: Device, absolute_id: int) -> int:
        pass

    @abstractmethod
    def allocate_or_get_null_block(self) -> Block:
        """
        Null blocks are used as a placeholders for KV cache blocks that have
        been dropped due to sliding window.
        There is at most one null block per allocator.
        """
        pass
