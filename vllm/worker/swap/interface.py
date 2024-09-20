import enum
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

import torch

from vllm.block import BlockTable, PhysicalTokenBlock
from vllm.config import CacheConfig


class DeviceStatus(enum.Enum):
    OK = enum.auto()
    ERROR = enum.auto()


class SwapDeviceBase(ABC):
    """
    This is the part that manages the connect 
    and is the common part for both DeviceClient (stateless)
    and DeviceManager (stateful)
    """

    @property
    @abstractmethod
    def dev_id(self) -> int:
        pass

    @abstractmethod
    def attach_device(self):
        pass

    @property
    @abstractmethod
    def is_attached(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def detach_device(self):
        pass


class SwapDeviceClient(SwapDeviceBase):
    """
    This is the stateless part of a swap device
    Each worker will initialize one
    This class only manages the connection and 
    the transmission
    """

    @abstractmethod
    def read_block_content(self, caches: List[torch.Tensor],
                           blocks_to_swap_in: List[Tuple[int, int]],
                           layers: List[int], head_range: Tuple[int, int]):
        pass

    @abstractmethod
    def write_block_content(self, caches: List[torch.Tensor],
                            blocks_to_swap_in: List[Tuple[int, int]],
                            layers: List[int], head_range: Tuple[int, int]):
        pass

    @abstractmethod
    def write_block_content_one_layer(self, cache: torch.Tensor,
                                      blocks_to_swap_out: List[Tuple[int,
                                                                     int]],
                                      layer: int,
                                      head_range: Tuple[int, int]) -> None:
        pass

    @abstractmethod
    def read_block_content_one_layer(self, cache: torch.Tensor,
                                     blocks_to_swap_in: List[Tuple[int, int]],
                                     layer: int,
                                     head_range: Tuple[int, int]) -> None:
        pass


class SwapDeviceManager(SwapDeviceBase):
    """
    This can be the stateful part stored with in the manager.
    It manages the block for the local disk
    """

    @abstractmethod
    def can_allocate_block(self) -> bool:
        pass

    @abstractmethod
    def allocate(self, block_hash: int,
                 num_hashed_tokens: int) -> PhysicalTokenBlock:
        pass

    @abstractmethod
    def free(self, block: PhysicalTokenBlock):
        pass

    @abstractmethod
    def get_num_total_blocks(self) -> int:
        pass

    @abstractmethod
    def get_num_free_blocks(self) -> int:
        pass

    @abstractmethod
    def contains_block(self, block_hash: int, num_hashed_tokens: int):
        pass

    @abstractmethod
    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        pass

    @abstractmethod
    def add_rmap(self, block: PhysicalTokenBlock, seq_id: int, block_id: int):
        pass

    @abstractmethod
    def remove_rmap(self, block: PhysicalTokenBlock, seq_id: int,
                    block_id: int):
        pass

    @abstractmethod
    def remove_rmap_all(self, block: PhysicalTokenBlock) -> None:
        pass

    @abstractmethod
    def get_rmap(self,
                 block: PhysicalTokenBlock) -> Optional[Set[Tuple[int, int]]]:
        pass

    @abstractmethod
    def n_rmap(self, block: PhysicalTokenBlock):
        pass

    @abstractmethod
    def move_swappable(self, block: PhysicalTokenBlock):
        pass


class SwapSpaceManagerBase(ABC):

    @staticmethod
    def get_swap_space_manager_class(version: str):
        if version == "default":
            from vllm.worker.swap.swap_manager import SwapSpaceManager
            return SwapSpaceManager
        return ValueError(f"Unknown version {version}")

    @abstractmethod
    def parse_and_add_swap_device(self, cache_config: CacheConfig):
        pass

    @abstractmethod
    def add_swap_device(self, swap_device: SwapDeviceManager) -> DeviceStatus:
        pass

    @abstractmethod
    def remove_swap_device(self,
                           swap_device: SwapDeviceManager) -> DeviceStatus:
        pass

    @abstractmethod
    def get_num_free_blocks_for_all(self) -> int:
        pass

    @abstractmethod
    def contains_block(self,
                       block_hash: int,
                       num_hashed_tokens: int = 0) -> bool:
        pass

    @abstractmethod
    def can_allocate(self,
                     block_hash: int,
                     num_hashed_tokens: int = 0) -> bool:
        pass

    @abstractmethod
    def allocate(self,
                 block_hash: int,
                 num_hashed_tokens: int = 0,
                 swap_dev_id: Optional[int] = None) -> PhysicalTokenBlock:
        pass

    @abstractmethod
    def free(self, block: PhysicalTokenBlock) -> None:
        pass

    @abstractmethod
    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        pass

    @abstractmethod
    def update_block_tables(self, seq_id: int,
                            disk_block_table: Dict[int, BlockTable]):
        pass

    @abstractmethod
    def free_block_tables(self):
        pass

    @abstractmethod
    def add_rmap(self, block: PhysicalTokenBlock, seq_id: int, block_id: int):
        pass

    @abstractmethod
    def remove_rmap(self, block: PhysicalTokenBlock, seq_id: int,
                    block_id: int):
        pass

    @abstractmethod
    def remove_rmap_all(self, block: PhysicalTokenBlock) -> None:
        pass

    @abstractmethod
    def get_rmap(self,
                 block: PhysicalTokenBlock) -> Optional[Set[Tuple[int, int]]]:
        pass

    @abstractmethod
    def n_rmap(self, block: PhysicalTokenBlock):
        pass

    @abstractmethod
    def move_swappable(self, block: PhysicalTokenBlock):
        pass


class SwapClientManagerBase(ABC):

    @staticmethod
    def get_swap_client_manager_class(version: str):
        if version == "default":
            from vllm.worker.swap.swap_manager import SwapClientManager
            return SwapClientManager
        return ValueError(f"Unknown version {version}")

    @abstractmethod
    def parse_and_add_swap_device(self, cache_config: CacheConfig):
        pass

    @abstractmethod
    def add_swap_device(self, swap_device: SwapDeviceClient) -> DeviceStatus:
        pass

    @abstractmethod
    def remove_swap_device(self,
                           swap_device: SwapDeviceClient) -> DeviceStatus:
        pass

    @abstractmethod
    def read_blocks(self, caches: List[torch.Tensor],
                    blocks_to_swap_in: Dict[int, List[Tuple[int, int]]],
                    layers: List[int], head_range: Tuple[int, int]):
        pass

    @abstractmethod
    def write_blocks(self, caches: List[torch.Tensor],
                     blocks_to_swap_out: Dict[int, List[Tuple[int, int]]],
                     layers: List[int], head_range: Tuple[int, int]):
        pass


class SwapSpaceManagerBuilder:
    _swap_space_manager: Optional[SwapSpaceManagerBase] = None

    @classmethod
    def build(cls, version) -> Optional[SwapSpaceManagerBase]:
        SwapManagerImpl = SwapSpaceManagerBase.get_swap_space_manager_class(
            version)
        cls._swap_space_manager = SwapManagerImpl()
        return cls._swap_space_manager

    @classmethod
    def get(cls) -> Optional[SwapSpaceManagerBase]:
        return cls._swap_space_manager
