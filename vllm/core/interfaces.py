import enum
from abc import ABC, abstractmethod
from typing import List
from typing import Sequence as GenericSequence
from typing import Tuple

from vllm.request import Request
from vllm.utils import Device


class AllocStatus(enum.Enum):
    """Result for BlockSpaceManager.can_allocate

    1. Ok: seq_group can be allocated now.
    2. Later: seq_group cannot be allocated.
      The capacity of allocator is larger than seq_group required.
    3. Never: seq_group can never be allocated.
      The seq_group is too large to allocated in GPU.
    """
    OK = enum.auto()
    LATER = enum.auto()
    NEVER = enum.auto()


class BlockSpaceManager(ABC):

    @staticmethod
    def get_block_space_manager_class(version: str):
        version = version.lower()

        if version == "v1":
            from vllm.core.block_manager_v1 import BlockSpaceManagerV1
            return BlockSpaceManagerV1

        if version == "v2":
            from vllm.core.block_manager_v2 import BlockSpaceManagerV2
            return BlockSpaceManagerV2

        if version == "embedding":
            from vllm.core.embedding_model_block_manager import (
                EmbeddingModelBlockSpaceManager)
            return EmbeddingModelBlockSpaceManager

        raise ValueError(f"Unknown version {version=}")

    @abstractmethod
    def can_allocate(self, request: Request) -> AllocStatus:
        pass

    @abstractmethod
    def allocate(self, request: Request) -> None:
        pass

    @abstractmethod
    def can_append_slots(self, request: Request,
                         num_lookahead_slots: int) -> bool:
        pass

    @abstractmethod
    def append_slots(
        self,
        request: Request,
        num_lookahead_slots: int,
    ) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def free(self, request: Request) -> None:
        pass

    @abstractmethod
    def get_block_table(self, request: Request) -> List[int]:
        pass

    @abstractmethod
    def get_num_free_gpu_blocks(self) -> int:
        pass

    @abstractmethod
    def get_num_free_cpu_blocks(self) -> int:
        pass

    @abstractmethod
    def access_all_blocks_in_seq(
        self,
        request: Request,
        access_time: float,
    ) -> None:
        pass

    @abstractmethod
    def get_common_computed_block_ids(
            self, reqs: List[Request]) -> GenericSequence[int]:
        pass

    @abstractmethod
    def mark_blocks_as_computed(self, request: Request, token_chunk_size: int):
        pass

    @abstractmethod
    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        """Prefix cache hit rate. -1 means not supported or disabled."""
        pass
