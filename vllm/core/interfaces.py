import enum
from abc import ABC, abstractmethod
from typing import Dict, List
from typing import Sequence as GenericSequence

from vllm.sequence import Sequence, SequenceGroup


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

        raise ValueError(f"Unknown version {version=}")

    @abstractmethod
    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        pass

    @abstractmethod
    def allocate(self, seq_group: SequenceGroup) -> None:
        pass

    @abstractmethod
    def can_append_slots(self, seq_group: SequenceGroup,
                         num_lookahead_slots: int) -> bool:
        pass

    @abstractmethod
    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int,
    ) -> Dict[int, List[int]]:
        pass

    @abstractmethod
    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        pass

    @abstractmethod
    def can_swap_in(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int) -> AllocStatus:
        pass

    @abstractmethod
    def swap_in(self, seq_group: SequenceGroup,
                num_lookahead_slots: int) -> Dict[int, int]:
        pass

    @abstractmethod
    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        pass

    @abstractmethod
    def swap_out(self, seq_group: SequenceGroup) -> Dict[int, int]:
        pass

    @abstractmethod
    def free(self, seq: Sequence) -> None:
        pass

    @abstractmethod
    def get_block_table(self, seq: Sequence) -> List[int]:
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
        seq: Sequence,
        access_time: float,
    ) -> None:
        pass

    @abstractmethod
    def get_common_computed_block_ids(
            self, seqs: List[Sequence]) -> GenericSequence[int]:
        pass

    @abstractmethod
    def mark_blocks_as_computed(self, seq_group: SequenceGroup):
        pass
