from collections import deque
from typing import Deque
import enum
import bisect

from vllm.sequence import SequenceGroup


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


class Policy:
    """Base class policy"""

    def sort(
        self,
        seq_groups: Deque[SequenceGroup],
    ) -> Deque[SequenceGroup]:
        raise NotImplementedError

    def get_preemption_mode(self, seq_group: SequenceGroup) -> PreemptionMode:
        raise NotImplementedError


class FCFS(Policy):

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def sort(
        self,
        seq_groups: Deque[SequenceGroup],
    ) -> Deque[SequenceGroup]:
        """We can just sort `Deque[SequenceGroup]` by `arrival_time`"""
        return deque(sorted(seq_groups, key=lambda x: x.metrics.arrival_time))

    def get_preemption_mode(self, seq_group: SequenceGroup) -> PreemptionMode:
        if seq_group.get_max_num_running_seqs() == 1:
            return PreemptionMode.RECOMPUTE
        else:
            return PreemptionMode.SWAP


class ReorderPolicy(Policy):
    """ReorderPolicy tries to maximize throughput by reordering incoming requests by length.
    
    Args:
        reorder_window: window size in sec within which `List[SequenceGroup]` is allowed to be reordered. 0 means no reorder. 
    """

    def __init__(self, reorder_window: float = 0, **kwargs) -> None:
        super().__init__()
        self.reorder_window = reorder_window

    def sort(
        self,
        seq_groups: Deque[SequenceGroup],
    ) -> Deque[SequenceGroup]:
        """Sort head within `reorder_window` of the `seq_groups` by length. It reduces padding computation overhead."""
        if len(seq_groups) == 0:
            return seq_groups
        arrival_time_sorted = sorted(seq_groups,
                                     key=lambda x: x.metrics.arrival_time)
        pos = bisect.bisect_left(arrival_time_sorted,
                                 arrival_time_sorted[0].metrics.arrival_time +
                                 self.reorder_window,
                                 key=lambda x: x.metrics.arrival_time)
        return deque(
            sorted(arrival_time_sorted[:pos],
                   key=lambda x: x.get_seqs()[0].get_len()) +
            arrival_time_sorted[pos:])

    def get_preemption_mode(self, seq_group: SequenceGroup) -> PreemptionMode:
        """Always use SWAP, as it is faster than `RECOMPUTE` for heavy models like llama."""
        return PreemptionMode.SWAP


class PolicyFactory:

    _POLICY_REGISTRY = {
        'fcfs': FCFS,
        'reorder': ReorderPolicy,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
