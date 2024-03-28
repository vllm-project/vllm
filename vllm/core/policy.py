from collections import deque
from typing import Deque

from vllm.sequence import SequenceGroup


class Policy:

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        raise NotImplementedError

    def sort_by_priority(
        self,
        now: float,
        seq_groups: Deque[SequenceGroup],
    ) -> Deque[SequenceGroup]:
        return deque(
            sorted(
                seq_groups,
                key=lambda seq_group: self.get_priority(now, seq_group),
                reverse=True,
            ))


class FCFS(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return now - seq_group.metrics.arrival_time


class MultilevelQueue(Policy):

    def __init__(self, num_levels: int, time_slices: List[int]):
        self.num_levels = num_levels
        self.time_slices = time_slices

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        if not hasattr(seq_group, 'priority') or not hasattr(
                seq_group, 'time_slice'):
            raise ValueError(
                "SequenceGroup must have 'priority' and 'time_slice' attributes"
            )

        if seq_group.priority < 0 or seq_group.priority >= self.num_levels:
            raise ValueError(
                f"Priority must be between 0 and {self.num_levels - 1}")

        return seq_group.priority + (
            now - seq_group.arrival_time) / seq_group.time_slice

    def sort_by_priority(
        self,
        now: float,
        seq_groups: List[SequenceGroup],
    ) -> List[SequenceGroup]:
        return sorted(
            seq_groups,
            key=lambda seq_group: self.get_priority(now, seq_group),
        )


class PolicyFactory:

    _POLICY_REGISTRY = {
        'fcfs': FCFS,
        'multilevel_queue': MultilevelQueue,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        if policy_name not in cls._POLICY_REGISTRY:
            raise ValueError("Unsupported policy name")
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
