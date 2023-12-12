from typing import List

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
        seq_groups: List[SequenceGroup],
    ) -> List[SequenceGroup]:
        return sorted(
            seq_groups,
            key=lambda seq_group: self.get_priority(now, seq_group),
            reverse=True,
        )


class FCFS(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return now - seq_group.arrival_time

class RoundRobin(Policy):
    def __init__(self, time_slice: float):
        self.time_slice = time_slice
        self.current_time = 0
        self.last_exec_time = {}

    def get_priority(
            self,
            now: float,
            seq_group: SequenceGroup,
    ) -> float:
        self.current_time = now

        if seq_group not in self.last_exec_time:
            self.last_exec_time[seq_group] = now

        time_since_last_exec = now - self.last_exec_time[seq_group]

        if time_since_last_exec >= self.time_slice or seq_group.is_new:
            return seq_group.arrival_time

        return now

class PolicyFactory:

    _POLICY_REGISTRY = {
        'fcfs': FCFS,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
