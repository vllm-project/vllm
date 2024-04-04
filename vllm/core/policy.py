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


class MaximalDecoding(Policy):
    """Policy to prioritize decoding requests as much as possible when a
        queue contains both prefill and decode requests.
    
    It prioritizes 1. seq_group with small number of tokens to compute
    (i.e., decode). 2. FCFS.
    """

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return (-seq_group.get_num_uncomputed_tokens(),
                now - seq_group.metrics.arrival_time)


class PolicyFactory:

    _POLICY_REGISTRY = {'fcfs': FCFS, 'maximal_decoding': MaximalDecoding}

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
