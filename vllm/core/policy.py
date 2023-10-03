from typing import List

from vllm.config import PolicyConfig
from vllm.sequence import SequenceGroup


class Policy:

    def __init__(self, **kwargs) -> None:
        pass

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


class PolicyFactory:

    _POLICY_REGISTRY = {
        'fcfs': FCFS,
    }

    @classmethod
    def get_policy(cls, policy_config: PolicyConfig) -> Policy:
        return cls._POLICY_REGISTRY[policy_config.policy](
            **policy_config.policy_kwargs)
