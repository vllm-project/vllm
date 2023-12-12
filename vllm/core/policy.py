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

        # If the time slice has been exhausted or if the job is new, then update its last execution time to the current time and set its priority to high (returning a smaller value).
        if time_since_last_exec >= self.time_slice or seq_group.is_new:
            return seq_group.arrival_time

        return now

class PriorityQueuesScheduler(Policy):
    def __init__(self):
        super().__init__()
        self.queues = {
            'high': [],
            'medium': [],
            'low': [],
        }

    def add_job(self, seq_group: SequenceGroup, priority: str):
        self.queues[priority].append(seq_group)

    def get_priority(
            self,
            now: float,
            seq_group: SequenceGroup,
    ) -> float:
        # In this strategy, the arrival_time can be combined with priority to determine the return value.
        # A simple method is to represent different priorities with fixed numerical values and then add the arrival time as the priority value.
        # Assume ‘high’ = 1, ‘medium’ = 2, ‘low’ = 3.
        priority_weights = {'high': 1, 'medium': 2, 'low': 3}
        seq_priority = None
        for priority, queue in self.queues.items():
            if seq_group in queue:
                seq_priority = priority
                break

        if seq_priority:
            return priority_weights[seq_priority] + seq_group.arrival_time

        #If a job is not in any queue, a method may be needed to handle it
        return float('inf')


class PolicyFactory:

    _POLICY_REGISTRY = {
        'fcfs': FCFS,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
