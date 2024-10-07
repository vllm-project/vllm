from dataclasses import dataclass
from typing import List

from vllm.inputs.prefill_only.data import Request, SchedulableRequest


@dataclass
class SchedulerOutput:
    pass


@dataclass
class PrefillOnlySchedulerOutput(SchedulerOutput):
    scheduled_requests: List[SchedulableRequest]
    ignored_requests: List[SchedulableRequest]

    def is_empty(self) -> bool:
        return not self.scheduled_requests


@dataclass
class RequestOutput(Request):
    finished: bool
