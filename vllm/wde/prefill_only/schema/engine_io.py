from dataclasses import dataclass
from typing import List

from vllm.wde.core.schema.engine_io import SchedulableRequest, SchedulerOutput


@dataclass
class PrefillOnlySchedulerOutput(SchedulerOutput):
    scheduled_requests: List[SchedulableRequest]
    ignored_requests: List[SchedulableRequest]

    def is_empty(self) -> bool:
        return not self.scheduled_requests