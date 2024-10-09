from dataclasses import dataclass, field
from typing import Set

from vllm.logger import init_logger
from vllm.wde.core.processor.input_processor import RequestProcessor
from vllm.wde.core.scheduler import Scheduler
from vllm.wde.prefill_only.config import PrefillOnlySchedulerConfig
from vllm.wde.prefill_only.schema.engine_io import (PrefillOnlySchedulerOutput,
                                                    SchedulableRequest)

logger = init_logger(__name__)


@dataclass
class SchedulingBudget:
    token_budget: int
    max_num_requests: int
    _curr_requests: Set[str] = field(default_factory=set)
    _num_batched_tokens: int = 0

    def can_schedule(self, *, num_new_tokens: int, num_new_request: int = 1):
        assert num_new_tokens != 0
        assert num_new_request != 0
        a = self.num_batched_tokens + num_new_tokens <= self.token_budget
        b = self.num_curr_request + num_new_request <= self.max_num_requests
        return a and b

    def add_num_batched_tokens(self, req_id: str, num_batched_tokens: int):
        if req_id in self._curr_requests:
            return

        self._curr_requests.add(req_id)
        self._num_batched_tokens += num_batched_tokens

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_curr_request(self):
        return len(self._curr_requests)


class PrefillOnlyScheduler(Scheduler):
    support_scheduling = ["sync_scheduling", "async_scheduling"]

    def __init__(
        self,
        scheduler_config: PrefillOnlySchedulerConfig,
        request_processor: RequestProcessor,
    ) -> None:
        super().__init__(scheduler_config, request_processor)

    @classmethod
    def from_engine(cls, engine):
        return cls(engine.engine_config.scheduler_config,
                   engine.request_processor)

    def schedule(self) -> PrefillOnlySchedulerOutput:
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_requests=self.scheduler_config.max_num_requests,
        )

        waiting_queue = self.waiting

        scheduled_requests = []
        ignored_requests = []
        while waiting_queue:
            request = waiting_queue[0]
            if request.request_id in self.aborted_requests:
                self.aborted_requests.remove(request.request_id)
                waiting_queue.popleft()
                continue

            if not isinstance(request, SchedulableRequest):
                request = self.request_processor(request)
                waiting_queue[0] = request

            num_new_tokens = request.num_new_tokens

            if num_new_tokens > self.scheduler_config.max_model_len:
                self.requests.remove(request.request_id)
                waiting_queue.popleft()
                ignored_requests.append(request)
                continue

            if not budget.can_schedule(num_new_tokens=num_new_tokens):
                break

            budget.add_num_batched_tokens(request.request_id, num_new_tokens)
            waiting_queue.popleft()
            scheduled_requests.append(request)

        return PrefillOnlySchedulerOutput(
            scheduled_requests=scheduled_requests,
            ignored_requests=ignored_requests)
