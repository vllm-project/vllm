from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, List, Set, Union, cast

from vllm.config import SchedulerConfig
from vllm.inputs.prefill_only.data import Request, SchedulableRequest
from vllm.inputs.prefill_only.preprocessor import RequestProcessor
from vllm.logger import init_logger
from vllm.model_executor.prefill_only.engine_io import (
    PrefillOnlySchedulerOutput, RequestOutput, SchedulerOutput)

logger = init_logger(__name__)


class Scheduler(ABC):
    support_scheduling: List[str] = []

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        request_processor: RequestProcessor,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.request_processor = request_processor

        self.waiting: Deque[Request] = deque()

        self.requests: Set[str] = set()
        self.aborted_requests: Set[str] = set()

    @classmethod
    def from_engine(cls, engine) -> "Scheduler":
        raise NotImplementedError

    def add_request(self, request: Request) -> None:
        if (request.request_id in self.requests
                or request.request_id in self.aborted_requests):
            logger.warning("[%s] request_id conflict")
            return

        self.waiting.append(request)
        self.requests.add(request.request_id)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)

        self.requests -= request_ids
        self.aborted_requests |= request_ids

    def remove_abort_request(
            self, request_outputs: List[RequestOutput]) -> List[RequestOutput]:
        if len(self.aborted_requests) == 0:
            return request_outputs

        current_ids = set(request.request_id for request in request_outputs)
        need_abort = self.aborted_requests & current_ids

        if len(need_abort) == 0:
            return request_outputs

        request_outputs = [
            request for request in request_outputs
            if request.request_id not in need_abort
        ]
        self.aborted_requests -= need_abort

        return request_outputs

    def has_unfinished_requests(self) -> bool:
        return len(self.requests) != 0

    def get_num_unfinished_requests(self) -> int:
        return len(self.requests)

    @abstractmethod
    def schedule(self) -> SchedulerOutput:
        raise NotImplementedError

    def free_finished_request(self, request_outputs: List[RequestOutput]):
        finished_request_ids = set(request.request_id
                                   for request in request_outputs
                                   if request.finished)
        self.requests -= finished_request_ids


@dataclass
class PrefillOnlySchedulingBudget:
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
        scheduler_config: SchedulerConfig,
        request_processor: RequestProcessor,
    ) -> None:
        super().__init__(scheduler_config, request_processor)

    @classmethod
    def from_engine(cls, engine):
        return cls(engine.engine_config.scheduler_config,
                   engine.request_processor)

    def schedule(self) -> PrefillOnlySchedulerOutput:
        budget = PrefillOnlySchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_requests=self.scheduler_config.max_num_seqs,
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

            request = cast(SchedulableRequest, request)

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
