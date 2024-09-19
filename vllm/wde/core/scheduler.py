from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Iterable, List, Union

from vllm.logger import init_logger
from vllm.wde.core.config import SchedulerConfig
from vllm.wde.core.llm_engine import LLMEngine
from vllm.wde.core.processor.input_processor import RequestProcessor
from vllm.wde.core.schema.engine_io import (Request, RequestOutput,
                                            SchedulerOutput)

logger = init_logger(__name__)


class Scheduler(ABC):
    support_scheduling = []

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        request_processor: RequestProcessor,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.request_processor = request_processor

        self.waiting: Deque[Request] = deque()

        self.requests = set()
        self.aborted_requests = set()

    @classmethod
    def from_engine(cls, engine: LLMEngine) -> "Scheduler":
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
        self.aborted_requests += request_ids

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
