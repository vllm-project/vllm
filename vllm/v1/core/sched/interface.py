# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.metrics.stats import SchedulerStats
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus


class SchedulerInterface(ABC):

    @abstractmethod
    def schedule(self) -> "SchedulerOutput":
        raise NotImplementedError

    @abstractmethod
    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> "EngineCoreOutputs":
        raise NotImplementedError

    @abstractmethod
    def add_request(self, request: "Request") -> None:
        raise NotImplementedError

    @abstractmethod
    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: "RequestStatus",
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        raise NotImplementedError

    def has_unfinished_requests(self) -> bool:
        return self.get_num_unfinished_requests() > 0

    @abstractmethod
    def has_finished_requests(self) -> bool:
        raise NotImplementedError

    def has_requests(self) -> bool:
        """Returns True if there are unfinished requests, or finished requests
        not yet returned in SchedulerOutputs."""
        return self.has_unfinished_requests() or self.has_finished_requests()

    @abstractmethod
    def get_num_unscheduled_requests(self) -> int:
        """Number of requests that are not being processed by the executor."""
        raise NotImplementedError

    @abstractmethod
    def reset_prefix_cache(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def make_stats(self) -> Optional["SchedulerStats"]:
        raise NotImplementedError
