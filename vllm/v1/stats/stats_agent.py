import threading
from typing import List

from vllm.logger import init_logger
from vllm.v1.core.common import SchedulerOutput
from vllm.v1.request import Request
from vllm.v1.stats.common import RequestStatsUpdate

logger = init_logger(__name__)


class EngineStatsAgent:
    """
    EngineStatsAgent accumulates stats from the EngineCore and materialize a
    EngineStatsSnapshot to be merged by the EngineStatsManager.

    - In the multi-process architecture (e.g. AsyncMPClient or SyncMPClient),
      this is running in the background engine core process.
    - In the single-process architecture, this is running in the same process
      as the LLMEngine.

    TODO(rickyx): We could further optimize the stats update by isolating the
    stats polling from the IO thread.
    """

    def __init__(self):
        self._updates: List[RequestStatsUpdate] = []

    def record_scheduler_output(
        self,
        scheduler_output: SchedulerOutput,
    ):
        for req in scheduler_output.scheduled_running_reqs:
            self._updates.append(
                RequestStatsUpdate(
                    request_id=req.req_id,
                    type="running",
                    was_running=True,
                    num_computed_tokens=req.num_computed_tokens,
                    # For a running sequence, cached tokens are irrelevant.
                    num_cached_tokens=None,
                ))

        for req in scheduler_output.scheduled_new_reqs:
            self._updates.append(
                RequestStatsUpdate(
                    request_id=req.req_id,
                    type="running",
                    was_running=False,
                    num_computed_tokens=req.num_computed_tokens,
                    # For a new sequence, a computed token is also a cached
                    # token.
                    num_cached_tokens=req.num_computed_tokens,
                ))

        for req in scheduler_output.scheduled_resumed_reqs:
            self._updates.append(
                RequestStatsUpdate(
                    request_id=req.req_id,
                    type="running",
                    was_running=False,
                    num_computed_tokens=req.num_computed_tokens,
                    # For a resumed sequence, a computed token is also a cached
                    # token.
                    num_cached_tokens=req.num_computed_tokens,
                ))

        for req_id in scheduler_output.preempted_req_ids:
            self._updates.append(
                RequestStatsUpdate(
                    request_id=req_id,
                    type="preempted",
                ))

    def record_queued_request(self, request: Request):
        self._updates.append(
            RequestStatsUpdate(
                request_id=request.request_id,
                type="queued",
            ))

    def take_requests_updates(self) -> List[RequestStatsUpdate]:
        updates = self._updates
        self._updates = []
        return updates


class ThreadSafeEngineStatsAgent(EngineStatsAgent):
    """
    A thread-safe version of EngineStatsAgent.
    """

    def __init__(self, *args, **kwargs):
        self._lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def record_scheduler_output(
            self,
            scheduler_output: "SchedulerOutput",  # noqa: F821
    ):
        with self._lock:
            super().record_scheduler_output(scheduler_output)

    def record_queued_request(self, request: Request):
        with self._lock:
            super().record_queued_request(request)

    def take_requests_updates(self) -> List[RequestStatsUpdate]:
        with self._lock:
            return super().take_requests_updates()
