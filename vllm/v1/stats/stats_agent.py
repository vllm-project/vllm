import threading
from typing import Dict

from vllm.logger import init_logger
from vllm.v1.core.common import SchedulerOutput
from vllm.v1.request import Request
from vllm.v1.stats.common import EngineStatsSnapshot

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
        self._snapshot = self._new_snapshot()

    def _new_snapshot(self) -> EngineStatsSnapshot:
        return EngineStatsSnapshot()

    def record_scheduler_output(
        self,
        requests: Dict[str, Request],
        scheduler_output: SchedulerOutput,
    ):
        for req in scheduler_output.scheduled_running_reqs:
            request = requests[req.req_id]
            self._snapshot.record_running_request(
                request,
                num_computed_tokens=req.num_computed_tokens,
                # For a running sequence, cached tokens are irrelevant.
                num_cached_tokens=None,
            )

        for req in scheduler_output.scheduled_new_reqs:
            request = requests[req.req_id]
            self._snapshot.record_running_request(
                request,
                num_computed_tokens=req.num_computed_tokens,
                # For a new sequence, a computed token is also a cached token.
                num_cached_tokens=req.num_computed_tokens,
            )

        for req in scheduler_output.scheduled_resumed_reqs:
            request = requests[req.req_id]
            self._snapshot.record_running_request(
                request,
                num_computed_tokens=req.num_computed_tokens,
                # For a resumed sequence, a computed token is also a cached
                # token.
                num_cached_tokens=req.num_computed_tokens,
            )

        for req_id in scheduler_output.preempted_req_ids:
            self._snapshot.record_preempted_request(requests[req_id])

    def record_finished_request(self, request: Request):
        self._snapshot.record_finished_request(request)

    def record_waiting_request(self, request: Request):
        self._snapshot.record_waiting_request(request)

    def record_preempted_request(self, request: Request):
        self._snapshot.record_preempted_request(request)

    def record_first_token_ts_ms(self, request: Request):
        self._snapshot.record_first_token_ts_ms(request)

    def get_and_reset_snapshot(self) -> EngineStatsSnapshot:
        snapshot = self._snapshot
        self._snapshot = self._new_snapshot()
        return snapshot


class ThreadSafeEngineStatsAgent(EngineStatsAgent):
    """
    A thread-safe version of EngineStatsAgent.
    """

    def __init__(self, *args, **kwargs):
        self._lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def record_scheduler_output(
            self,
            requests: Dict[str, Request],
            scheduler_output: "SchedulerOutput",  # noqa: F821
    ):
        with self._lock:
            super().record_scheduler_output(requests, scheduler_output)

    def record_finished_request(self, request: Request):
        with self._lock:
            super().record_finished_request(request)

    def record_waiting_request(self, request: Request):
        with self._lock:
            super().record_waiting_request(request)

    def get_and_reset_snapshot(self) -> EngineStatsSnapshot:
        with self._lock:
            return super().get_and_reset_snapshot()

    def record_first_token_ts_ms(self, request: Request):
        with self._lock:
            super().record_first_token_ts_ms(request)
