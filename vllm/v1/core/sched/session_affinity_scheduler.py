# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from itertools import islice

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.request import Request, RequestStatus


class SessionAffinityScheduler(AsyncScheduler):
    """Prefer cache-warm continuations from recently scheduled sessions."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.policy != SchedulingPolicy.FCFS:
            raise ValueError("SessionAffinityScheduler only supports FCFS policy")
        config = self.scheduler_config
        if not config.async_scheduling:
            raise ValueError(
                "SessionAffinityScheduler requires asynchronous scheduling"
            )
        self._affinity_window = config.session_affinity_window
        self._affinity_min_blocks = config.session_affinity_min_blocks
        self._affinity_max_wait_s = config.session_affinity_max_wait_s
        self._affinity_ttl_s = config.session_affinity_ttl_s
        self._session_last_scheduled_at: dict[str, float] = {}

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        now = time.time()
        self._prune_expired_sessions(now)
        promotion = self._promote_session_continuation(now)
        scheduler_output = super().schedule(throttle_prefills)
        if promotion is not None:
            promoted_request, original_successor = promotion
            if promoted_request.request_id not in scheduler_output.num_scheduled_tokens:
                self._restore_unscheduled_promotion(
                    promoted_request, original_successor
                )
        for request_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(request_id)
            if request is not None and request.session_id is not None:
                self._session_last_scheduled_at[request.session_id] = now
        return scheduler_output

    def _promote_session_continuation(
        self, now: float
    ) -> tuple[Request, Request | None] | None:
        if not self.cache_config.enable_prefix_caching or len(self.waiting) < 2:
            return None

        candidates = list(islice(self.waiting, self._affinity_window))
        head = candidates[0]
        if head.status == RequestStatus.PREEMPTED:
            return None
        if now - head.arrival_time >= self._affinity_max_wait_s:
            return None

        best_request: Request | None = None
        best_index: int | None = None
        best_key: tuple[int, float, float] | None = None
        for index, request in enumerate(candidates[1:], start=1):
            session_id = request.session_id
            if request.status != RequestStatus.WAITING or session_id is None:
                continue
            last_scheduled_at = self._session_last_scheduled_at.get(session_id)
            if last_scheduled_at is None:
                continue
            num_cached_tokens = self.kv_cache_manager.peek_num_cached_tokens(request)
            num_cached_blocks = num_cached_tokens // self.block_size
            if num_cached_blocks < self._affinity_min_blocks:
                continue
            key = (
                num_cached_blocks,
                last_scheduled_at,
                -request.arrival_time,
            )
            if best_key is None or key > best_key:
                best_request = request
                best_index = index
                best_key = key

        if best_request is not None:
            assert best_index is not None
            original_successor = next(
                islice(self.waiting, best_index + 1, best_index + 2), None
            )
            self.waiting.remove_request(best_request)
            self.waiting.prepend_request(best_request)
            return best_request, original_successor
        return None

    def _restore_unscheduled_promotion(
        self,
        request: Request,
        original_successor: Request | None,
    ) -> None:
        waiting = self.waiting
        if request not in waiting:
            return
        assert isinstance(waiting, FCFSRequestQueue)
        waiting.remove_request(request)
        if original_successor is None:
            waiting.add_request(request)
        else:
            waiting.insert(waiting.index(original_successor), request)

    def _prune_expired_sessions(self, now: float) -> None:
        oldest_allowed = now - self._affinity_ttl_s
        expired = [
            session_id
            for session_id, timestamp in self._session_last_scheduled_at.items()
            if timestamp < oldest_allowed
        ]
        for session_id in expired:
            del self._session_last_scheduled_at[session_id]
