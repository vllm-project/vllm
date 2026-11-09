# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nonblocking GPU iteration timing.

Intervals span model execution, sampling and speculative drafting on the
recorded stream, including host-submission gaps. Independent work on other
streams and EngineCore scheduling are outside this timing scope.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def time_model_execution(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, scheduler_output: SchedulerOutput, *args, **kwargs):
        timer = self.iteration_timer
        # Nested dummy forwards have tokens but no scheduler-assigned ID.
        if timer is None or (
            scheduler_output.iteration_metrics_id is None
            and scheduler_output.total_num_scheduled_tokens > 0
        ):
            return func(self, scheduler_output, *args, **kwargs)
        timer.start(
            scheduler_output.iteration_metrics_id
            if scheduler_output.total_num_scheduled_tokens > 0
            else None
        )
        try:
            output = func(self, scheduler_output, *args, **kwargs)
        except Exception:
            timer.cancel()
            raise
        # A deferred step ends after sampling and speculative drafting.
        if output is not None:
            timer.finish()
        return timer.drain_into(output)

    return wrapper


def finish_iteration_timing(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self: Worker, *args, **kwargs):
        timer = self.model_runner.iteration_timer
        if timer is None:
            return func(self, *args, **kwargs)
        try:
            output = func(self, *args, **kwargs)
        except Exception:
            timer.cancel()
            raise
        timer.finish()
        return timer.drain_into(output)

    return wrapper


def time_dummy_batch(
    func: Callable[..., None],
) -> Callable[..., tuple[tuple[int, float], ...] | None]:
    """Time a dummy step and return previously completed iteration samples."""

    @wraps(func)
    def wrapper(
        self: Worker, iteration_metrics_id: int | None = None
    ) -> tuple[tuple[int, float], ...] | None:
        timer = self.model_runner.iteration_timer
        if iteration_metrics_id is None or timer is None:
            func(self, iteration_metrics_id)
            return None if iteration_metrics_id is None else ()
        # Recycle old ready pairs before queuing the next dummy interval.
        samples = timer.drain_samples()
        timer.start(iteration_metrics_id)
        try:
            func(self, iteration_metrics_id)
        except Exception:
            timer.cancel()
            raise
        timer.finish()
        return samples

    return wrapper


@dataclass(slots=True)
class _PendingTiming:
    iteration_id: int
    start_event: Any
    end_event: Any


class IterationTimer:
    """Pool timing events and collect only completed GPU intervals.

    Output-copy and worker threads may drain concurrently. A contended drain
    returns immediately instead of delaying the next model execution.
    """

    def __init__(
        self,
        num_event_pairs: int,
        event_factory: Callable[[], Any] | None = None,
    ) -> None:
        if num_event_pairs <= 0:
            raise ValueError("num_event_pairs must be positive")
        self._event_factory = event_factory or self._make_event
        self._event_pool: queue.SimpleQueue[tuple[Any, Any]] = queue.SimpleQueue()
        for _ in range(num_event_pairs):
            self._event_pool.put((self._event_factory(), self._event_factory()))
        self._pending: deque[_PendingTiming] = deque()
        self._active: _PendingTiming | None = None
        self._drain_lock = threading.Lock()

    @staticmethod
    def _make_event() -> torch.Event:
        return torch.Event(enable_timing=True)

    def start(self, iteration_id: int | None) -> None:
        if self._active is not None:
            raise RuntimeError("An iteration timing interval is already active")
        if iteration_id is None:
            return
        try:
            start_event, end_event = self._event_pool.get_nowait()
        except queue.Empty:
            # DP dummy steps can outlive the ordinary batch-queue bound.
            logger.warning_once(
                "Iteration CUDA event pool exhausted; allocating one extra pair"
            )
            start_event, end_event = self._event_factory(), self._event_factory()
        start_event.record()
        self._active = _PendingTiming(iteration_id, start_event, end_event)

    def finish(self) -> None:
        if self._active is None:
            return
        pending = self._active
        self._active = None
        pending.end_event.record()
        self._pending.append(pending)

    def cancel(self) -> None:
        # Discard this pair instead of recycling an event still in flight.
        self._active = None

    def drain_samples(self) -> tuple[tuple[int, float], ...]:
        """Return ready (iteration_id, elapsed_ms) pairs without synchronization."""
        if not self._drain_lock.acquire(blocking=False):
            return ()
        ready: list[tuple[int, float]] = []
        try:
            while self._pending:
                # The producer only appends; _drain_lock keeps this head stable.
                pending = self._pending[0]
                if not pending.end_event.query():
                    break

                elapsed_ms = pending.start_event.elapsed_time(pending.end_event)
                self._pending.popleft()
                ready.append((pending.iteration_id, elapsed_ms))
                self._event_pool.put((pending.start_event, pending.end_event))
        finally:
            self._drain_lock.release()
        return tuple(ready)

    def drain_into(self, output: Any) -> Any:
        """Attach ready samples without mutating shared empty runner outputs.

        Async handles collect their samples later in get_output().
        """
        if output is None or not hasattr(output, "iteration_timing_samples"):
            return output
        ready = self.drain_samples()
        if not ready:
            return output
        output = copy(output)
        output.iteration_timing_samples += ready
        return output


def make_iteration_timer(
    vllm_config: VllmConfig,
    rank: int,
) -> IterationTimer | None:
    """Allocate events only when timing is enabled on the model-output rank."""
    if not vllm_config.observability_config.iteration_timing_enabled:
        return None
    parallel_config = vllm_config.parallel_config
    output_rank = parallel_config.world_size - (
        parallel_config.tensor_parallel_size
        * parallel_config.prefill_context_parallel_size
    )
    if rank != output_rank:
        return None
    return IterationTimer(num_event_pairs=vllm_config.max_concurrent_batches + 1)
