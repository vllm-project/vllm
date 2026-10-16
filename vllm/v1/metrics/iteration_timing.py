# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nonblocking GPU iteration timing shared by both model runners.

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
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


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
        self._pending_lock = threading.Lock()
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
        with self._pending_lock:
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
            while True:
                if not self._pending_lock.acquire(blocking=False):
                    break
                try:
                    pending = self._pending[0] if self._pending else None
                finally:
                    self._pending_lock.release()
                if pending is None or not pending.end_event.query():
                    break

                elapsed_ms = pending.start_event.elapsed_time(pending.end_event)
                if not self._pending_lock.acquire(blocking=False):
                    break
                try:
                    self._pending.popleft()
                finally:
                    self._pending_lock.release()
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
