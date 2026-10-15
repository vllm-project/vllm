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
    from vllm.sequence import IntermediateTensors
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

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
        """Return ready (iteration ID, seconds) pairs without synchronization."""
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
                    completed = self._pending.popleft()
                finally:
                    self._pending_lock.release()
                assert completed is pending
                ready.append((pending.iteration_id, elapsed_ms * 1e-3))
                self._event_pool.put((pending.start_event, pending.end_event))
        finally:
            self._drain_lock.release()
        return tuple(ready)

    def drain_into(self, output: Any) -> Any:
        """Attach ready samples without mutating shared empty runner outputs."""
        if output is None or not hasattr(output, "iteration_timing_samples"):
            return output
        ready = self.drain_samples()
        if not ready:
            return output
        output = copy(output)
        output.iteration_timing_samples += ready
        return output


def execute_model_with_timing(
    model_runner: GPUModelRunner | GPUModelRunnerV2,
    scheduler_output: SchedulerOutput,
    intermediate_tensors: IntermediateTensors | None,
) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
    timer = model_runner.iteration_timer
    if timer is None:
        return model_runner.execute_model(scheduler_output, intermediate_tensors)
    timer.start(
        scheduler_output.iteration_metrics_id
        if scheduler_output.total_num_scheduled_tokens > 0
        else None
    )
    try:
        output = model_runner.execute_model(scheduler_output, intermediate_tensors)
    except Exception:
        timer.cancel()
        raise
    # A deferred step ends after sampling and speculative drafting.
    if output is not None:
        timer.finish()
    return output


def sample_tokens_with_timing(
    model_runner: GPUModelRunner | GPUModelRunnerV2,
    grammar_output: GrammarOutput | None,
) -> ModelRunnerOutput | AsyncModelRunnerOutput:
    timer = model_runner.iteration_timer
    if timer is None:
        return model_runner.sample_tokens(grammar_output)
    try:
        output = model_runner.sample_tokens(grammar_output)
    except Exception:
        timer.cancel()
        raise
    timer.finish()
    return timer.drain_into(output)


def execute_dummy_with_timing(
    model_runner: GPUModelRunner | GPUModelRunnerV2,
    iteration_id: int | None,
    *,
    randomize_inputs: bool,
) -> tuple[tuple[int, float], ...] | None:
    num_tokens = getattr(model_runner, "uniform_decode_query_len", 1)
    timer = model_runner.iteration_timer
    if iteration_id is None or timer is None:
        model_runner._dummy_run(
            num_tokens, uniform_decode=True, randomize_inputs=randomize_inputs
        )
        return None if iteration_id is None else ()
    # Recycle old ready pairs before queuing the next dummy interval.
    samples = timer.drain_samples()
    timer.start(iteration_id)
    try:
        model_runner._dummy_run(
            num_tokens, uniform_decode=True, randomize_inputs=randomize_inputs
        )
    except Exception:
        timer.cancel()
        raise
    timer.finish()
    return samples


def make_iteration_timer(
    vllm_config: VllmConfig,
    is_output_rank: bool,
) -> IterationTimer | None:
    """Allocate events only when timing is enabled on the model-output rank."""
    if (
        not vllm_config.observability_config.enable_iteration_timing
        or not is_output_rank
    ):
        return None
    return IterationTimer(
        num_event_pairs=max(2, vllm_config.max_concurrent_batches + 1)
    )


def is_iteration_output_rank(vllm_config: VllmConfig, rank: int) -> bool:
    """Match the executor's ModelRunnerOutput rank within a DP replica."""
    parallel_config = vllm_config.parallel_config
    output_rank = parallel_config.world_size - (
        parallel_config.tensor_parallel_size
        * parallel_config.prefill_context_parallel_size
    )
    return rank == output_rank
