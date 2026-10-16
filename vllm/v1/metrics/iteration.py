# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Associate scheduled batches with asynchronously completed worker timings."""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING

from vllm.v1.metrics.stats import SchedulerIterationDetails

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.executor.abstract import Executor, IterationTimingPoll
    from vllm.v1.outputs import ModelRunnerOutput

ITERATION_TIMING_SCOPE = "model_step_cuda"
# Retry tail collection when no later model output can carry ready timings.
# This is a CPU wakeup/RPC cadence, not a GPU sampling or measurement interval.
ITERATION_TIMING_POLL_INTERVAL = 0.005
# Best-effort budget for collecting pending timings before executor teardown;
# it does not bound total engine shutdown or guarantee frontend delivery.
ITERATION_TIMING_SHUTDOWN_TIMEOUT = 1.0


@dataclass
class _PendingIteration:
    details: SchedulerIterationDetails
    output_processed: bool = False


class IterationMetricsCollector:
    def __init__(self) -> None:
        self._iteration_ids = count()
        # DP dummy steps can run ahead of the normal batch queue. Keep each
        # association until both output processing and timing have completed.
        self._pending: dict[int, _PendingIteration] = {}
        self._ready: list[SchedulerIterationDetails] = []
        self._timing_poll: IterationTimingPoll | None = None
        self._next_timing_poll = 0.0

    def begin_iteration(self, output: SchedulerOutput) -> None:
        if output.total_num_scheduled_tokens == 0:
            return
        counts = output.iteration_details
        assert counts is not None
        iteration_id = next(self._iteration_ids)
        output.iteration_metrics_id = iteration_id
        self._pending[iteration_id] = _PendingIteration(
            SchedulerIterationDetails(
                iteration_index=iteration_id,
                num_ctx_requests=counts.num_prefill_requests,
                num_ctx_tokens=counts.num_prefill_tokens,
                num_generation_requests=counts.num_decode_requests,
                num_generation_tokens=counts.num_decode_tokens,
                elapsed_ms=None,
                num_encoder_inputs=counts.num_encoder_inputs,
                num_encoder_output_tokens=counts.num_encoder_output_tokens,
                timing_scope=ITERATION_TIMING_SCOPE,
            )
        )

    def begin_dummy_iteration(self) -> int:
        self._receive_ready_timing()
        iteration_id = next(self._iteration_ids)
        self._pending[iteration_id] = _PendingIteration(
            SchedulerIterationDetails(
                iteration_index=iteration_id,
                num_ctx_requests=0,
                num_ctx_tokens=0,
                num_generation_requests=0,
                num_generation_tokens=0,
                elapsed_ms=None,
                is_dummy=True,
                timing_scope=ITERATION_TIMING_SCOPE,
            ),
            output_processed=True,
        )
        return iteration_id

    def complete_iteration(
        self, output: SchedulerOutput, model_output: ModelRunnerOutput
    ) -> None:
        self._receive_ready_timing()
        iteration_id = output.iteration_metrics_id
        if iteration_id is not None:
            self._pending[iteration_id].output_processed = True
            self._complete_if_ready(iteration_id)
        # A real output can carry earlier dummy timings, even on a zero-token step.
        self.complete_timing_samples(model_output.iteration_timing_samples)

    def complete_timing_samples(self, samples: tuple[tuple[int, float], ...]) -> None:
        for iteration_id, elapsed_ms in samples:
            pending = self._pending.get(iteration_id)
            if pending is not None:
                pending.details.elapsed_ms = elapsed_ms
                self._complete_if_ready(iteration_id)

    def _complete_if_ready(self, iteration_id: int) -> None:
        pending = self._pending[iteration_id]
        if pending.output_processed and pending.details.elapsed_ms is not None:
            self._ready.append(pending.details)
            del self._pending[iteration_id]

    def take_ready(self) -> list[SchedulerIterationDetails]:
        ready, self._ready = self._ready, []
        return ready

    def has_pending_timing(self) -> bool:
        return bool(self._pending) or self._timing_poll is not None

    def _receive_ready_timing(self) -> None:
        if self._timing_poll is not None:
            samples = self._timing_poll()
            if samples is None:
                return
            self._timing_poll = None
            self.complete_timing_samples(samples)

    def poll_timing(self, executor: Executor) -> None:
        self._receive_ready_timing()
        if self._timing_poll is not None:
            return
        now = time.monotonic()
        if not self._pending or now < self._next_timing_poll:
            return
        self._next_timing_poll = now + ITERATION_TIMING_POLL_INTERVAL
        self._timing_poll = executor.start_iteration_timing_poll()
