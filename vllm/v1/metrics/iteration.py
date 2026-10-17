# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Associate scheduled batches with asynchronously completed worker timings."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from itertools import count
from typing import TYPE_CHECKING

from vllm.utils.math_utils import WelfordAccumulator
from vllm.v1.metrics.iteration_workload import (
    ScheduledIterationWorkload,
    extract_queued_workload,
)
from vllm.v1.metrics.stats import SchedulerIterationDetails

if TYPE_CHECKING:
    from vllm.v1.core.sched.interface import SchedulerInterface
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.executor.abstract import Executor, IterationTimingPoll
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request

ITERATION_TIMING_SCOPE = "model_step_cuda"
# Retry tail collection when no later model output can carry ready timings.
# This is a CPU wakeup/RPC cadence, not a GPU sampling or measurement interval.
ITERATION_TIMING_POLL_INTERVAL = 0.005
# Best-effort budget for collecting pending timings before executor teardown;
# it does not bound total engine shutdown or guarantee frontend delivery.
ITERATION_TIMING_SHUTDOWN_TIMEOUT = 1.0
ITERATION_IDLE_RECORD_INTERVAL = 1.0


@dataclass
class _PendingIteration:
    details: SchedulerIterationDetails
    output_processed: bool = False
    request_generations: dict[str, tuple[Request, int]] | None = None
    decode_lengths: dict[str, int] | None = None


class IterationMetricsCollector:
    def __init__(
        self,
        scheduler: SchedulerInterface | None = None,
        correct_async_spec_lengths: bool = False,
    ) -> None:
        if scheduler is not None:
            assert scheduler.requests is not None, (
                "Iteration workload metrics require scheduler.requests"
            )
            assert scheduler.waiting is not None, (
                "Iteration workload metrics require scheduler.waiting"
            )
            assert scheduler.kv_holding_waiting is not None, (
                "Iteration workload metrics require scheduler.kv_holding_waiting"
            )
        self._scheduler = scheduler
        self._correct_async_spec_lengths = (
            scheduler is not None and correct_async_spec_lengths
        )
        self._spec_requests: dict[int, dict[str, tuple[Request, int]]] = {}
        self._iteration_ids = count()
        # DP dummy steps can run ahead of the normal batch queue. Keep each
        # association until both output processing and timing have completed.
        self._pending: dict[int, _PendingIteration] = {}
        self._ready: list[SchedulerIterationDetails] = []
        self._timing_poll: IterationTimingPoll | None = None
        self._next_timing_poll = 0.0

    def begin_iteration(self, output: SchedulerOutput) -> None:
        generations = None
        if self._correct_async_spec_lengths:
            assert self._scheduler is not None
            requests = self._scheduler.requests
            assert requests is not None
            generations = {
                rid: (request, request.num_preemptions)
                for rid in output.num_scheduled_tokens
                if (request := requests.get(rid)) is not None
            }
            self._spec_requests[id(output)] = generations
        if output.total_num_scheduled_tokens == 0:
            return
        counts = output.iteration_details
        assert counts is not None
        if self._scheduler is not None:
            assert counts.scheduled_workload is not None
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
                scheduled_workload=counts.scheduled_workload,
            )
        )
        if self._correct_async_spec_lengths:
            cached = output.scheduled_cached_reqs
            assert cached.prefill_req_ids is not None
            pending = self._pending[iteration_id]
            pending.request_generations = generations
            pending.decode_lengths = {
                rid: cached.num_computed_tokens[index]
                for index, rid in enumerate(cached.req_ids)
                if rid not in cached.prefill_req_ids
            }

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
                scheduled_workload=(
                    ScheduledIterationWorkload()
                    if self._scheduler is not None
                    else None
                ),
                queued_workload=(
                    extract_queued_workload(self._scheduler)
                    if self._scheduler is not None
                    else None
                ),
            ),
            output_processed=True,
        )
        return iteration_id

    def make_idle_record(self) -> SchedulerIterationDetails | None:
        """Snapshot engine-confirmed idle; callers establish global quiescence."""
        if self.has_pending_timing():
            return None
        return SchedulerIterationDetails(
            iteration_index=next(self._iteration_ids),
            num_ctx_requests=0,
            num_ctx_tokens=0,
            num_generation_requests=0,
            num_generation_tokens=0,
            elapsed_ms=0.0,
            timing_scope=ITERATION_TIMING_SCOPE,
            is_idle=True,
            scheduled_workload=(
                ScheduledIterationWorkload() if self._scheduler is not None else None
            ),
            queued_workload=(
                extract_queued_workload(self._scheduler)
                if self._scheduler is not None
                else None
            ),
        )

    def before_update(
        self, output: SchedulerOutput, model_output: ModelRunnerOutput
    ) -> None:
        """Correct later snapshots before scheduler stop handling truncates tokens."""
        if not self._correct_async_spec_lengths:
            return
        generations = self._spec_requests.pop(id(output), None)
        if generations is None or not model_output.sampled_token_ids:
            return
        iteration_id = output.iteration_metrics_id
        for rid, drafts in output.scheduled_spec_decode_tokens.items():
            index = model_output.req_id_to_index.get(rid)
            generation = generations.get(rid)
            if index is None or generation is None:
                continue
            sampled = model_output.sampled_token_ids[index]
            if not sampled:
                continue
            rejected = len(drafts) - max(len(sampled) - 1, 0)
            if rejected <= 0:
                continue
            for later_id, later in self._pending.items():
                if (
                    (iteration_id is None or later_id > iteration_id)
                    and later.decode_lengths is not None
                    and rid in later.decode_lengths
                    and later.request_generations is not None
                    and (later_generation := later.request_generations.get(rid))
                    is not None
                    and later_generation[0] is generation[0]
                    and later_generation[1] == generation[1]
                ):
                    later.decode_lengths[rid] -= rejected

    def complete_iteration(
        self, output: SchedulerOutput, model_output: ModelRunnerOutput
    ) -> None:
        self._receive_ready_timing()
        self._spec_requests.pop(id(output), None)
        iteration_id = output.iteration_metrics_id
        if iteration_id is not None:
            pending = self._pending[iteration_id]
            if pending.decode_lengths is not None:
                lengths = WelfordAccumulator()
                for length in pending.decode_lengths.values():
                    lengths.add(length)
                assert pending.details.scheduled_workload is not None
                pending.details.scheduled_workload = replace(
                    pending.details.scheduled_workload,
                    sum_decode_kv_tokens=lengths.total,
                    var_decode_kv_tokens=lengths.variance,
                )
                pending.decode_lengths = None
                pending.request_generations = None
            if self._scheduler is not None:
                pending.details.queued_workload = extract_queued_workload(
                    self._scheduler
                )
            pending.output_processed = True
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
