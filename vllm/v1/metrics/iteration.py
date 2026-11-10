# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Associate scheduled batches with asynchronously completed worker timings."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING

from vllm.v1.metrics.stats import SchedulerIterationDetails

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

ITERATION_TIMING_SCOPE = "model_step_cuda"


@dataclass
class _PendingIteration:
    details: SchedulerIterationDetails
    output_processed: bool = False


class IterationMetricsCollector:
    def __init__(self) -> None:
        self._iteration_ids = count()
        # DP dummy steps can run ahead of the normal batch queue. Keep their
        # associations until timings arrive or the engine enters idle.
        self._pending: dict[int, _PendingIteration] = {}
        self._ready: list[SchedulerIterationDetails] = []

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
                num_ctx_requests=counts.num_ctx_requests,
                num_ctx_tokens=counts.num_ctx_tokens,
                num_generation_requests=counts.num_generation_requests,
                num_generation_tokens=counts.num_generation_tokens,
                elapsed_ms=None,
                num_encoder_inputs=counts.num_encoder_inputs,
                num_encoder_output_tokens=counts.num_encoder_output_tokens,
                timing_scope=ITERATION_TIMING_SCOPE,
            )
        )

    def begin_dummy_iteration(self) -> int:
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

    def discard_pending(self) -> None:
        """Drop idle-tail records; late samples cannot attach to a new iteration.

        Workers retain their CUDA events until ready for normal recycling.
        Discarding records does not imply GPU completion.
        """
        self._pending.clear()
