# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record per-step timing of a real deployment in the simulator's log format.

`RecordingWorker` wraps the stock GPU worker: it builds the same
`StepDescriptor` the simulated runner would build for the step, brackets the
step's device work with CUDA events and times the host side with the wall
clock. The record is written when the step's output is resolved, so the
measurement never adds a synchronization to the serving path. The resulting
file is the input of the `table` and `replay` timing providers, and the
reference a simulated run of the same deployment is validated against.
"""

import time

import torch

from vllm.logger import init_logger
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.simulated.timing import (
    StepCost,
    StepDescriptor,
    StepLogWriter,
    build_step_descriptor,
    parallel_summary,
)

logger = init_logger(__name__)


class _RecordedOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        inner: AsyncModelRunnerOutput,
        start: torch.cuda.Event,
        end: torch.cuda.Event,
        descriptor: StepDescriptor,
        host_s: float,
        t_start: float,
        finished_req_ids: list[str],
        writer: StepLogWriter,
    ):
        self.inner = inner
        self.start = start
        self.end = end
        self.descriptor = descriptor
        self.host_s = host_s
        self.t_start = t_start
        self.finished_req_ids = finished_req_ids
        self.writer = writer

    def get_output(self) -> ModelRunnerOutput:
        output = self.inner.get_output()
        # The inner output waited for its D2H copy, which was enqueued after
        # the sampler, so the end event is complete; this does not block.
        self.end.synchronize()
        device_s = self.start.elapsed_time(self.end) * 1e-3
        self.writer.write(
            self.descriptor,
            StepCost(
                device_s=device_s,
                host_s=self.host_s,
                num_accepted=_num_accepted(self.descriptor, output),
            ),
            self.t_start,
            self.t_start + max(device_s, self.host_s),
            self.finished_req_ids,
        )
        return output


def _num_accepted(
    descriptor: StepDescriptor, output: ModelRunnerOutput
) -> tuple[int, ...] | None:
    """Accepted drafts per request of a speculative step, in descriptor order."""
    sampled = output.sampled_token_ids
    if not sampled or not any(r.num_spec_tokens for r in descriptor.requests):
        return None
    index = output.req_id_to_index
    return tuple(
        max(len(sampled[index[r.req_id]]) - 1, 0) if r.req_id in index else 0
        for r in descriptor.requests
    )


class RecordingWorker(Worker):
    """GPU worker that logs each step's descriptor and measured timing."""

    def init_device(self) -> None:
        super().init_device()
        sim = self.vllm_config.simulation_config
        if sim.log_path is None:
            raise ValueError("record_step_timing requires simulation_config.log_path")
        self._writer = StepLogWriter(sim.log_path)
        self._parallel = parallel_summary(self.vllm_config)
        self._prompt_lens: dict[str, int] = {}
        self._step_index = 0
        self._pending: tuple[torch.cuda.Event, StepDescriptor, float, list[str]] | None
        self._pending = None
        logger.info("Recording per-step timing to %s", sim.log_path)

    def execute_model(self, scheduler_output):  # type: ignore[override]
        if scheduler_output.total_num_scheduled_tokens == 0:
            # Still let the descriptor builder see finished requests.
            build_step_descriptor(
                scheduler_output, self._prompt_lens, self._step_index, self._parallel
            )
            return super().execute_model(scheduler_output)
        descriptor = build_step_descriptor(
            scheduler_output, self._prompt_lens, self._step_index, self._parallel
        )
        self._step_index += 1
        start = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(self.device))
        t_start = time.perf_counter()
        self._pending = (
            start,
            descriptor,
            t_start,
            list(scheduler_output.finished_req_ids),
        )
        return super().execute_model(scheduler_output)

    def sample_tokens(self, grammar_output):  # type: ignore[override]
        output = super().sample_tokens(grammar_output)
        pending, self._pending = self._pending, None
        if pending is None or output is None:
            return output
        start, descriptor, t_start, finished = pending
        end = torch.cuda.Event(enable_timing=True)
        end.record(torch.cuda.current_stream(self.device))
        host_s = time.perf_counter() - t_start
        if isinstance(output, AsyncModelRunnerOutput):
            return _RecordedOutput(
                output, start, end, descriptor, host_s, t_start, finished, self._writer
            )
        end.synchronize()
        device_s = start.elapsed_time(end) * 1e-3
        self._writer.write(
            descriptor,
            StepCost(
                device_s=device_s,
                host_s=host_s,
                num_accepted=_num_accepted(descriptor, output),
            ),
            t_start,
            t_start + max(device_s, host_s),
            finished,
        )
        return output
