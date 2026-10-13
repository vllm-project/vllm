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
from typing import Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlPullConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.dp_utils import should_skip_dp_coordination
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.simulated.kv_transfer import completed_transfers, nixl_reads
from vllm.v1.worker.simulated.timing import (
    StepCost,
    StepDescriptor,
    StepLogWriter,
    build_step_descriptor,
    empty_step_descriptor,
    gather_dp_step,
    parallel_summary,
    step_log_path,
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
        _write_transfers(self.writer, output)
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


def _write_transfers(
    writer: StepLogWriter, output: ModelRunnerOutput | AsyncModelRunnerOutput | None
) -> None:
    """Log the KV transfers this rank saw complete in the step."""
    if isinstance(output, ModelRunnerOutput):
        transfers = completed_transfers(output.kv_connector_output)
        if transfers:
            writer.write_transfers(transfers, time.perf_counter())


def _num_accepted(
    descriptor: StepDescriptor, output: ModelRunnerOutput
) -> tuple[int, ...] | None:
    """Accepted drafts per request of a speculative step, in descriptor order."""
    sampled = output.sampled_token_ids
    if not sampled or not any(r.num_spec_tokens for r in descriptor.requests):
        return None
    index = output.req_id_to_index
    if not all(r.req_id in index for r in descriptor.requests):
        # A joint MoE-DP step: the other ranks' acceptance is not known here.
        return None
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
        self._pending: tuple[torch.cuda.Event, StepDescriptor, float, list[str]] | None
        self._pending = None
        # The output rank of every DP replica describes its steps; under MoE
        # data parallelism they also join each other's lockstep steps.
        self._describes = (
            get_pp_group().is_last_rank and get_tp_group().rank_in_group == 0
        )
        self._joins_dp = self._describes and self.parallel_config.data_parallel_size > 1
        log_path = step_log_path(self.vllm_config)
        self._writer = StepLogWriter(log_path) if log_path else None
        self._parallel = parallel_summary(self.vllm_config)
        self._prompt_lens: dict[str, int] = {}
        self._step_index = 0
        # A NIXL decode rank's read postings in the current step: when the
        # first began (on the device stream) and how long they took.
        self._posts_wrapped = False
        self._post_event: torch.cuda.Event | None = None
        self._post_s = 0.0
        if self._writer is not None:
            logger.info("Recording per-step timing to %s", log_path)

    def _wrap_read_postings(self) -> None:
        """Time the reads a NIXL decode rank posts at the end of a step. They
        hold the engine thread, and the device idles once the step's work is
        done, so the step's record leaves them out and they are logged on
        their own (see `PostCost`)."""
        self._posts_wrapped = True
        connector = getattr(
            getattr(self.model_runner, "kv_connector", None), "kv_connector", None
        )
        writer = self._writer
        if not isinstance(connector, NixlPullConnector) or writer is None:
            return
        start_load_kv = connector.start_load_kv

        def timed_start_load_kv(forward_context: Any, **kwargs: Any) -> None:
            metadata = connector._get_connector_metadata()
            assert isinstance(metadata, NixlConnectorMetadata)
            num_reads, num_runs = nixl_reads(metadata)
            if not num_reads:
                return start_load_kv(forward_context, **kwargs)
            if self._pending is not None and self._post_event is None:
                self._post_event = torch.cuda.Event(enable_timing=True)
                self._post_event.record(torch.cuda.current_stream(self.device))
            t = time.perf_counter()
            start_load_kv(forward_context, **kwargs)
            seconds = time.perf_counter() - t
            if self._pending is not None:
                self._post_s += seconds
            writer.write_kv_post(num_reads, num_runs, seconds, t)

        connector.start_load_kv = timed_start_load_kv  # type: ignore[method-assign]

    def execute_model(self, scheduler_output):  # type: ignore[override]
        if not self._posts_wrapped:
            self._wrap_read_postings()
        if not self._describes:
            return super().execute_model(scheduler_output)
        descriptor = build_step_descriptor(
            scheduler_output, self._prompt_lens, self._step_index, self._parallel
        )
        if scheduler_output.total_num_scheduled_tokens == 0:
            output = super().execute_model(scheduler_output)
            if self._writer is not None:
                _write_transfers(self._writer, output)
            return output
        self._step_index += 1
        if self._joins_dp and not should_skip_dp_coordination():
            descriptor = gather_dp_step(descriptor)
        if self._writer is not None:
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

    def execute_dummy_batch(self) -> None:
        if self._joins_dp and not should_skip_dp_coordination():
            gather_dp_step(empty_step_descriptor(self._step_index, self._parallel))
        super().execute_dummy_batch()

    def sample_tokens(self, grammar_output):  # type: ignore[override]
        output = super().sample_tokens(grammar_output)
        pending, self._pending = self._pending, None
        if pending is None or output is None:
            return output
        start, descriptor, t_start, finished = pending
        # A step is pending only when this rank writes a log.
        writer = self._writer
        assert writer is not None
        end, self._post_event = self._post_event, None
        if end is None:
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream(self.device))
        host_s = time.perf_counter() - t_start - self._post_s
        self._post_s = 0.0
        if isinstance(output, AsyncModelRunnerOutput):
            return _RecordedOutput(
                output, start, end, descriptor, host_s, t_start, finished, writer
            )
        end.synchronize()
        _write_transfers(writer, output)
        device_s = start.elapsed_time(end) * 1e-3
        writer.write(
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
