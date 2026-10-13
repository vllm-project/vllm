# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record a real deployment's per-step timing in the simulator's step-log
format."""

import time
from collections import deque
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlPullConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from vllm.logger import init_logger
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.dp_utils import should_skip_dp_coordination
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector, KVConnector
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.simulated.kv_transfer import completed_transfers, nixl_reads
from vllm.v1.worker.simulated.timing import (
    StepCost,
    StepDescriptor,
    StepLogWriter,
    accepted_in_step_order,
    build_step_descriptor,
    empty_step_descriptor,
    gather_dp_step,
    is_output_rank,
    parallel_summary,
    step_log_path,
)

logger = init_logger(__name__)


@dataclass
class _StepRecord:
    start: torch.cuda.Event
    end: torch.cuda.Event
    descriptor: StepDescriptor
    host_s: float
    t_start: float
    finished_req_ids: list[str]
    # Set once the step's output is known.
    resolved: bool = False
    num_accepted: tuple[int, ...] | None = None


class _RecordedOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        inner: AsyncModelRunnerOutput,
        record: _StepRecord,
        worker: "RecordingWorker",
    ):
        self.inner = inner
        self.record = record
        self.worker = worker

    def get_output(self) -> ModelRunnerOutput:
        output = self.inner.get_output()
        self.worker._resolve(self.record, output)
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
    return accepted_in_step_order(
        descriptor,
        {r: max(len(sampled[i]) - 1, 0) for r, i in output.req_id_to_index.items()},
    )


class _TimedKVConnector(ActiveKVConnector):
    """Times the reads a NIXL decode rank posts (see `PostCost`)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_caches_dict: dict[str, torch.Tensor],
        recorder: "RecordingWorker",
    ):
        super().__init__(vllm_config, kv_caches_dict)
        self.recorder = recorder

    def _start_load_kv(self) -> None:
        metadata = self.kv_connector._get_connector_metadata()
        assert isinstance(metadata, NixlConnectorMetadata)
        num_reads, num_runs = nixl_reads(metadata)
        if not num_reads:
            return super()._start_load_kv()
        self.recorder._posting_started()
        t = time.perf_counter()
        super()._start_load_kv()
        self.recorder._posted(num_reads, num_runs, time.perf_counter() - t, t)


class _RecordingModelRunner(GPUModelRunnerV2):
    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, recorder: "RecordingWorker"
    ):
        super().__init__(vllm_config, device)
        self.recorder = recorder

    def _make_kv_connector(
        self, kv_caches_dict: dict[str, torch.Tensor]
    ) -> KVConnector:
        if has_kv_transfer_group() and isinstance(
            get_kv_transfer_group(), NixlPullConnector
        ):
            return _TimedKVConnector(self.vllm_config, kv_caches_dict, self.recorder)
        return super()._make_kv_connector(kv_caches_dict)


class RecordingWorker(Worker):
    """GPU worker that logs each step's descriptor and measured timing."""

    def init_device(self) -> None:
        super().init_device()
        self._pending: tuple[torch.cuda.Event, StepDescriptor, float, list[str]] | None
        self._pending = None
        # Records waiting for their step's output or device work, in order.
        self._unwritten: deque[_StepRecord] = deque()
        self._sampling = False
        # The output rank of every DP replica describes its steps; under MoE
        # data parallelism they also join each other's lockstep steps.
        self._describes = is_output_rank()
        self._joins_dp = self._describes and self.parallel_config.data_parallel_size > 1
        log_path = step_log_path(self.vllm_config)
        self._writer = StepLogWriter(log_path) if log_path else None
        self._parallel = parallel_summary(self.vllm_config)
        self._prompt_lens: dict[str, int] = {}
        self._step_index = 0
        # A NIXL decode rank's read postings in the current step: when the
        # first began (on the device stream) and how long they took.
        self._post_event: torch.cuda.Event | None = None
        self._post_s = 0.0
        if self._writer is not None:
            logger.info("Recording per-step timing to %s", log_path)

    def _make_model_runner(self):  # type: ignore[override]
        if self.use_v2_model_runner and not self.vllm_config.is_mm_encoder_only:
            return _RecordingModelRunner(self.vllm_config, self.device, self)
        return super()._make_model_runner()

    def _posting_started(self) -> None:
        # Postings at the end of a step (not synchronous loads before its
        # forward) end its device work.
        if self._sampling and self._post_event is None:
            self._post_event = torch.cuda.Event(enable_timing=True)
            self._post_event.record(torch.cuda.current_stream(self.device))

    def _posted(self, reads: int, runs: int, seconds: float, t: float) -> None:
        if self._sampling:
            self._post_s += seconds
        if self._writer is not None:
            self._writer.write_kv_post(reads, runs, seconds, t)

    def execute_model(self, scheduler_output):  # type: ignore[override]
        if self._unwritten:
            self._write_completed()
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
        if self._syncs_dp():
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

    def _syncs_dp(self) -> bool:
        return self._joins_dp and not should_skip_dp_coordination()

    def execute_dummy_batch(self) -> None:
        if self._syncs_dp():
            gather_dp_step(empty_step_descriptor(self._step_index, self._parallel))
        super().execute_dummy_batch()

    def sample_tokens(self, grammar_output):  # type: ignore[override]
        self._sampling = True
        try:
            output = super().sample_tokens(grammar_output)
        finally:
            self._sampling = False
        pending, self._pending = self._pending, None
        if pending is None or output is None:
            return output
        start, descriptor, t_start, finished = pending
        end, self._post_event = self._post_event, None
        if end is None:
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream(self.device))
        record = _StepRecord(
            start,
            end,
            descriptor,
            host_s=time.perf_counter() - t_start - self._post_s,
            t_start=t_start,
            finished_req_ids=finished,
        )
        self._post_s = 0.0
        self._unwritten.append(record)
        if isinstance(output, AsyncModelRunnerOutput):
            return _RecordedOutput(output, record, self)
        self._resolve(record, output)
        return output

    def _resolve(self, record: _StepRecord, output: ModelRunnerOutput) -> None:
        assert self._writer is not None
        _write_transfers(self._writer, output)
        record.num_accepted = _num_accepted(record.descriptor, output)
        record.resolved = True
        # Outputs are taken in step order: earlier steps still unresolved
        # (warmup steps, whose outputs are dropped) never will be.
        while self._unwritten[0] is not record and not self._unwritten[0].resolved:
            self._unwritten.popleft()
        self._write_completed()

    def _write_completed(self, wait: bool = False) -> None:
        """Write the records, in step order, whose device work has completed
        (with `wait`, all whose output is known)."""
        assert self._writer is not None
        while self._unwritten and self._unwritten[0].resolved:
            record = self._unwritten[0]
            if wait:
                record.end.synchronize()
            elif not record.end.query():
                return
            self._unwritten.popleft()
            device_s = record.start.elapsed_time(record.end) * 1e-3
            self._writer.write(
                record.descriptor,
                StepCost(
                    device_s=device_s,
                    host_s=record.host_s,
                    num_accepted=record.num_accepted,
                ),
                record.t_start,
                record.t_start + max(device_s, record.host_s),
                record.finished_req_ids,
            )

    def shutdown(self) -> None:
        if getattr(self, "_writer", None) is not None:
            self._write_completed(wait=True)
        super().shutdown()
