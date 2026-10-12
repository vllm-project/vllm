# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model runner for simulated execution.

Request bookkeeping (states, block tables, input preparation, sampling
post-processing) is the stock V2 runner's. The forward pass is replaced by a
`TimingProvider` that predicts how long the step would take, and sampling by
trace replay (`SamplingParams.trace_decode_token_ids`) with a fixed fallback
token. The predicted time is paid in `SimulatedAsyncOutput.get_output()`, so
the engine core can keep scheduling the next step meanwhile.
"""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ModelRunnerOutput,
)
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.simulated.timing import (
    RealClock,
    SimulatedAsyncOutput,
    StepCost,
    StepDescriptor,
    StepLogWriter,
    build_step_descriptor,
    get_timing_provider,
    parallel_summary,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

logger = init_logger(__name__)


@dataclass
class _PendingStep:
    input_batch: InputBatch
    descriptor: StepDescriptor
    cost: StepCost
    t_start: float
    finished_req_ids: list[str]


class SimulatedModelRunner(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        sim = vllm_config.simulation_config
        self.timing_provider = get_timing_provider(
            sim.timing, dict(sim.timing_args), vllm_config
        )
        self.clock = RealClock()
        self.step_log = StepLogWriter(sim.log_path) if sim.log_path else None
        self.fallback_token_id = sim.fallback_token_id
        self._pending_step: _PendingStep | None = None
        self._step_index = 0
        # The device is a serial resource: under async scheduling the host
        # prepares step N+1 while step N "runs", but N+1 cannot start on the
        # device before N finishes. Track when the device becomes free.
        self._device_free_at = 0.0
        self._parallel = parallel_summary(vllm_config)
        self._prompt_lens: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Virtual KV cache: specs and block tables are real, tensors are not.
    # ------------------------------------------------------------------ #

    def initialize_kv_cache_tensors(
        self,
        *,
        is_profiling: bool,
        kv_cache_allocation_context: AbstractContextManager | None,
    ) -> None:
        self.kv_caches = []
        logger.info(
            "Initialized virtual KV cache: %d groups, %d blocks; no tensors allocated.",
            len(self.kv_cache_config.kv_cache_groups),
            self.kv_cache_config.num_blocks,
        )

    def _init_kv_zero_meta(self) -> None:
        """No KV tensors, nothing to zero."""

    def _apply_kv_cache_memory_updates(
        self, scheduler_output: "SchedulerOutput"
    ) -> None:
        """No KV tensors, nothing to update."""

    def warming_up_model(self) -> None:
        logger.info("Skipping model warmup under simulated execution.")

    def warm_up_step_kernels(self) -> None:
        """Push one two-token request through the real step path so the Triton
        input-preparation and post-processing kernels are compiled before
        traffic arrives; otherwise the first scheduled step stalls for
        hundreds of milliseconds and distorts early TTFT samples.

        Best effort: a failure costs only the warmup, and the step is neither
        timed nor logged."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

        req_id = "__simulated_warmup__"
        num_groups = len(self.kv_cache_config.kv_cache_groups)
        new_req = NewRequestData(
            req_id=req_id,
            prompt_token_ids=[0, 1],
            mm_features=[],
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_ids=tuple([0] for _ in range(num_groups)),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=[0, 1],
        )
        step = SchedulerOutput.make_empty()
        step.scheduled_new_reqs = [new_req]
        step.num_scheduled_tokens = {req_id: 2}
        step.total_num_scheduled_tokens = 2
        done = SchedulerOutput.make_empty()
        done.finished_req_ids = {req_id}

        step_log, self.step_log = self.step_log, None
        try:
            assert self.execute_model(step) is None
            out = self.sample_tokens(None)
            assert isinstance(out, SimulatedAsyncOutput)
            # Take the result without paying the simulated time.
            assert out.output.sampled_token_ids == [[self.fallback_token_id]]
            self.execute_model(done)
            if self.device.type != "cpu":
                torch.accelerator.synchronize()
        finally:
            self.step_log = step_log
            self._pending_step = None
            self._prompt_lens.clear()
            self._step_index = 0
            self._device_free_at = 0.0

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def execute_model(  # type: ignore[override]
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        context_len: int = 0,
        valid_dummy_state_slots: bool = False,
        randomize_inputs: bool = False,
    ) -> ModelRunnerOutput | None:
        if dummy_run:
            return EMPTY_MODEL_RUNNER_OUTPUT

        self.finish_requests(scheduler_output)
        self.free_states(scheduler_output)
        self.add_requests(scheduler_output)
        self.update_requests(scheduler_output)
        self.block_tables.apply_staged_writes()

        if not scheduler_output.total_num_scheduled_tokens:
            # Nothing to run; the engine core uses this output directly and
            # does not call sample_tokens().
            return EMPTY_MODEL_RUNNER_OUTPUT

        batch_req_state, _ = self.gather_batch_req_state(scheduler_output, dummy_run)
        assert batch_req_state is not None
        input_batch = self.prepare_inputs(
            scheduler_output,
            batch_req_state,
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=batch_req_state.num_tokens,
                num_reqs=len(batch_req_state.req_ids),
            ),
            num_active_loras=0,
        )

        descriptor = build_step_descriptor(
            scheduler_output, self._prompt_lens, self._step_index, self._parallel
        )
        cost = self.timing_provider.estimate(descriptor)
        # Queue the step behind whatever the device is still "executing".
        t_start = max(self.clock.now(), self._device_free_at)
        self._device_free_at = t_start + cost.total_s
        self._pending_step = _PendingStep(
            input_batch=input_batch,
            descriptor=descriptor,
            cost=cost,
            t_start=t_start,
            finished_req_ids=list(scheduler_output.finished_req_ids),
        )
        self._step_index += 1
        # The output is produced by sample_tokens(), as with the real runner.
        return None

    @torch.inference_mode()
    def sample_tokens(  # type: ignore[override]
        self, grammar_output: "GrammarOutput | None"
    ) -> AsyncModelRunnerOutput | ModelRunnerOutput | None:
        pending = self._pending_step
        if pending is None:
            # The prior execute_model call must have failed or scheduled nothing.
            return None
        self._pending_step = None
        input_batch = pending.input_batch
        num_reqs = input_batch.num_reqs

        # Output tokens: the recorded trace where a request has one, otherwise
        # the fallback token. Logits are never computed.
        sampled = torch.full(
            (num_reqs,), self.fallback_token_id, dtype=torch.int64, device=self.device
        )
        assert self.sampler is not None
        trace_state = self.sampler.trace_replay_state
        if trace_state is not None:
            trace_state.apply_trace(sampled, input_batch.idx_mapping)
        sampled_token_ids = sampled.view(-1, 1)

        # A request yields a token only once its prefill is complete.
        seq_lens = input_batch.seq_lens[:num_reqs]
        prefill_lens = torch.from_numpy(input_batch.prefill_len_np).to(
            seq_lens.device, non_blocking=True
        )
        num_sampled = (seq_lens >= prefill_lens).to(torch.int32)
        num_rejected = torch.zeros_like(num_sampled)
        self.postprocess_sampled(
            idx_mapping=input_batch.idx_mapping,
            sampled_tokens=sampled_token_ids,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            query_start_loc=input_batch.query_start_loc,
        )
        sampled_token_ids_list = [
            token_ids[:n]
            for token_ids, n in zip(sampled_token_ids.tolist(), num_sampled.tolist())
        ]
        output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            sampled_token_ids=sampled_token_ids_list,
        )

        deadline = pending.t_start + pending.cost.total_s
        if self.step_log is not None:
            self.step_log.write(
                pending.descriptor,
                pending.cost,
                pending.t_start,
                deadline,
                pending.finished_req_ids,
            )
        return SimulatedAsyncOutput(output, deadline, self.clock)

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return None
