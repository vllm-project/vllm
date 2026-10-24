# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model runner for simulated execution: the V2 runner's bookkeeping, with each
step's forward and sampling replaced by its predicted time and trace or
fallback tokens."""

import dataclasses
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.distributed.parallel_state import get_dp_group, get_pp_group
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    KVConnectorOutput,
    ModelRunnerOutput,
)
from vllm.v1.worker.dp_utils import should_skip_dp_coordination
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.simulated.kv_transfer import (
    SimulatedTransfers,
    completed_transfers,
    simulated_kv_connector,
)
from vllm.v1.worker.simulated.timing import (
    Clock,
    RealClock,
    SimulatedAsyncOutput,
    StepCost,
    StepDescriptor,
    StepLogWriter,
    ZeroTimingProvider,
    acceptance_rates_from_log,
    accepted_in_step_order,
    build_step_descriptor,
    driven_engine_clock,
    empty_step_descriptor,
    gather_dp_step,
    get_timing_provider,
    new_clock,
    parallel_summary,
    step_log_path,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

logger = init_logger(__name__)

_WARMUP_REQ_ID = "__simulated_warmup__"
_WARMUP_PROMPT_LEN = 2
# The draft token id the runner reports for drafts it does not propose.
_PLACEHOLDER_DRAFT = -1


@dataclass
class _PendingStep:
    input_batch: InputBatch
    descriptor: StepDescriptor
    cost: StepCost
    t_entry: float
    t_start: float
    t_deadline: float
    finished_req_ids: set[str]
    num_spec_tokens_to_schedule: int


class SimulatedModelRunner(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        sim = vllm_config.simulation_config
        self.timing_provider = get_timing_provider(
            sim.timing, dict(sim.timing_args), vllm_config
        )
        if sim.clock == "real":
            self.clock: Clock = new_clock(sim.clock)
        else:
            clock = driven_engine_clock()
            if clock is None:
                raise ValueError(
                    f"The {sim.clock} clock needs the engine core in the process "
                    "of a driver reading the same clock; build the engine with "
                    "vllm.v1.worker.simulated.offline.make_engine()."
                )
            self.clock = clock
        log_path = step_log_path(vllm_config)
        self.step_log = StepLogWriter(log_path) if log_path else None
        self.fallback_token_id = sim.fallback_token_id
        self._pending_step: _PendingStep | None = None
        self._step_index = 0
        self._warming_up = False
        # When each pipeline stage's device becomes free; every rank models
        # the whole pipeline, so all agree on every deadline.
        self._stage_free_at = [0.0] * self.parallel_config.pipeline_parallel_size
        self._kv_transfers: SimulatedTransfers | None = None
        self._last_output: SimulatedAsyncOutput | None = None
        self._parallel = parallel_summary(vllm_config)
        self._prompt_lens: dict[str, int] = {}
        self._rng = np.random.default_rng(vllm_config.model_config.seed)
        self._acceptance_rates = None
        spec = vllm_config.speculative_config
        if spec is not None and spec.enable_adaptive_verification:
            raise ValueError(
                "Simulated execution does not support adaptive verification."
            )
        if vllm_config.model_config.is_diffusion:
            raise ValueError("Simulated execution does not support diffusion models.")
        if self.num_speculative_steps > 0:
            rates = sim.spec_acceptance_rates
            if rates is None:
                raise ValueError(
                    "Simulated speculative decoding needs "
                    "simulation_config.spec_acceptance_rates: per-position "
                    "acceptance probabilities, or the path of a step log "
                    "recorded with speculative decoding."
                )
            if isinstance(rates, str):
                rates = acceptance_rates_from_log(rates)
            self._acceptance_rates = np.asarray(rates, dtype=np.float64)
            logger.info("Draft acceptance rates by position: %s", rates)

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
        if not is_profiling and has_kv_transfer_group():
            # This rank's copy streams wait for its own pipeline stage.
            stage = get_pp_group().rank_in_group
            connector = simulated_kv_connector(
                self.vllm_config,
                self.kv_cache_config,
                self.clock,
                device_free_at=lambda: self._stage_free_at[stage],
            )
            self.kv_connector = connector
            self._kv_transfers = connector.transfers

    def _init_kv_zero_meta(self) -> None:
        """No KV tensors, nothing to zero."""

    def _apply_kv_cache_memory_updates(
        self, scheduler_output: "SchedulerOutput"
    ) -> None:
        """No KV tensors, nothing to update."""

    def warming_up_model(self) -> None:
        logger.info("Skipping model warmup under simulated execution.")

    def warm_up_step_kernels(self) -> None:
        """Compile the step path's Triton kernels with an unpriced, unlogged
        step, before the first request's step would."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

        req_id = _WARMUP_REQ_ID
        prompt = [0] * _WARMUP_PROMPT_LEN
        num_groups = len(self.kv_cache_config.kv_cache_groups)
        new_req = NewRequestData(
            req_id=req_id,
            prompt_token_ids=prompt,
            mm_features=[],
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_ids=tuple([0] for _ in range(num_groups)),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=prompt,
        )
        step = SchedulerOutput.make_empty()
        step.scheduled_new_reqs = [new_req]
        step.num_scheduled_tokens = {req_id: len(prompt)}
        step.total_num_scheduled_tokens = len(prompt)
        done = SchedulerOutput.make_empty()
        done.finished_req_ids = {req_id}

        step_log, self.step_log = self.step_log, None
        provider = self.timing_provider
        self.timing_provider = ZeroTimingProvider(self.vllm_config)
        # Ranks warm up independently; keep DP ranks out of the step sync.
        self._warming_up = True
        self.kv_connector.set_disabled(True)
        try:
            self.execute_model(step)
            out = self.sample_tokens(None)
            # Take the result without paying the simulated time.
            if not (
                isinstance(out, SimulatedAsyncOutput)
                and out.output.sampled_token_ids == [[self.fallback_token_id]]
            ):
                raise RuntimeError(f"Unexpected simulated warmup output: {out}")
            if self.device.type != "cpu":
                torch.accelerator.synchronize()
            self.execute_model(done)
        finally:
            self.step_log = step_log
            self.timing_provider = provider
            self._pending_step = None
            self._last_output = None
            self._prompt_lens.clear()
            self._step_index = 0
            self._stage_free_at = [0.0] * len(self._stage_free_at)
            self._warming_up = False
            self.kv_connector.set_disabled(False)

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
        # Blocks until any stores the scheduler must flush have completed.
        self.kv_connector.pre_forward(scheduler_output)
        t_entry = self.clock.now()
        self._update_request_states(scheduler_output)

        if not scheduler_output.total_num_scheduled_tokens:
            # Nothing to run; the engine core uses this output directly and
            # does not call sample_tokens().
            self.kv_connector.finish_forward()
            kv_output = self._post_forward(scheduler_output.finished_req_ids)
            self._skip_to_transfer()
            return ModelRunnerOutput.with_kv_conn_output_only(kv_output)

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
        if self.pcp_manager is not None:
            # Samples over the whole batch, not this rank's PCP shard.
            input_batch = self.pcp_manager.global_batch

        descriptor = build_step_descriptor(
            scheduler_output, self._prompt_lens, self._step_index, self._parallel
        )
        descriptor, now = self._dp_joint_step(descriptor, t_entry)
        cost = self._estimate(descriptor)
        # Queue the step behind whatever the device is still "executing". It
        # cannot complete before its host work (launches, sampling) is done.
        t_start = self._agree_start(max(t_entry, self._stage_free_at[0]), now)
        device_end = self._run_pipeline(t_start, cost.device_s)
        self._pending_step = _PendingStep(
            input_batch=input_batch,
            descriptor=descriptor,
            cost=cost,
            t_entry=t_entry,
            t_start=t_start,
            t_deadline=max(device_end, t_entry + cost.host_s),
            finished_req_ids=scheduler_output.finished_req_ids,
            num_spec_tokens_to_schedule=scheduler_output.num_spec_tokens_to_schedule,
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

        # A request yields tokens only once its prefill is complete; then one
        # per accepted draft plus the bonus token.
        completes = (
            input_batch.num_computed_tokens_np + input_batch.num_scheduled_tokens
            >= input_batch.prefill_len_np
        )
        row = {req_id: i for i, req_id in enumerate(input_batch.req_ids)}
        num_drafts = input_batch.num_draft_tokens_per_req
        if num_drafts is None or input_batch.num_draft_tokens == 0:
            num_drafts = np.zeros(num_reqs, dtype=np.int32)
            num_accepted = num_drafts
        else:
            num_accepted = np.where(
                completes, self._num_accepted(pending, input_batch, num_drafts), 0
            )
            pending.cost = dataclasses.replace(
                pending.cost,
                num_accepted=accepted_in_step_order(
                    pending.descriptor,
                    {r: int(num_accepted[i]) for r, i in row.items()},
                ),
            )
        num_sampled_np = np.where(completes, 1 + num_accepted, 0).astype(np.int32)

        # Output tokens: the request's trace, else the fallback token.
        sampled = torch.full(
            (num_reqs, 1 + self.num_speculative_steps),
            self.fallback_token_id,
            dtype=torch.int64,
            device=self.device,
        )
        # Only the last pipeline stage has a sampler; the token values matter
        # only there (the counts must agree on all ranks).
        trace_state = self.sampler.trace_replay_state if self.sampler else None
        if trace_state is not None and self.num_speculative_steps == 0:
            trace_state.apply_trace(sampled.view(-1), input_batch.idx_mapping)

        self.postprocess_sampled(
            idx_mapping=input_batch.idx_mapping,
            sampled_tokens=sampled,
            num_sampled=torch.from_numpy(num_sampled_np).to(self.device),
            num_rejected=torch.from_numpy(
                np.where(completes, num_drafts - num_accepted, 0).astype(np.int32)
            ).to(self.device),
            query_start_loc=input_batch.query_start_loc,
        )
        if self.num_speculative_steps > 0:
            # Placeholder drafts, as the real runner reports under async
            # scheduling; the scheduler only counts them.
            self.draft_tokens_handler.set_draft_tokens(
                input_batch,
                torch.full(
                    (num_reqs, pending.num_spec_tokens_to_schedule),
                    _PLACEHOLDER_DRAFT,
                    dtype=torch.int64,
                    device=self.device,
                ),
            )
        output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            req_id_to_index=row,
            sampled_token_ids=[
                tokens[:n]
                for tokens, n in zip(sampled.tolist(), num_sampled_np.tolist())
            ],
        )

        if self.step_log is not None:
            self.step_log.write(
                pending.descriptor,
                pending.cost,
                pending.t_start,
                pending.t_deadline,
                list(pending.finished_req_ids),
            )
        # Hold the engine thread for the rest of the step's host time.
        self.clock.sleep_until(pending.t_entry + pending.cost.host_s)
        self.kv_connector.finish_forward()
        output.kv_connector_output = self._post_forward(pending.finished_req_ids)
        self._last_output = SimulatedAsyncOutput(output, pending.t_deadline, self.clock)
        return self._last_output

    def _post_forward(self, finished_req_ids: set[str]) -> KVConnectorOutput | None:
        """Run the KV connector's end-of-step hooks: submit this step's loads
        and stores, and collect the transfers that have completed."""
        kv_output = self.kv_connector.post_forward(finished_req_ids)
        transfers = completed_transfers(kv_output)
        if self.step_log is not None and transfers:
            self.step_log.write_transfers(transfers, self.clock.now())
        return kv_output

    def _skip_to_transfer(self) -> None:
        """With nothing to run, skip to the next KV transfer completion, but
        not past the driver's next request arrival."""
        if self._kv_transfers is None or isinstance(self.clock, RealClock):
            return
        if self._last_output is not None and not self._last_output.resolved:
            return
        t = self._kv_transfers.next_completion()
        if t is not None:
            self.clock.sleep_until(min(t, self.clock.horizon))

    def _run_pipeline(self, t_start: float, device_s: float) -> float:
        """Push a step entering stage 0 at `t_start` through balanced pipeline
        stages; returns when it leaves the last."""
        stage_s = device_s / len(self._stage_free_at)
        t = t_start
        for stage, free_at in enumerate(self._stage_free_at):
            t = max(t, free_at) + stage_s
            self._stage_free_at[stage] = t
        return t

    def _syncs_dp(self) -> bool:
        """Whether this step joins the other MoE-DP ranks' lockstep step."""
        return not (
            self.dp_size == 1 or self._warming_up or should_skip_dp_coordination()
        )

    def _dp_joint_step(
        self, descriptor: StepDescriptor, now: float
    ) -> tuple[StepDescriptor, float]:
        """Under MoE data parallelism, the joint step of all DP ranks and the
        time at which they all reached it (the gather is a barrier); this
        rank's own step otherwise."""
        if not self._syncs_dp():
            return descriptor, now
        joint = gather_dp_step(descriptor)
        return joint, self.clock.now()

    def _agree_start(self, t_start: float, now: float) -> float:
        """The latest DP rank's start time. Clocks need not agree across
        hosts, so the ranks exchange delays relative to the moment they all
        reached the step."""
        if not self._syncs_dp():
            return t_start
        delay = torch.tensor([t_start - now], dtype=torch.float64)
        dist.all_reduce(delay, op=dist.ReduceOp.MAX, group=get_dp_group().cpu_group)
        return now + float(delay[0])

    def _estimate(self, descriptor: StepDescriptor) -> StepCost:
        if not descriptor.requests:
            return StepCost(device_s=0.0)
        return self.timing_provider.estimate(descriptor)

    def _dummy_run(self, num_tokens: int, *args, **kwargs):  # type: ignore[override]
        """An idle DP rank's dummy batch: join the lockstep step and occupy
        the pipeline like the busy ranks. The host does not wait for it, as
        the real dummy forward returns once launched."""
        descriptor, now = self._dp_joint_step(
            empty_step_descriptor(self._step_index, self._parallel),
            self.clock.now(),
        )
        cost = self._estimate(descriptor)
        t_start = self._agree_start(max(now, self._stage_free_at[0]), now)
        self._run_pipeline(t_start, cost.device_s)
        return None, None

    def _num_accepted(
        self, pending: _PendingStep, input_batch: InputBatch, num_drafts: np.ndarray
    ) -> np.ndarray:
        """Accepted drafts per batch row: the timing provider's, if it gives
        them, else sampled from the per-position acceptance rates."""
        if pending.cost.num_accepted is not None:
            by_req = dict(
                zip(
                    (r.req_id for r in pending.descriptor.requests),
                    pending.cost.num_accepted,
                )
            )
            accepted = np.array([by_req[r] for r in input_batch.req_ids])
            return np.minimum(accepted, num_drafts)
        assert self._acceptance_rates is not None
        width = int(num_drafts.max())
        rates = self._acceptance_rates[
            np.minimum(np.arange(width), len(self._acceptance_rates) - 1)
        ]
        accepted_at = self._rng.random((len(num_drafts), width)) < rates
        accepted_at &= np.arange(width) < num_drafts[:, None]
        # Drafts are accepted up to the first rejection.
        return np.cumprod(accepted_at, axis=1).sum(axis=1)
