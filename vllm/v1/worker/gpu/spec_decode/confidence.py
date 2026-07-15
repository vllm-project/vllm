# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Confidence-based verification for DSpark speculative decoding."""

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu.async_utils import stream
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import prepare_prefill_inputs
from vllm.v1.worker.gpu.spec_decode.dspark.online_sts import DSparkOnlineSTS

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig
    from vllm.v1.worker.gpu.attn_utils import AttentionCGSupportInfo
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.states import RequestState


@triton.jit
def _rewrite_compact_batch_kernel(
    query_start_loc_ptr,
    cu_num_logits_ptr,
    input_ids_ptr,
    positions_ptr,
    seq_lens_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    logits_indices_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    last_sampled_tokens_ptr,
    prefill_len_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    max_num_reqs,
    num_tokens,
    POS_BLOCK: tl.constexpr,
    SPEC_BLOCK: tl.constexpr,
    NUM_NEW_SAMPLED_TOKENS: tl.constexpr = 1,
):
    """Rebuild the compact verifier batch, one program per request."""
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_idx == num_reqs:
        for i in tl.range(num_reqs, max_num_reqs + 1, POS_BLOCK):
            block = i + tl.arange(0, POS_BLOCK)
            tl.store(seq_lens_ptr + block, 0, mask=block < max_num_reqs)
            # query_start_loc[num_reqs:] = num_tokens (end of the last
            # request plus full-CG padding rows).
            tl.store(
                query_start_loc_ptr + block,
                num_tokens,
                mask=block < max_num_reqs + 1,
            )
        tl.store(
            cu_num_logits_ptr + num_reqs,
            tl.load(cu_num_logits_ptr + num_reqs),
        )
        return

    start = tl.load(query_start_loc_ptr + req_idx)
    end = tl.load(query_start_loc_ptr + req_idx + 1)
    cu_start = tl.load(cu_num_logits_ptr + req_idx)
    cu_end = tl.load(cu_num_logits_ptr + req_idx + 1)

    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)
    query_len = end - start
    seq_len = num_computed_tokens + query_len
    tl.store(seq_lens_ptr + req_idx, seq_len)
    for i in tl.range(0, query_len, POS_BLOCK):
        block = i + tl.arange(0, POS_BLOCK)
        tl.store(
            positions_ptr + start + block,
            num_computed_tokens + block,
            mask=block < query_len,
        )

    num_logits = cu_end - cu_start
    logits_start = end - num_logits
    block = tl.arange(0, SPEC_BLOCK)
    mask = block < num_logits
    tl.store(logits_indices_ptr + cu_start + block, logits_start + block, mask=mask)
    tl.store(expanded_idx_mapping_ptr + cu_start + block, req_state_idx, mask=mask)
    tl.store(expanded_local_pos_ptr + cu_start + block, block, mask=mask)

    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    if seq_len <= prefill_len:
        # Pure prefill: prompt tokens come from prepare_prefill_inputs.
        return
    if NUM_NEW_SAMPLED_TOKENS > 0 and seq_len - num_logits >= prefill_len:
        last_token_id = tl.load(last_sampled_tokens_ptr + req_state_idx)
        tl.store(input_ids_ptr + logits_start, last_token_id)
    num_draft_tokens = num_logits - NUM_NEW_SAMPLED_TOKENS
    if num_draft_tokens > 0:
        draft_mask = block < num_draft_tokens
        draft_tokens = tl.load(
            draft_tokens_ptr + req_state_idx * draft_tokens_stride + block,
            mask=draft_mask,
        )
        tl.store(
            input_ids_ptr + end - num_draft_tokens + block,
            draft_tokens,
            mask=draft_mask,
        )


@triton.jit
def _compute_prefix_survival_kernel(
    confidence_logits_ptr,
    confidence_logits_stride,
    idx_mapping_ptr,
    valid_draft_tokens_ptr,
    temperatures_ptr,
    survival_ptr,
    NUM_STEPS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    num_valid = tl.load(valid_draft_tokens_ptr + req_idx)
    survival = 1.0
    for step in tl.static_range(NUM_STEPS):
        logit = tl.load(
            confidence_logits_ptr + req_state_idx * confidence_logits_stride + step
        )
        if temperatures_ptr is not None:
            logit /= tl.load(temperatures_ptr + step)
        survival *= tl.sigmoid(logit)
        tl.store(
            survival_ptr + req_idx * NUM_STEPS + step,
            survival,
            mask=step < num_valid,
        )
        tl.store(
            survival_ptr + req_idx * NUM_STEPS + step,
            -float("inf"),
            mask=step >= num_valid,
        )


@triton.jit
def _scatter_confidence_logits_kernel(
    source_ptr,
    source_stride,
    destination_ptr,
    destination_stride,
    idx_mapping_ptr,
    NUM_STEPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    steps = tl.arange(0, BLOCK_SIZE)
    mask = steps < NUM_STEPS
    logits = tl.load(source_ptr + req_idx * source_stride + steps, mask=mask)
    tl.store(
        destination_ptr + req_state_idx * destination_stride + steps,
        logits,
        mask=mask,
    )


@triton.jit
def _build_forward_skip_mask_kernel(
    is_padding_ptr,
    query_start_loc_ptr,
    draft_capacity_ptr,
    num_bonus_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start = tl.load(query_start_loc_ptr + req_idx)
    end = tl.load(query_start_loc_ptr + req_idx + 1)
    kept = tl.load(draft_capacity_ptr + req_idx)
    scheduled = end - start - num_bonus_tokens
    offsets = tl.arange(0, BLOCK_SIZE)
    tl.store(
        is_padding_ptr + start + num_bonus_tokens + kept + offsets,
        True,
        mask=offsets < scheduled - kept,
    )


def _count_valid_draft_tokens(
    draft_token_lists: Sequence[Sequence[int]],
    num_reqs: int,
) -> np.ndarray:
    """Grammar-validated draft ids round-trip through the scheduler; negative
    ids mark invalidated drafts."""
    return np.fromiter(
        (sum(token_id >= 0 for token_id in tokens) for tokens in draft_token_lists),
        dtype=np.int32,
        count=num_reqs,
    )


def compute_prefix_survival(
    confidence_logits: np.ndarray,
    temperatures: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Prefix-survival scores a_{r,j} = prod_{i<=j} sigmoid(logit_i / T_i)."""
    scaled = confidence_logits.astype(np.float64) / temperatures
    probs = np.exp(-np.logaddexp(0.0, -scaled))
    return np.cumprod(probs, axis=1)


def select_draft_token_budget(
    survival: np.ndarray,
    num_batch_requests: int,
    num_sampling_requests: int,
    num_required_target_tokens: int,
    budget_frac: float = 1.0,
    cost_tables: tuple[np.ndarray, np.ndarray] | None = None,
) -> int:
    """Choose the scalar draft budget from stale prefix scores.

    Current GPU confidences assign this budget to requests. The SPS objective
    includes all target tokens already required by the scheduler, while only
    requests that sample this step contribute mandatory useful output.
    """
    scores = survival[np.isfinite(survival) & (survival > 0.0)]
    scores = np.sort(scores)[::-1]
    scores = scores[: math.ceil(scores.size * budget_frac)]
    if scores.size == 0:
        return 0

    if cost_tables is None:
        return scores.size

    draft_cost_ms, verify_and_sample_cost_ms = cost_tables
    max_candidates = verify_and_sample_cost_ms.size - num_required_target_tokens - 1
    if max_candidates <= 0:
        return 0
    scores = scores[:max_candidates]
    if scores.size == 0:
        return 0
    verify_ms = verify_and_sample_cost_ms[
        num_required_target_tokens + 1 : num_required_target_tokens + 1 + scores.size
    ]
    draft_ms = draft_cost_ms[num_batch_requests]
    throughput = (num_sampling_requests + np.cumsum(scores)) / (draft_ms + verify_ms)
    best = int(np.argmax(throughput))
    baseline = num_sampling_requests / (
        draft_ms + verify_and_sample_cost_ms[num_required_target_tokens]
    )
    return 0 if baseline >= throughput[best] else best + 1


def build_cost_tables_from_curves(
    draft_curve: list[tuple[int, float]],
    verify_curve: list[tuple[int, float]],
    max_num_reqs: int,
    max_batch_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the measured draft and verifier costs."""

    def interpolate(values: np.ndarray, curve: list[tuple[int, float]]) -> np.ndarray:
        xs = np.array([x for x, _ in curve], dtype=np.float64)
        ys = np.array([y for _, y in curve], dtype=np.float64)
        result = np.interp(values, xs, ys)
        if len(xs) > 1:
            after = values > xs[-1]
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            result[after] = ys[-1] + slope * (values[after] - xs[-1])
        return result

    all_reqs = np.arange(max_num_reqs + 1)
    all_tokens = np.arange(max_batch_tokens + 1)
    draft_table = np.maximum(interpolate(all_reqs, draft_curve), 0.0)
    verify_table = np.maximum(interpolate(all_tokens, verify_curve), 1e-6)
    return draft_table, verify_table


class ConfidenceManager:
    varlen_spec_decode = False

    def __init__(
        self,
        req_states: "RequestState",
        speculative_config: "SpeculativeConfig",
    ):
        self.req_states = req_states
        self.num_speculative_steps = req_states.num_speculative_steps
        device = req_states.device
        self._copy_stream = torch.cuda.Stream(device)

        self.budget_frac = speculative_config.dspark_budget_frac
        self.should_profile = (
            self.varlen_spec_decode and speculative_config.dspark_sps_curve == "auto"
        )
        self.cost_tables: tuple[np.ndarray, np.ndarray] | None = None
        self.online_sts = (
            DSparkOnlineSTS(self.num_speculative_steps)
            if speculative_config.dspark_online_sts
            else None
        )

        max_num_reqs = req_states.max_num_reqs
        num_candidates = max_num_reqs * self.num_speculative_steps
        self._confidence_logits = torch.empty(
            (max_num_reqs, self.num_speculative_steps),
            dtype=torch.float32,
            device=device,
        )
        self._survival = torch.empty(
            (max_num_reqs, self.num_speculative_steps),
            dtype=torch.float32,
            device=device,
        )
        self._sort_buffers = (
            torch.empty(num_candidates, dtype=torch.float32, device=device),
            torch.empty(num_candidates, dtype=torch.int64, device=device),
        )
        self._capacity_ones = torch.ones(
            num_candidates, dtype=torch.int32, device=device
        )
        self._batch_draft_capacity = torch.empty(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self._valid_draft_tokens_cpu = torch.empty(
            max_num_reqs, dtype=torch.int32, pin_memory=True
        )
        self._valid_draft_tokens = torch.empty(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self._temperature_buffers = (
            (
                torch.empty(
                    self.num_speculative_steps,
                    dtype=torch.float32,
                    pin_memory=True,
                ),
                torch.empty(
                    self.num_speculative_steps,
                    dtype=torch.float32,
                    device=device,
                ),
            )
            if self.online_sts is not None
            else None
        )

        # Two D2H slots preserve stale inputs for budget selection and STS.
        shape = (2, req_states.max_num_reqs)
        self._staged_logits = torch.empty(
            (*shape, self.num_speculative_steps),
            dtype=torch.float32,
            pin_memory=True,
        )
        self._staged_outcomes = torch.empty(
            (2, 2, max_num_reqs), dtype=torch.int32, pin_memory=True
        )
        self._outcomes = torch.empty(
            (2, max_num_reqs), dtype=torch.int32, device=device
        )
        self._staged_req_ids: list[list[str]] = [[], []]
        self._copy_events = [torch.cuda.Event(blocking=True) for _ in range(2)]
        self._next_idx = 0
        self._stale_confidences: tuple[list[str], np.ndarray] | None = None
        self._previous_confidences: tuple[list[str], np.ndarray] | None = None
        self._profile_capacity: int | None = None

    def profile_step_costs(
        self,
        model_runner: "GPUModelRunner",
        worker_execute_model: Callable[[Any], Any],
        worker_sample_tokens: Callable[[Any], Any],
        decode_output: Any,
    ) -> None:
        """Profile draft-by-request and verifier-by-token step costs."""
        from vllm import SamplingParams

        cached = decode_output.scheduled_cached_reqs
        req_ids = list(cached.req_ids)
        query_len = model_runner.decode_query_len
        max_reqs = min(
            model_runner.max_num_reqs,
            model_runner.max_num_tokens // query_len,
            len(req_ids),
        )
        req_counts = sorted({1 << i for i in range(max_reqs.bit_length())} | {max_reqs})

        if model_runner.sampler is not None:
            params = SamplingParams()
            for req_id in req_ids:
                model_runner.sampler.add_request(
                    model_runner.req_states.req_id_to_index[req_id],
                    cached.num_computed_tokens[0],
                    params,
                )
            model_runner.sampler.apply_staged_writes()

        def measure_step() -> float:
            for _ in range(3):
                worker_execute_model(decode_output)
                worker_sample_tokens(None)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                worker_execute_model(decode_output)
                worker_sample_tokens(None)
            end.record()
            with gpu_sync_allowed():
                end.synchronize()
            return start.elapsed_time(end) / 10

        def resize_batch(num_reqs: int) -> None:
            selected_req_ids = req_ids[:num_reqs]
            cached.req_ids = selected_req_ids
            cached.num_computed_tokens = [cached.num_computed_tokens[0]] * num_reqs
            cached.num_output_tokens = [1] * num_reqs
            cached.new_block_ids = [None] * num_reqs
            decode_output.num_scheduled_tokens = dict.fromkeys(
                selected_req_ids, query_len
            )
            decode_output.scheduled_spec_decode_tokens = {
                req_id: [0] * self.num_speculative_steps for req_id in selected_req_ids
            }
            decode_output.total_num_scheduled_tokens = query_len * num_reqs

        speculator = model_runner.speculator
        assert speculator is not None
        propose = speculator.propose

        def skip_draft(input_batch: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
            return speculator.draft_tokens[: input_batch.num_reqs]

        try:
            draft_curve: list[tuple[int, float]] = []
            verify_costs: dict[int, float] = {}
            for num_reqs in req_counts:
                resize_batch(num_reqs)
                speculator.propose = skip_draft  # type: ignore[method-assign]
                full_verify_ms = 0.0
                for capacity in (0, self.num_speculative_steps):
                    self._profile_capacity = capacity
                    num_tokens = num_reqs * (1 + capacity)
                    verify_ms = measure_step()
                    verify_costs.setdefault(num_tokens, verify_ms)
                    full_verify_ms = verify_ms

                self._profile_capacity = self.num_speculative_steps
                speculator.propose = propose  # type: ignore[method-assign]
                draft_curve.append(
                    (num_reqs, max(measure_step() - full_verify_ms, 0.0))
                )
        finally:
            speculator.propose = propose  # type: ignore[method-assign]
            self._profile_capacity = None

        verify_curve = sorted(verify_costs.items())
        draft_curve, verify_curve = get_tp_group().broadcast_object(
            (draft_curve, verify_curve), src=0
        )
        self.cost_tables = build_cost_tables_from_curves(
            draft_curve,
            verify_curve,
            self.req_states.max_num_reqs,
            self.req_states.max_num_batched_tokens,
        )
        if self.online_sts is not None:
            self.online_sts.reset()
        logger.info_once(
            "DSpark auto-profiled draft cost (requests -> ms): %s",
            tuple((r, round(ms, 3)) for r, ms in draft_curve),
        )
        logger.info_once(
            "DSpark auto-profiled verifier + sampler cost (tokens -> ms): %s",
            tuple((b, round(ms, 3)) for b, ms in verify_curve),
        )

    def stage_confidences(
        self,
        confidence_logits: torch.Tensor,
        num_sampled: torch.Tensor,
        input_batch: "InputBatch",
    ) -> None:
        """Preserve current GPU scores and stage stale calibration inputs."""
        num_reqs = input_batch.num_reqs
        num_bonus = self._get_num_bonus_tokens(input_batch)
        _scatter_confidence_logits_kernel[(num_reqs,)](
            confidence_logits,
            confidence_logits.stride(0),
            self._confidence_logits,
            self._confidence_logits.stride(0),
            input_batch.idx_mapping,
            NUM_STEPS=self.num_speculative_steps,
            BLOCK_SIZE=triton.next_power_of_2(self.num_speculative_steps),
        )

        idx = self._next_idx
        self._next_idx ^= 1
        if self._staged_req_ids[idx]:
            with gpu_sync_allowed():
                self._copy_events[idx].synchronize()
            self._update_staged_confidences(idx)

        self._staged_req_ids[idx] = list(input_batch.req_ids)
        accepted_gpu, verified_gpu = self._outcomes[:, :num_reqs]
        accepted_gpu.copy_(num_sampled[:num_reqs]).sub_(num_bonus)
        verified_gpu.copy_(self._batch_draft_capacity[:num_reqs])
        current_stream = torch.cuda.current_stream(self.req_states.device)
        self._copy_stream.wait_stream(current_stream)
        with stream(self._copy_stream, current_stream):
            self._staged_logits[idx, :num_reqs].copy_(
                confidence_logits[:num_reqs], non_blocking=True
            )
            self._staged_outcomes[idx, :, :num_reqs].copy_(
                self._outcomes[:, :num_reqs], non_blocking=True
            )
            self._copy_events[idx].record()

    def wait_for_staged_copy(self) -> None:
        """Protect the persistent confidence buffer before its next replay."""
        if self._staged_req_ids[self._next_idx ^ 1]:
            torch.cuda.current_stream(self.req_states.device).wait_event(
                self._copy_events[self._next_idx ^ 1]
            )

    def get_num_tokens(
        self,
        num_tokens_per_req: dict[str, int],
        draft_tokens: dict[str, list[int]],
        has_structured_output_requests: bool = False,
    ) -> int:
        raise NotImplementedError

    def _update_staged_confidences(self, idx: int) -> None:
        """Consume a two-step-old snapshot for calibration and budgeting."""
        req_ids = self._staged_req_ids[idx]
        num_reqs = len(req_ids)
        confidence_logits_np = self._staged_logits[idx, :num_reqs].numpy()
        accepted_np, verified_np = self._staged_outcomes[idx, :, :num_reqs].numpy()
        if self.online_sts is not None and self._previous_confidences is not None:
            previous_req_ids, previous_logits = self._previous_confidences
            prev_row = {req_id: i for i, req_id in enumerate(previous_req_ids)}
            rows = np.array(
                [prev_row.get(req_id, -1) for req_id in req_ids],
                dtype=np.int64,
            )
            sel = rows >= 0
            if sel.any():
                self.online_sts.update(
                    previous_logits[rows[sel]],
                    accepted_np[sel],
                    verified_np[sel],
                )
        if self.online_sts is not None:
            self._previous_confidences = (req_ids, confidence_logits_np.copy())

        temperatures: np.ndarray | float = 1.0
        if self.online_sts is not None:
            temperatures = self.online_sts.temperatures
        self._stale_confidences = (
            req_ids,
            compute_prefix_survival(confidence_logits_np, temperatures),
        )

    def _plan_draft_token_budget(
        self,
        req_ids: list[str],
        num_required_target_tokens_per_req: np.ndarray,
        valid_draft_tokens_per_req: np.ndarray,
    ) -> int:
        num_batch_requests = len(req_ids)
        if self._profile_capacity is not None:
            return int(
                np.minimum(valid_draft_tokens_per_req, self._profile_capacity).sum()
            )

        survival = np.ones(
            (num_batch_requests, self.num_speculative_steps), dtype=np.float64
        )
        if self._stale_confidences is not None:
            stale_req_ids, stale_survival = self._stale_confidences
            stale_rows = {req_id: i for i, req_id in enumerate(stale_req_ids)}
            for row, req_id in enumerate(req_ids):
                stale_row = stale_rows.get(req_id)
                if stale_row is not None:
                    survival[row] = stale_survival[stale_row]
        steps = np.arange(self.num_speculative_steps)
        survival[steps[None, :] >= valid_draft_tokens_per_req[:, None]] = -np.inf

        slots = np.fromiter(
            (self.req_states.req_id_to_index[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_batch_requests,
        )
        num_computed_tokens = self.req_states.num_computed_tokens_np[slots]
        prefill_lens = self.req_states.prefill_len.np[slots]
        num_sampling_requests = int(
            np.count_nonzero(
                num_computed_tokens + num_required_target_tokens_per_req >= prefill_lens
            )
        )
        budget = select_draft_token_budget(
            survival,
            num_batch_requests=num_batch_requests,
            num_sampling_requests=num_sampling_requests,
            num_required_target_tokens=int(num_required_target_tokens_per_req.sum()),
            budget_frac=self.budget_frac,
            cost_tables=self.cost_tables,
        )
        budget = min(budget, int(valid_draft_tokens_per_req.sum()))
        return budget

    def _assign_draft_token_budget(
        self,
        input_batch: "InputBatch",
        valid_draft_tokens_per_req: np.ndarray,
        draft_token_budget: int,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        valid_np = self._valid_draft_tokens_cpu[:num_reqs].numpy()
        valid_np[:] = valid_draft_tokens_per_req
        async_copy_to_gpu(valid_np, out=self._valid_draft_tokens[:num_reqs])
        capacities = self._batch_draft_capacity[:num_reqs]
        if self._profile_capacity is not None or draft_token_budget == int(
            valid_draft_tokens_per_req.sum()
        ):
            capacities.copy_(self._valid_draft_tokens[:num_reqs])
            if self._profile_capacity is not None:
                capacities.clamp_(max=self._profile_capacity)
            return capacities

        temperatures = None
        if self.online_sts is not None:
            assert self._temperature_buffers is not None
            temperatures_cpu, temperatures = self._temperature_buffers
            temperatures_cpu.numpy()[:] = self.online_sts.temperatures
            temperatures.copy_(temperatures_cpu, non_blocking=True)
        _compute_prefix_survival_kernel[(num_reqs,)](
            self._confidence_logits,
            self._confidence_logits.stride(0),
            input_batch.idx_mapping,
            self._valid_draft_tokens,
            temperatures,
            self._survival,
            NUM_STEPS=self.num_speculative_steps,
        )
        num_candidates = num_reqs * self.num_speculative_steps
        sorted_survival, sorted_indices = self._sort_buffers
        torch.sort(
            self._survival[:num_reqs].flatten(),
            descending=True,
            stable=True,
            out=(
                sorted_survival[:num_candidates],
                sorted_indices[:num_candidates],
            ),
        )
        capacities.zero_()
        if draft_token_budget > 0:
            admitted = sorted_indices[:draft_token_budget]
            torch.div(
                admitted,
                self.num_speculative_steps,
                rounding_mode="floor",
                out=admitted,
            )
            capacities.scatter_add_(
                0,
                admitted,
                self._capacity_ones[:draft_token_budget],
            )
        return capacities

    def warmup(self, input_buffers: "InputBuffers") -> None:
        pass

    @staticmethod
    def _get_num_bonus_tokens(input_batch: "InputBatch") -> int:
        num_bonus_tokens, remainder = divmod(
            int(input_batch.cu_num_logits_np[-1]) - input_batch.num_draft_tokens,
            input_batch.num_reqs,
        )
        assert remainder == 0
        return num_bonus_tokens

    def trim_batch(
        self,
        input_batch: "InputBatch",
        draft_tokens: dict[str, list[int]],
    ) -> None:
        raise NotImplementedError


class VarlenConfidenceManager(ConfidenceManager):
    varlen_spec_decode = True

    def __init__(
        self,
        req_states: "RequestState",
        speculative_config: "SpeculativeConfig",
    ):
        super().__init__(req_states, speculative_config)
        device = req_states.device
        max_num_reqs = req_states.max_num_reqs
        max_num_logits = max_num_reqs * (1 + self.num_speculative_steps)
        self._required_target_tokens_cpu = torch.empty(
            max_num_reqs, dtype=torch.int32, pin_memory=True
        )
        self._required_target_tokens_np = self._required_target_tokens_cpu.numpy()
        self._required_target_tokens = torch.empty(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self._scheduled_tokens = torch.empty(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self._num_logits = torch.empty(max_num_reqs, dtype=torch.int32, device=device)
        self._cu_num_logits = torch.empty(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self._expanded_idx_mapping = torch.empty(
            max_num_logits, dtype=torch.int32, device=device
        )
        self._expanded_local_pos = torch.empty(
            max_num_logits, dtype=torch.int32, device=device
        )
        self._logits_indices = torch.empty(
            max_num_logits, dtype=torch.int64, device=device
        )
        self._planned_draft_token_budget: int | None = None

    def get_num_tokens(
        self,
        num_tokens_per_req: dict[str, int],
        draft_tokens: dict[str, list[int]],
        has_structured_output_requests: bool = False,
    ) -> int:
        num_reqs = len(num_tokens_per_req)
        decode_query_len = 1 + self.num_speculative_steps
        req_ids = sorted(
            num_tokens_per_req,
            key=lambda req_id: (
                num_tokens_per_req[req_id] != decode_query_len,
                num_tokens_per_req[req_id],
            ),
        )
        num_scheduled_tokens = np.fromiter(
            (num_tokens_per_req[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        draft_token_lists = [draft_tokens.get(req_id, ()) for req_id in req_ids]
        num_draft_tokens_per_req = np.fromiter(
            (len(tokens) for tokens in draft_token_lists),
            dtype=np.int32,
            count=num_reqs,
        )
        if has_structured_output_requests:
            valid_num_draft_tokens_per_req = _count_valid_draft_tokens(
                draft_token_lists, num_reqs
            )
        else:
            valid_num_draft_tokens_per_req = num_draft_tokens_per_req
        required_target_tokens_per_req = num_scheduled_tokens - num_draft_tokens_per_req
        draft_token_budget = self._plan_draft_token_budget(
            req_ids,
            required_target_tokens_per_req,
            valid_num_draft_tokens_per_req,
        )
        self._planned_draft_token_budget = draft_token_budget
        return int(required_target_tokens_per_req.sum()) + draft_token_budget

    def warmup(self, input_buffers: "InputBuffers") -> None:
        query_len = 1 + self.num_speculative_steps
        input_buffers.query_start_loc[:2] = torch.tensor(
            [0, query_len], dtype=torch.int32, device=self.req_states.device
        )
        self._cu_num_logits[:2] = torch.tensor(
            [0, query_len], dtype=torch.int32, device=self.req_states.device
        )
        idx_mapping = torch.zeros(1, dtype=torch.int32, device=self.req_states.device)
        _rewrite_compact_batch_kernel[(2,)](
            input_buffers.query_start_loc,
            self._cu_num_logits,
            input_buffers.input_ids,
            input_buffers.positions,
            input_buffers.seq_lens,
            self._expanded_idx_mapping,
            self._expanded_local_pos,
            self._logits_indices,
            idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.last_sampled_tokens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            self.req_states.draft_tokens.stride(0),
            self.req_states.max_num_reqs,
            query_len,
            POS_BLOCK=1024,
            SPEC_BLOCK=triton.next_power_of_2(self.num_speculative_steps + 1),
        )

    def _rewrite_compact_batch(
        self,
        input_batch: "InputBatch",
        required_target_tokens_per_req: np.ndarray,
        valid_draft_tokens_per_req: np.ndarray,
        metadata_draft_tokens_per_req: np.ndarray,
        draft_token_budget: int,
        num_bonus_tokens: int,
    ) -> None:
        num_reqs = input_batch.num_reqs
        num_tokens = int(required_target_tokens_per_req.sum()) + draft_token_budget
        metadata_scheduled_tokens = (
            required_target_tokens_per_req + metadata_draft_tokens_per_req
        )
        input_batch.num_scheduled_tokens = metadata_scheduled_tokens
        input_batch.num_tokens = num_tokens
        input_batch.num_draft_tokens_per_req = metadata_draft_tokens_per_req
        input_batch.num_draft_tokens = draft_token_budget

        capacities = self._assign_draft_token_budget(
            input_batch,
            valid_draft_tokens_per_req,
            draft_token_budget,
        )
        required_np = self._required_target_tokens_np[:num_reqs]
        required_np[:] = required_target_tokens_per_req
        async_copy_to_gpu(
            required_np,
            out=self._required_target_tokens[:num_reqs],
        )
        torch.add(
            self._required_target_tokens[:num_reqs],
            capacities,
            out=self._scheduled_tokens[:num_reqs],
        )
        input_batch.query_start_loc[:1].zero_()
        torch.cumsum(
            self._scheduled_tokens[:num_reqs],
            dim=0,
            out=input_batch.query_start_loc[1 : num_reqs + 1],
        )
        torch.add(
            capacities,
            num_bonus_tokens,
            out=self._num_logits[:num_reqs],
        )
        self._cu_num_logits[:1].zero_()
        torch.cumsum(
            self._num_logits[:num_reqs],
            dim=0,
            out=self._cu_num_logits[1 : num_reqs + 1],
        )

        query_start_loc_np = np.empty(num_reqs + 1, dtype=np.int32)
        cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(metadata_scheduled_tokens, out=query_start_loc_np[1:])
        cu_num_logits_np[0] = 0
        np.cumsum(
            metadata_draft_tokens_per_req + num_bonus_tokens,
            out=cu_num_logits_np[1:],
        )
        num_logits = num_reqs * num_bonus_tokens + draft_token_budget

        input_batch.cu_num_logits_np = cu_num_logits_np
        input_batch.cu_num_logits = self._cu_num_logits[: num_reqs + 1]
        input_batch.expanded_idx_mapping = self._expanded_idx_mapping[:num_logits]
        input_batch.expanded_local_pos = self._expanded_local_pos[:num_logits]
        input_batch.logits_indices = self._logits_indices[:num_logits]

        padded_query_start_loc_np = np.empty(
            input_batch.num_reqs_after_padding + 1, dtype=np.int32
        )
        padded_query_start_loc_np[: num_reqs + 1] = query_start_loc_np
        padded_query_start_loc_np[num_reqs + 1 :] = num_tokens
        input_batch.query_start_loc_np = padded_query_start_loc_np

        _rewrite_compact_batch_kernel[(num_reqs + 1,)](
            input_batch.query_start_loc,
            self._cu_num_logits,
            input_batch.input_ids,
            input_batch.positions,
            input_batch.seq_lens,
            self._expanded_idx_mapping,
            self._expanded_local_pos,
            self._logits_indices,
            input_batch.idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.last_sampled_tokens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            self.req_states.draft_tokens.stride(0),
            self.req_states.max_num_reqs,
            num_tokens,
            POS_BLOCK=1024,
            SPEC_BLOCK=triton.next_power_of_2(self.num_speculative_steps + 1),
        )
        if np.any(input_batch.is_prefilling_np):
            prepare_prefill_inputs(
                input_batch.input_ids,
                self.req_states.next_prefill_tokens,
                input_batch.idx_mapping,
                input_batch.query_start_loc,
                self.req_states.all_token_ids.gpu,
                self.req_states.prefill_len.gpu,
                self.req_states.num_computed_tokens.gpu,
            )

    def trim_batch(
        self,
        input_batch: "InputBatch",
        draft_tokens: dict[str, list[int]],
    ) -> None:
        if (
            input_batch.num_draft_tokens == 0
            or input_batch.num_draft_tokens_per_req is None
        ):
            self._batch_draft_capacity[: input_batch.num_reqs].zero_()
            return

        num_bonus_tokens = self._get_num_bonus_tokens(input_batch)
        draft_token_budget = self._planned_draft_token_budget
        self._planned_draft_token_budget = None
        scheduled_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
        valid_draft_tokens_per_req = scheduled_draft_tokens_per_req
        if input_batch.has_structured_output_reqs:
            valid_draft_tokens_per_req = _count_valid_draft_tokens(
                [draft_tokens.get(req_id, ()) for req_id in input_batch.req_ids],
                input_batch.num_reqs,
            )
        required_target_tokens_per_req = (
            input_batch.num_scheduled_tokens - scheduled_draft_tokens_per_req
        )
        if draft_token_budget is None:
            draft_token_budget = self._plan_draft_token_budget(
                input_batch.req_ids,
                required_target_tokens_per_req,
                valid_draft_tokens_per_req,
            )

        metadata_draft_tokens_per_req = np.zeros(input_batch.num_reqs, dtype=np.int32)
        remaining = draft_token_budget
        for step in range(self.num_speculative_steps):
            rows = np.flatnonzero(valid_draft_tokens_per_req > step)
            admitted = min(remaining, rows.size)
            metadata_draft_tokens_per_req[rows[:admitted]] += 1
            remaining -= admitted
            if remaining == 0:
                break
        self._rewrite_compact_batch(
            input_batch,
            required_target_tokens_per_req,
            valid_draft_tokens_per_req,
            metadata_draft_tokens_per_req,
            draft_token_budget,
            num_bonus_tokens,
        )


class MaskedConfidenceManager(ConfidenceManager):
    def _prepare_forward_skip_mask(
        self,
        input_batch: "InputBatch",
        num_bonus_tokens: int,
        draft_tokens: dict[str, list[int]],
    ) -> None:
        assert input_batch.num_draft_tokens_per_req is not None
        scheduled_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
        valid_draft_tokens_per_req = scheduled_draft_tokens_per_req
        if input_batch.has_structured_output_reqs:
            valid_draft_tokens_per_req = _count_valid_draft_tokens(
                [draft_tokens.get(req_id, ()) for req_id in input_batch.req_ids],
                input_batch.num_reqs,
            )
        required_target_tokens_per_req = (
            input_batch.num_scheduled_tokens - scheduled_draft_tokens_per_req
        )
        draft_token_budget = self._plan_draft_token_budget(
            input_batch.req_ids,
            required_target_tokens_per_req,
            valid_draft_tokens_per_req,
        )
        capacities = self._assign_draft_token_budget(
            input_batch,
            valid_draft_tokens_per_req,
            draft_token_budget,
        )
        _build_forward_skip_mask_kernel[(input_batch.num_reqs,)](
            input_batch.is_padding,
            input_batch.query_start_loc,
            capacities,
            num_bonus_tokens,
            BLOCK_SIZE=triton.next_power_of_2(self.num_speculative_steps),
        )

    def trim_batch(
        self,
        input_batch: "InputBatch",
        draft_tokens: dict[str, list[int]],
    ) -> None:
        if (
            input_batch.num_draft_tokens == 0
            or input_batch.num_draft_tokens_per_req is None
        ):
            self._batch_draft_capacity[: input_batch.num_reqs].zero_()
            return

        num_bonus_tokens = self._get_num_bonus_tokens(input_batch)
        self._prepare_forward_skip_mask(input_batch, num_bonus_tokens, draft_tokens)
        input_batch.input_ids[: input_batch.num_tokens].masked_fill_(
            input_batch.is_padding[: input_batch.num_tokens],
            0,
        )


def make_confidence_manager(
    mode: str,
    attn_cg_support: "AttentionCGSupportInfo",
    req_states: "RequestState",
    speculative_config: "SpeculativeConfig",
) -> ConfidenceManager:
    if mode == "auto":
        if attn_cg_support.min_cg_support == AttentionCGSupport.ALWAYS:
            mode = "varlen"
        else:
            logger.info_once(
                "Using masked confidence-based verification because %s "
                "reports CUDA graph support %s.",
                attn_cg_support.min_cg_attn_backend,
                attn_cg_support.min_cg_support.name,
            )
            mode = "mask"
    if mode == "varlen":
        return VarlenConfidenceManager(req_states, speculative_config)
    if mode == "mask":
        return MaskedConfidenceManager(req_states, speculative_config)
    raise ValueError(f"Unknown confidence-based verification mode: {mode}")
