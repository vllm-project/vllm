# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Confidence-based verification for DSpark speculative decoding."""

import math
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
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState


@triton.jit
def _build_compact_offsets_kernel(
    query_start_loc_ptr,
    old_cu_num_logits_ptr,
    draft_capacity_ptr,
    new_cu_num_logits_ptr,
    seq_lens_ptr,
    num_reqs,
    max_num_reqs,
    num_bonus_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    active = offsets < num_reqs
    query_start = tl.load(query_start_loc_ptr + offsets, mask=active, other=0)
    query_end = tl.load(query_start_loc_ptr + offsets + 1, mask=active, other=0)
    logits_start = tl.load(old_cu_num_logits_ptr + offsets, mask=active, other=0)
    logits_end = tl.load(old_cu_num_logits_ptr + offsets + 1, mask=active, other=0)
    required = tl.where(
        active,
        query_end - query_start - (logits_end - logits_start - num_bonus_tokens),
        0,
    )
    capacity = tl.load(draft_capacity_ptr + offsets, mask=active, other=0)
    query_end = tl.cumsum(required + capacity, axis=0)
    logits_end = tl.cumsum(capacity + num_bonus_tokens, axis=0)
    num_tokens = tl.sum(required + capacity, axis=0)

    tl.store(query_start_loc_ptr, 0)
    tl.store(new_cu_num_logits_ptr, 0)
    in_range = offsets < max_num_reqs
    tl.store(
        query_start_loc_ptr + offsets + 1,
        tl.where(active, query_end, num_tokens),
        mask=in_range,
    )
    tl.store(new_cu_num_logits_ptr + offsets + 1, logits_end, mask=active)

    tl.store(seq_lens_ptr + offsets, 0, mask=in_range & ~active)


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
    POS_BLOCK: tl.constexpr,
    SPEC_BLOCK: tl.constexpr,
    NUM_NEW_SAMPLED_TOKENS: tl.constexpr = 1,
):
    """Rebuild the compact verifier batch, one program per request."""
    req_idx = tl.program_id(0)
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
            tl.where(step < num_valid, survival, -float("inf")),
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
def _mask_excess_draft_tokens_kernel(
    is_padding_ptr,
    input_ids_ptr,
    query_start_loc_ptr,
    cu_num_logits_ptr,
    draft_capacity_ptr,
    num_bonus_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    end = tl.load(query_start_loc_ptr + req_idx + 1)
    num_logits = tl.load(cu_num_logits_ptr + req_idx + 1) - tl.load(
        cu_num_logits_ptr + req_idx
    )
    logits_start = end - num_logits
    kept = tl.load(draft_capacity_ptr + req_idx)
    offsets = tl.arange(0, BLOCK_SIZE)
    token_idx = logits_start + num_bonus_tokens + kept + offsets
    mask = token_idx < end
    tl.store(
        is_padding_ptr + token_idx,
        True,
        mask=mask,
    )
    tl.store(input_ids_ptr + token_idx, 0, mask=mask)


def compute_prefix_survival(
    confidence_logits: np.ndarray,
    temperatures: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Prefix-survival scores a_{r,j} = prod_{i<=j} sigmoid(logit_i / T_i)."""
    scaled = confidence_logits.astype(np.float64) / temperatures
    probs = np.exp(-np.logaddexp(0.0, -scaled))
    return np.cumprod(probs, axis=1)


def build_cost_tables_from_curves(
    draft_curve: list[tuple[int, float]],
    verify_curve: list[tuple[int, float]],
    max_num_reqs: int,
    max_batch_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build graph-padded costs, extrapolating beyond the captured region."""

    def build_table(limit: int, curve: list[tuple[int, float]]) -> np.ndarray:
        xs, ys = np.asarray(curve, dtype=np.float64).T
        ys = np.maximum.accumulate(ys)
        values = np.arange(limit + 1)
        result = ys[np.minimum(np.searchsorted(xs, values), len(xs) - 1)]
        if len(xs) > 1:
            after = values > xs[-1]
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            result[after] = ys[-1] + slope * (values[after] - xs[-1])
        return result

    draft_table = np.maximum(build_table(max_num_reqs, draft_curve), 0.0)
    verify_table = np.maximum(build_table(max_batch_tokens, verify_curve), 1e-6)
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
        self.time_graphs = (
            self.varlen_spec_decode and speculative_config.dspark_sps_curve == "auto"
        )
        self.cost_tables: tuple[np.ndarray, np.ndarray] | None = None
        self.online_sts = (
            DSparkOnlineSTS(self.num_speculative_steps, device)
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
        self._valid_draft_tokens_np = self._valid_draft_tokens_cpu.numpy()
        self._valid_draft_tokens = torch.empty(
            max_num_reqs, dtype=torch.int32, device=device
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
        self._staged_logits_np = self._staged_logits.numpy()
        self._staged_outcomes_np = self._staged_outcomes.numpy()
        self._outcomes = torch.empty(
            (2, max_num_reqs), dtype=torch.int32, device=device
        )
        self._staged_req_ids: list[list[str]] = [[], []]
        self._copy_events = [torch.cuda.Event(blocking=True) for _ in range(2)]
        self._next_idx = 0
        self._stale_confidences: tuple[list[str], np.ndarray] | None = None
        self._previous_confidences: tuple[list[str], np.ndarray] | None = None

    def set_graph_costs(self, model_graphs: Any, speculator: Any) -> None:
        draft_graphs = speculator.query_cudagraph_manager
        if draft_graphs is None:
            return

        def collapse(timings: dict[Any, float], field: str) -> list[tuple[int, float]]:
            points: dict[int, float] = {}
            for desc, ms in timings.items():
                if desc.num_active_loras == 0:
                    x = getattr(desc, field)
                    assert x is not None
                    points[x] = max(points.get(x, 0.0), ms)
            return sorted(points.items())

        draft_curve = collapse(draft_graphs.graph_timings, "num_reqs")
        verify_curve = collapse(model_graphs.graph_timings, "num_tokens")
        draft_curve, verify_curve = get_tp_group().broadcast_object(
            (draft_curve, verify_curve), src=0
        )
        if not draft_curve or not verify_curve:
            logger.warning_once("DSpark could not time FULL CUDA graphs.")
            return
        self.cost_tables = build_cost_tables_from_curves(
            draft_curve,
            verify_curve,
            self.req_states.max_num_reqs,
            self.req_states.max_num_batched_tokens,
        )
        if self.online_sts is not None:
            self.online_sts.reset()
        logger.info_once(
            "DSpark timed draft CUDA graphs (requests -> ms): %s",
            tuple((r, round(ms, 3)) for r, ms in draft_curve),
        )
        logger.info_once(
            "DSpark timed verifier CUDA graphs (tokens -> ms): %s",
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
        confidence_logits_np = self._staged_logits_np[idx, :num_reqs]
        accepted_np, verified_np = self._staged_outcomes_np[idx, :, :num_reqs]
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
        if self.cost_tables is None and self.budget_frac >= 1.0:
            return int(valid_draft_tokens_per_req.sum())

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

        scores = survival[survival > 0.0]
        scores = np.sort(scores)[::-1]
        scores = scores[: math.ceil(scores.size * self.budget_frac)]
        if scores.size == 0 or self.cost_tables is None:
            return scores.size

        num_required_target_tokens = int(num_required_target_tokens_per_req.sum())
        draft_cost_ms, verify_cost_ms = self.cost_tables
        max_candidates = verify_cost_ms.size - num_required_target_tokens - 1
        if max_candidates <= 0:
            return 0
        scores = scores[:max_candidates]
        if scores.size == 0:
            return 0

        slots = np.fromiter(
            (self.req_states.req_id_to_index[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_batch_requests,
        )
        num_sampling_requests = np.count_nonzero(
            self.req_states.num_computed_tokens_np[slots]
            + num_required_target_tokens_per_req
            >= self.req_states.prefill_len.np[slots]
        )
        verify_ms = verify_cost_ms[
            num_required_target_tokens + 1 : num_required_target_tokens
            + 1
            + scores.size
        ]
        draft_ms = draft_cost_ms[num_batch_requests]
        throughput = (num_sampling_requests + np.cumsum(scores)) / (
            draft_ms + verify_ms
        )
        best = int(np.argmax(throughput))
        baseline = num_sampling_requests / (
            draft_ms + verify_cost_ms[num_required_target_tokens]
        )
        return 0 if baseline >= throughput[best] else best + 1

    def _assign_draft_token_budget(
        self,
        input_batch: "InputBatch",
        valid_draft_tokens_per_req: np.ndarray,
        draft_token_budget: int,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        capacities = self._batch_draft_capacity[:num_reqs]
        if draft_token_budget == 0:
            return capacities.zero_()

        valid_np = self._valid_draft_tokens_np[:num_reqs]
        valid_np[:] = valid_draft_tokens_per_req
        if draft_token_budget == int(valid_draft_tokens_per_req.sum()):
            async_copy_to_gpu(valid_np, out=capacities)
            return capacities
        async_copy_to_gpu(valid_np, out=self._valid_draft_tokens[:num_reqs])

        temperatures = (
            self.online_sts.copy_temperatures_to_gpu()
            if self.online_sts is not None
            else None
        )
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

    def warmup(self) -> None:
        self._confidence_logits[:1].zero_()
        self._valid_draft_tokens[:1].fill_(self.num_speculative_steps)
        temperatures = (
            self.online_sts.copy_temperatures_to_gpu()
            if self.online_sts is not None
            else None
        )
        idx_mapping = torch.zeros(1, dtype=torch.int32, device=self.req_states.device)
        _compute_prefix_survival_kernel[(1,)](
            self._confidence_logits,
            self._confidence_logits.stride(0),
            idx_mapping,
            self._valid_draft_tokens,
            temperatures,
            self._survival,
            NUM_STEPS=self.num_speculative_steps,
        )

        self._staged_req_ids = [[], []]
        self._next_idx = 0
        self._stale_confidences = None
        self._previous_confidences = None
        if self.online_sts is not None:
            self.online_sts.reset()

    @staticmethod
    def _get_num_bonus_tokens(input_batch: "InputBatch") -> int:
        num_bonus_tokens, remainder = divmod(
            int(input_batch.cu_num_logits_np[-1]) - input_batch.num_draft_tokens,
            input_batch.num_reqs,
        )
        assert remainder == 0
        return num_bonus_tokens

    def _plan_batch(
        self,
        req_ids: list[str],
        scheduled_tokens: np.ndarray,
        scheduled_drafts: np.ndarray,
        draft_tokens: dict[str, list[int]],
        has_structured_output: bool,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        valid_drafts = scheduled_drafts
        if has_structured_output:
            valid_drafts = np.fromiter(
                (
                    sum(token >= 0 for token in draft_tokens.get(req_id, ()))
                    for req_id in req_ids
                ),
                dtype=np.int32,
                count=len(req_ids),
            )
        required_targets = scheduled_tokens - scheduled_drafts
        return (
            required_targets,
            valid_drafts,
            self._plan_draft_token_budget(req_ids, required_targets, valid_drafts),
        )

    def _plan_input_batch(
        self,
        input_batch: "InputBatch",
        draft_tokens: dict[str, list[int]],
    ) -> tuple[np.ndarray, np.ndarray, int]:
        assert input_batch.num_draft_tokens_per_req is not None
        return self._plan_batch(
            input_batch.req_ids,
            input_batch.num_scheduled_tokens,
            input_batch.num_draft_tokens_per_req,
            draft_tokens,
            input_batch.has_structured_output_reqs,
        )

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
        self._planned_batch: tuple[np.ndarray, np.ndarray, int] | None = None

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
        num_draft_tokens_per_req = np.fromiter(
            (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        self._planned_batch = self._plan_batch(
            req_ids,
            num_scheduled_tokens,
            num_draft_tokens_per_req,
            draft_tokens,
            has_structured_output_requests,
        )
        required_target_tokens_per_req, _, draft_token_budget = self._planned_batch
        return int(required_target_tokens_per_req.sum()) + draft_token_budget

    def _rewrite_compact_batch(
        self,
        input_batch: "InputBatch",
        required_target_tokens_per_req: np.ndarray,
        valid_draft_tokens_per_req: np.ndarray,
        draft_token_budget: int,
        num_bonus_tokens: int,
    ) -> None:
        num_reqs = input_batch.num_reqs
        num_tokens = int(required_target_tokens_per_req.sum()) + draft_token_budget

        admitted_rows = np.nonzero(
            np.arange(self.num_speculative_steps)[:, None] < valid_draft_tokens_per_req
        )[1][:draft_token_budget]
        metadata_drafts = np.bincount(admitted_rows, minlength=num_reqs).astype(
            np.int32
        )

        np.add(
            required_target_tokens_per_req,
            metadata_drafts,
            out=input_batch.num_scheduled_tokens,
        )
        input_batch.num_tokens = num_tokens
        input_batch.num_draft_tokens_per_req = metadata_drafts
        input_batch.num_draft_tokens = draft_token_budget

        capacities = self._assign_draft_token_budget(
            input_batch,
            valid_draft_tokens_per_req,
            draft_token_budget,
        )
        _build_compact_offsets_kernel[(1,)](
            input_batch.query_start_loc,
            input_batch.cu_num_logits,
            capacities,
            self._cu_num_logits,
            input_batch.seq_lens,
            num_reqs,
            self.req_states.max_num_reqs,
            num_bonus_tokens,
            BLOCK_SIZE=triton.next_power_of_2(self.req_states.max_num_reqs),
        )

        query_start_loc_np = input_batch.query_start_loc_np
        query_start_loc_np[0] = 0
        np.cumsum(
            input_batch.num_scheduled_tokens,
            out=query_start_loc_np[1 : num_reqs + 1],
        )
        query_start_loc_np[num_reqs + 1 :] = num_tokens

        cu_num_logits_np = input_batch.cu_num_logits_np
        cu_num_logits_np[0] = 0
        np.cumsum(
            metadata_drafts + num_bonus_tokens,
            out=cu_num_logits_np[1:],
        )
        num_logits = num_reqs * num_bonus_tokens + draft_token_budget

        input_batch.cu_num_logits = self._cu_num_logits[: num_reqs + 1]
        input_batch.expanded_idx_mapping = self._expanded_idx_mapping[:num_logits]
        input_batch.expanded_local_pos = self._expanded_local_pos[:num_logits]
        input_batch.logits_indices = self._logits_indices[:num_logits]

        _rewrite_compact_batch_kernel[(num_reqs,)](
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
            self._planned_batch = None
            self._batch_draft_capacity[: input_batch.num_reqs].zero_()
            return

        num_bonus_tokens = self._get_num_bonus_tokens(input_batch)
        planned_batch = self._planned_batch
        self._planned_batch = None
        if planned_batch is None:
            planned_batch = self._plan_input_batch(input_batch, draft_tokens)
        required_target_tokens_per_req, valid_draft_tokens_per_req, budget = (
            planned_batch
        )
        self._rewrite_compact_batch(
            input_batch,
            required_target_tokens_per_req,
            valid_draft_tokens_per_req,
            budget,
            num_bonus_tokens,
        )


class MaskedConfidenceManager(ConfidenceManager):
    def trim_batch(
        self,
        input_batch: "InputBatch",
        draft_tokens: dict[str, list[int]],
    ) -> None:
        scheduled_drafts = input_batch.num_draft_tokens_per_req
        if input_batch.num_draft_tokens == 0 or scheduled_drafts is None:
            self._batch_draft_capacity[: input_batch.num_reqs].zero_()
            return

        _, valid_drafts, draft_token_budget = self._plan_input_batch(
            input_batch, draft_tokens
        )
        capacities = self._assign_draft_token_budget(
            input_batch, valid_drafts, draft_token_budget
        )
        _mask_excess_draft_tokens_kernel[(input_batch.num_reqs,)](
            input_batch.is_padding,
            input_batch.input_ids,
            input_batch.query_start_loc,
            input_batch.cu_num_logits,
            capacities,
            self._get_num_bonus_tokens(input_batch),
            BLOCK_SIZE=triton.next_power_of_2(self.num_speculative_steps),
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
