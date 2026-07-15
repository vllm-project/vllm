# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Confidence-based verification for DSpark speculative decoding."""

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

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig
    from vllm.v1.worker.gpu.attn_utils import AttentionCGSupportInfo
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState


@triton.jit
def _keyed_product(left_value, left_key, right_value, right_key):
    value = tl.where(left_key == right_key, left_value * right_value, right_value)
    return value, right_key


@triton.jit(do_not_specialize=["num_reqs", "draft_token_budget"])
def _assign_draft_token_budget_kernel(
    confidence_logits_ptr,
    confidence_logits_stride,
    idx_mapping_ptr,
    capacities_ptr,
    num_reqs,
    draft_token_budget,
    NUM_STEPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    candidate_idx = tl.arange(0, BLOCK_SIZE)
    req_idx = candidate_idx // NUM_STEPS
    step = candidate_idx % NUM_STEPS
    valid_candidate = req_idx < num_reqs
    req_state_idx = tl.load(idx_mapping_ptr + req_idx, mask=valid_candidate, other=0)
    logit = tl.load(
        confidence_logits_ptr + req_state_idx * confidence_logits_stride + step,
        mask=valid_candidate,
        other=-float("inf"),
    )
    probability = tl.sigmoid(logit)
    survival, _ = tl.associative_scan(
        (probability, req_idx),
        axis=0,
        combine_fn=_keyed_product,
    )
    num_valid = tl.load(
        capacities_ptr + req_idx,
        mask=valid_candidate,
        other=0,
    )
    survival = tl.where(valid_candidate & (step < num_valid), survival, -float("inf"))

    min_int32: tl.constexpr = -2147483648
    score_bits = survival.to(tl.int32, bitcast=True)
    sort_key = tl.where(
        score_bits >> 31 == 0,
        score_bits ^ -1,
        score_bits ^ min_int32,
    )
    packed = ((sort_key.to(tl.int64) & 0xFFFFFFFF) << 32) | candidate_idx.to(tl.int64)
    admitted_idx = (tl.sort(packed) & 0xFFFFFFFF).to(tl.int32)

    tl.store(capacities_ptr + candidate_idx, 0, mask=candidate_idx < num_reqs)
    tl.debug_barrier()
    tl.atomic_add(
        capacities_ptr + admitted_idx // NUM_STEPS,
        1,
        mask=candidate_idx < draft_token_budget,
    )


def compute_prefix_survival(confidence_logits: np.ndarray) -> np.ndarray:
    """Prefix-survival scores a_{r,j} = prod_{i<=j} sigmoid(logit_i)."""
    scaled = confidence_logits.astype(np.float64)
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
    compact_batch = False

    def __init__(
        self,
        req_states: "RequestState",
        speculative_config: "SpeculativeConfig",
        num_bonus_tokens: int,
    ):
        self.req_states = req_states
        self.num_speculative_steps = req_states.num_speculative_steps
        device = req_states.device
        self._copy_stream = torch.cuda.Stream(device)

        self.time_graphs = (
            self.varlen_spec_decode and speculative_config.dspark_sps_curve == "auto"
        )
        self.num_bonus_tokens = num_bonus_tokens
        self.cost_tables: tuple[np.ndarray, np.ndarray] | None = None
        max_num_reqs = req_states.max_num_reqs
        self._confidence_logits = torch.empty(
            (max_num_reqs, self.num_speculative_steps),
            dtype=torch.float32,
            device=device,
        )
        self._batch_draft_capacity = torch.empty(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self._draft_steps = torch.arange(
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )

        # Two D2H slots preserve stale inputs for budget selection.
        shape = (2, req_states.max_num_reqs)
        self._staged_logits_gpu = torch.empty(
            (*shape, self.num_speculative_steps),
            dtype=torch.float32,
            device=device,
        )
        self._staged_logits = torch.empty(
            (*shape, self.num_speculative_steps),
            dtype=torch.float32,
            pin_memory=True,
        )
        self._staged_logits_np = self._staged_logits.numpy()
        self._staged_req_ids: list[list[str]] = [[], []]
        self._copy_events = [torch.cuda.Event(blocking=True) for _ in range(2)]
        self._next_idx = 0
        self._stale_confidences: tuple[list[str], np.ndarray] | None = None

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
        logger.debug("DSpark cost tables: %s", self.cost_tables)

    def stage_confidences(
        self,
        confidence_logits: torch.Tensor,
        input_batch: "InputBatch",
    ) -> None:
        """Preserve current GPU scores and stage stale budget inputs."""
        num_reqs = input_batch.num_reqs
        idx = self._next_idx
        self._next_idx ^= 1
        if self._staged_req_ids[idx]:
            with gpu_sync_allowed():
                self._copy_events[idx].synchronize()
            self._update_staged_confidences(idx)

        logits = confidence_logits[:num_reqs]
        self._confidence_logits[input_batch.idx_mapping] = logits
        self._staged_logits_gpu[idx, :num_reqs].copy_(logits)

        self._staged_req_ids[idx] = list(input_batch.req_ids)
        current_stream = torch.cuda.current_stream(self.req_states.device)
        self._copy_stream.wait_stream(current_stream)
        with stream(self._copy_stream, current_stream):
            self._staged_logits[idx, :num_reqs].copy_(
                self._staged_logits_gpu[idx, :num_reqs], non_blocking=True
            )
            self._copy_events[idx].record()

    def get_num_tokens(
        self,
        num_tokens_per_req: dict[str, int],
        draft_tokens: dict[str, list[int]],
        has_structured_output_requests: bool = False,
    ) -> int:
        raise NotImplementedError

    def _update_staged_confidences(self, idx: int) -> None:
        """Consume a two-step-old snapshot for budgeting."""
        req_ids = self._staged_req_ids[idx]
        num_reqs = len(req_ids)
        confidence_logits_np = self._staged_logits_np[idx, :num_reqs]
        self._stale_confidences = (
            req_ids,
            compute_prefix_survival(confidence_logits_np),
        )

    def _compute_draft_budget(
        self,
        req_ids: list[str],
        num_required_target_tokens_per_req: np.ndarray,
        valid_draft_tokens_per_req: np.ndarray,
    ) -> int:
        num_reqs = len(req_ids)
        survival = np.ones((num_reqs, self.num_speculative_steps), dtype=np.float64)
        if self._stale_confidences is not None:
            stale_req_ids, stale_survival = self._stale_confidences
            stale_rows = {req_id: i for i, req_id in enumerate(stale_req_ids)}
            for row, req_id in enumerate(req_ids):
                stale_row = stale_rows.get(req_id)
                if stale_row is not None:
                    survival[row] = stale_survival[stale_row]
        steps = np.arange(self.num_speculative_steps)
        valid = steps[None, :] < valid_draft_tokens_per_req[:, None]
        scores = np.sort(survival[valid & (survival > 0.0)])[::-1]
        if scores.size == 0 or self.cost_tables is None:
            return scores.size

        num_required_target_tokens = int(num_required_target_tokens_per_req.sum())
        draft_cost_ms, verify_cost_ms = self.cost_tables
        max_budget = min(
            scores.size,
            verify_cost_ms.size - num_required_target_tokens - 1,
        )
        if max_budget <= 0:
            return 0
        scores = scores[:max_budget]

        slots = np.fromiter(
            (self.req_states.req_id_to_index[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_sampling_requests = np.count_nonzero(
            self.req_states.num_computed_tokens_np[slots]
            + num_required_target_tokens_per_req
            >= self.req_states.prefill_len.np[slots]
        )
        accepted_tokens = np.concatenate(
            ([num_sampling_requests], num_sampling_requests + np.cumsum(scores))
        )
        costs = (
            draft_cost_ms[num_reqs]
            + verify_cost_ms[
                num_required_target_tokens : num_required_target_tokens + max_budget + 1
            ]
        )
        return int(np.argmax(accepted_tokens / costs))

    def _assign_draft_token_budget(
        self,
        idx_mapping: torch.Tensor,
        valid_draft_tokens_per_req: np.ndarray,
        draft_token_budget: int,
    ) -> torch.Tensor:
        num_reqs = idx_mapping.shape[0]
        capacities = self._batch_draft_capacity[:num_reqs]
        if draft_token_budget == 0:
            return capacities.zero_()

        async_copy_to_gpu(valid_draft_tokens_per_req, out=capacities)
        if draft_token_budget == int(valid_draft_tokens_per_req.sum()):
            return capacities

        block_size = triton.next_power_of_2(num_reqs * self.num_speculative_steps)
        _assign_draft_token_budget_kernel[(1,)](
            self._confidence_logits,
            self._confidence_logits.stride(0),
            idx_mapping,
            capacities,
            num_reqs,
            draft_token_budget,
            NUM_STEPS=self.num_speculative_steps,
            BLOCK_SIZE=block_size,
            num_warps=4 if block_size <= 256 else 8,
        )
        return capacities

    def warmup(self) -> None:
        max_num_reqs = self.req_states.max_num_reqs
        self._confidence_logits.zero_()
        idx_mapping = torch.arange(
            max_num_reqs, dtype=torch.int32, device=self.req_states.device
        )
        valid_drafts = np.full(max_num_reqs, self.num_speculative_steps, dtype=np.int32)
        block_size = triton.next_power_of_2(self.num_speculative_steps)
        max_block_size = triton.next_power_of_2(
            max_num_reqs * self.num_speculative_steps
        )
        while block_size <= max_block_size:
            num_reqs = min(max_num_reqs, block_size // self.num_speculative_steps)
            draft_token_budget = max(1, num_reqs * self.num_speculative_steps // 2)
            self._assign_draft_token_budget(
                idx_mapping[:num_reqs],
                valid_drafts[:num_reqs],
                draft_token_budget,
            )
            block_size *= 2

        self._staged_req_ids = [[], []]
        self._next_idx = 0
        self._stale_confidences = None

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
            self._compute_draft_budget(req_ids, required_targets, valid_drafts),
        )

    def allocate_draft_token_budget(
        self, idx_mapping: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        raise NotImplementedError

    def mask_batch(
        self,
        input_batch: "InputBatch",
        draft_tokens: dict[str, list[int]],
    ) -> None:
        pass


class VarlenConfidenceManager(ConfidenceManager):
    varlen_spec_decode = True

    def __init__(
        self,
        req_states: "RequestState",
        speculative_config: "SpeculativeConfig",
        query_start_loc: torch.Tensor,
        num_bonus_tokens: int,
    ):
        super().__init__(
            req_states,
            speculative_config,
            num_bonus_tokens,
        )
        self.query_start_loc = query_start_loc
        self._planned_batch: tuple[np.ndarray, np.ndarray, int] | None = None
        self.compact_batch = False

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
        self.compact_batch = bool(
            np.all(num_scheduled_tokens == num_draft_tokens_per_req + 1)
        )
        if not self.compact_batch:
            self._planned_batch = None
            return int(num_scheduled_tokens.sum())
        self._planned_batch = self._plan_batch(
            req_ids,
            num_scheduled_tokens,
            num_draft_tokens_per_req,
            draft_tokens,
            has_structured_output_requests,
        )
        required_target_tokens_per_req, _, draft_token_budget = self._planned_batch
        return int(required_target_tokens_per_req.sum()) + draft_token_budget

    def allocate_draft_token_budget(
        self, idx_mapping: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        planned_batch = self._planned_batch
        self._planned_batch = None
        assert planned_batch is not None
        required_targets, valid_drafts, budget = planned_batch
        assert self.compact_batch
        capacities = self._assign_draft_token_budget(idx_mapping, valid_drafts, budget)
        num_reqs = idx_mapping.shape[0]
        num_tokens = int(required_targets.sum()) + budget
        self.query_start_loc[:1].zero_()
        torch.cumsum(
            capacities + self.num_bonus_tokens,
            dim=0,
            out=self.query_start_loc[1 : num_reqs + 1],
        )
        self.query_start_loc[num_reqs + 1 :].fill_(num_tokens)
        return (
            self.query_start_loc[: num_reqs + 1],
            self.query_start_loc,
            budget,
        )


class MaskedConfidenceManager(ConfidenceManager):
    def mask_batch(
        self,
        input_batch: "InputBatch",
        draft_tokens: dict[str, list[int]],
    ) -> None:
        scheduled_drafts = input_batch.num_draft_tokens_per_req
        if scheduled_drafts is None:
            self._batch_draft_capacity[: input_batch.num_reqs].zero_()
            return
        _, valid_drafts, budget = self._plan_batch(
            input_batch.req_ids,
            input_batch.num_scheduled_tokens,
            scheduled_drafts,
            draft_tokens,
            input_batch.has_structured_output_reqs,
        )
        self._assign_draft_token_budget(input_batch.idx_mapping, valid_drafts, budget)
        query_ends = input_batch.query_start_loc[1 : input_batch.num_reqs + 1]
        num_logits = input_batch.cu_num_logits[1:] - input_batch.cu_num_logits[:-1]
        prune_starts = (
            query_ends
            - num_logits
            + self.num_bonus_tokens
            + self._batch_draft_capacity[: input_batch.num_reqs]
        )
        token_indices = prune_starts[:, None] + self._draft_steps
        pruned = token_indices < query_ends[:, None]
        token_indices.masked_fill_(~pruned, 0)
        input_batch.is_padding.scatter_(0, token_indices.flatten(), pruned.flatten())
        input_batch.input_ids[: input_batch.is_padding.shape[0]].masked_fill_(
            input_batch.is_padding, 0
        )


def make_confidence_manager(
    mode: str,
    attn_cg_support: "AttentionCGSupportInfo",
    req_states: "RequestState",
    speculative_config: "SpeculativeConfig",
    query_start_loc: torch.Tensor,
    num_bonus_tokens: int,
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
        return VarlenConfidenceManager(
            req_states,
            speculative_config,
            query_start_loc,
            num_bonus_tokens,
        )
    if mode == "mask":
        return MaskedConfidenceManager(
            req_states,
            speculative_config,
            num_bonus_tokens,
        )
    raise ValueError(f"Unknown confidence-based verification mode: {mode}")
