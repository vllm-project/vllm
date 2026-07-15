# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptive verification for DSpark speculative decoding."""

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.async_utils import stream
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.attn_utils import AttentionCGSupportInfo
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState


@triton.jit
def _keyed_product(left_value, left_key, right_value, right_key):
    value = tl.where(left_key == right_key, left_value * right_value, right_value)
    return value, right_key


@triton.jit(do_not_specialize=["num_reqs", "draft_token_budget"])
def _assign_draft_token_budget_kernel(
    confidence_probs_ptr,
    confidence_probs_stride,
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
    probability = tl.load(
        confidence_probs_ptr + req_state_idx * confidence_probs_stride + step,
        mask=valid_candidate,
        other=0.0,
    )
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
        result = np.interp(values, xs, ys)
        if len(xs) > 1:
            after = values > xs[-1]
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            result[after] = ys[-1] + slope * (values[after] - xs[-1])
        return result

    draft_table = np.maximum(build_table(max_num_reqs, draft_curve), 0.0)
    verify_table = np.maximum(build_table(max_batch_tokens, verify_curve), 1e-6)
    return draft_table, verify_table


class AdaptiveVerificationManager:
    varlen_spec_decode = False

    def __init__(
        self,
        req_states: "RequestState",
        num_bonus_tokens: int,
    ):
        self.req_states = req_states
        self.num_speculative_steps = req_states.num_speculative_steps
        device = req_states.device
        self._copy_stream = torch.cuda.Stream(device)

        self.num_bonus_tokens = num_bonus_tokens
        self.cost_tables: tuple[np.ndarray, np.ndarray] | None = None
        self._batch_budget: tuple[dict[str, int], dict[str, int], int] | None = None
        max_num_reqs = req_states.max_num_reqs
        self._confidence_probs = torch.empty(
            (max_num_reqs, self.num_speculative_steps),
            dtype=torch.float32,
            device=device,
        )
        self._batch_draft_capacity = torch.empty(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self._draft_steps_arange = torch.arange(
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )

        # Two D2H slots preserve stale inputs for budget selection.
        self._staged_probs = [
            CpuGpuBuffer(
                max_num_reqs,
                self.num_speculative_steps,
                dtype=torch.float32,
                device=device,
            )
            for _ in range(2)
        ]
        self._copy_events = [torch.cuda.Event(blocking=True) for _ in range(2)]
        self._pending_resets: list[int] = []
        self._stale_idx = 0
        for staged_probs in self._staged_probs:
            staged_probs.np.fill(1.0)

    def add_request(self, req_idx: int) -> None:
        self._staged_probs[self._stale_idx].np[req_idx].fill(1.0)
        self._pending_resets.append(req_idx)
        self._confidence_probs[req_idx].fill_(1.0)

    def set_graph_costs(
        self,
        model_graphs: dict[Any, float],
        draft_graphs: dict[Any, float],
    ) -> None:
        def collapse(timings: dict[Any, float], field: str) -> list[tuple[int, float]]:
            full_points: dict[int, float] = {}
            piecewise_points: dict[int, float] = {}
            for desc, ms in timings.items():
                if desc.num_active_loras == 0:
                    x = getattr(desc, field)
                    assert x is not None
                    if desc.cg_mode == CUDAGraphMode.FULL:
                        full_points[x] = max(full_points.get(x, 0.0), ms)
                    elif desc.cg_mode == CUDAGraphMode.PIECEWISE:
                        piecewise_points[x] = max(piecewise_points.get(x, 0.0), ms)
            return sorted((piecewise_points | full_points).items())

        draft_curve = collapse(draft_graphs, "num_reqs")
        verify_curve = collapse(model_graphs, "num_tokens")
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
        confidence_probs: torch.Tensor,
        input_batch: "InputBatch",
    ) -> None:
        """Preserve current GPU scores and stage stale budget inputs."""
        num_reqs = input_batch.num_reqs
        write_idx = self._stale_idx ^ 1
        with gpu_sync_allowed():
            self._copy_events[write_idx].synchronize()
        if self._pending_resets:
            self._staged_probs[write_idx].np[self._pending_resets] = 1.0
            self._pending_resets.clear()
        self._stale_idx = write_idx
        write_idx ^= 1

        probs = confidence_probs[:num_reqs]
        self._confidence_probs[input_batch.idx_mapping] = probs
        staged_probs = self._staged_probs[write_idx]
        staged_probs.gpu.copy_(self._confidence_probs)

        current_stream = torch.cuda.current_stream(self.req_states.device)
        self._copy_stream.wait_stream(current_stream)
        with stream(self._copy_stream, current_stream):
            staged_probs.copy_to_cpu()
            self._copy_events[write_idx].record()

    def get_num_tokens(
        self,
        num_tokens_per_req: dict[str, int],
        draft_tokens: dict[str, list[int]],
        has_structured_output_requests: bool = False,
        uniform_tok_count: int | None = None,
    ) -> tuple[int, int | None]:
        self._batch_budget = self._compute_budget(
            num_tokens_per_req,
            draft_tokens,
            has_structured_output_requests,
        )
        _, num_non_draft_tokens_per_req, draft_budget = self._batch_budget
        if self.varlen_spec_decode:
            return sum(num_non_draft_tokens_per_req.values()) + draft_budget, None
        return sum(num_tokens_per_req.values()), uniform_tok_count

    def _compute_budget(
        self,
        num_tokens_per_req: dict[str, int],
        draft_tokens: dict[str, list[int]],
        has_structured_output: bool,
    ) -> tuple[dict[str, int], dict[str, int], int]:
        assert self.cost_tables is not None
        req_ids = list(num_tokens_per_req)
        num_reqs = len(req_ids)
        scheduled_tokens = np.fromiter(
            num_tokens_per_req.values(), dtype=np.int32, count=num_reqs
        )
        scheduled_drafts = np.fromiter(
            (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
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
        num_non_draft_tokens = scheduled_tokens - scheduled_drafts
        slots = np.fromiter(
            (self.req_states.req_id_to_index[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=len(req_ids),
        )
        survival_probability = np.cumprod(
            self._staged_probs[self._stale_idx].np[slots].astype(np.float64),
            axis=1,
        )
        steps = np.arange(self.num_speculative_steps)
        valid = steps[None, :] < valid_drafts[:, None]
        scores = np.sort(survival_probability[valid])[::-1]
        num_non_draft_tokens_total = int(num_non_draft_tokens.sum())
        max_draft_budget = int(valid_drafts.sum())
        draft_cost_ms, verify_cost_ms = self.cost_tables
        num_sampling_requests = np.count_nonzero(
            self.req_states.num_computed_tokens_np[slots] + num_non_draft_tokens
            >= self.req_states.prefill_len.np[slots]
        )
        num_tokens_to_estimated_accepted_tokens = np.concatenate(
            ([num_sampling_requests], num_sampling_requests + np.cumsum(scores))
        )
        costs = (
            draft_cost_ms[len(req_ids)]
            + verify_cost_ms[
                num_non_draft_tokens_total : num_non_draft_tokens_total
                + max_draft_budget
                + 1
            ]
        )
        valid_drafts_per_req = {
            req_id: int(num_valid_drafts)
            for req_id, num_valid_drafts in zip(req_ids, valid_drafts, strict=True)
        }
        num_non_draft_tokens_per_req = {
            req_id: int(num_tokens)
            for req_id, num_tokens in zip(req_ids, num_non_draft_tokens, strict=True)
        }
        draft_budget = int(np.argmax(num_tokens_to_estimated_accepted_tokens / costs))
        return valid_drafts_per_req, num_non_draft_tokens_per_req, draft_budget

    def allocate_draft_token_budget(
        self,
        req_ids: list[str],
        idx_mapping: torch.Tensor,
        num_tokens: int | None = None,
    ) -> tuple[torch.Tensor, np.ndarray, int]:
        batch_budget = self._batch_budget
        self._batch_budget = None
        assert batch_budget is not None
        valid_drafts_per_req, num_non_draft_tokens_per_req, draft_budget = batch_budget
        valid_drafts = np.fromiter(
            (valid_drafts_per_req[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=len(req_ids),
        )
        num_non_draft_tokens = np.fromiter(
            (num_non_draft_tokens_per_req[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=len(req_ids),
        )
        num_scheduled_draft_tokens = int(valid_drafts.sum())
        if num_tokens is not None:
            draft_budget = min(
                num_scheduled_draft_tokens,
                num_tokens - int(num_non_draft_tokens.sum()),
            )
        num_reqs = idx_mapping.shape[0]
        capacities = self._batch_draft_capacity[:num_reqs]
        if draft_budget == 0:
            capacities.zero_()
        else:
            async_copy_to_gpu(valid_drafts, out=capacities)
            if draft_budget != num_scheduled_draft_tokens:
                block_size = triton.next_power_of_2(
                    num_reqs * self.num_speculative_steps
                )
                _assign_draft_token_budget_kernel[(1,)](
                    self._confidence_probs,
                    self._confidence_probs.stride(0),
                    idx_mapping,
                    capacities,
                    num_reqs,
                    draft_budget,
                    NUM_STEPS=self.num_speculative_steps,
                    BLOCK_SIZE=block_size,
                    num_warps=4 if block_size <= 256 else 8,
                )
        return capacities, num_non_draft_tokens, draft_budget

    def compact_batch(
        self, req_ids: list[str], idx_mapping: torch.Tensor, num_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        raise NotImplementedError

    def mask_batch(self, input_batch: "InputBatch") -> None:
        pass


class VarlenAdaptiveVerificationManager(AdaptiveVerificationManager):
    varlen_spec_decode = True

    def __init__(
        self,
        req_states: "RequestState",
        query_start_loc: torch.Tensor,
        num_bonus_tokens: int,
    ):
        super().__init__(
            req_states,
            num_bonus_tokens,
        )
        self.query_start_loc = query_start_loc
        self._num_non_draft_tokens = torch.empty_like(query_start_loc[:-1])
        self._cu_num_logits = torch.empty_like(query_start_loc)

    def compact_batch(
        self, req_ids: list[str], idx_mapping: torch.Tensor, num_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        capacities, num_non_draft_tokens, draft_budget = (
            self.allocate_draft_token_budget(req_ids, idx_mapping, num_tokens)
        )
        num_reqs = idx_mapping.shape[0]
        num_tokens = int(num_non_draft_tokens.sum()) + draft_budget
        num_non_draft_tokens_gpu = self._num_non_draft_tokens[:num_reqs]
        async_copy_to_gpu(
            num_non_draft_tokens,
            out=num_non_draft_tokens_gpu,
        )
        self._cu_num_logits[:1].zero_()
        torch.cumsum(
            capacities + self.num_bonus_tokens,
            dim=0,
            out=self._cu_num_logits[1 : num_reqs + 1],
        )
        self.query_start_loc[:1].zero_()
        torch.cumsum(
            capacities + num_non_draft_tokens_gpu,
            dim=0,
            out=self.query_start_loc[1 : num_reqs + 1],
        )
        self.query_start_loc[num_reqs + 1 :].fill_(num_tokens)
        return (
            self._cu_num_logits[: num_reqs + 1],
            self.query_start_loc,
            draft_budget,
        )


class MaskedAdaptiveVerificationManager(AdaptiveVerificationManager):
    def mask_batch(self, input_batch: "InputBatch") -> None:
        scheduled_drafts = input_batch.num_draft_tokens_per_req
        if scheduled_drafts is None:
            self._batch_draft_capacity[: input_batch.num_reqs].zero_()
            return
        capacities, _, _ = self.allocate_draft_token_budget(
            input_batch.req_ids, input_batch.idx_mapping
        )
        query_ends = input_batch.query_start_loc[1 : input_batch.num_reqs + 1]
        num_logits = input_batch.cu_num_logits[1:] - input_batch.cu_num_logits[:-1]
        prune_starts = query_ends - num_logits + self.num_bonus_tokens + capacities
        token_indices = prune_starts[:, None] + self._draft_steps_arange
        pruned = token_indices < query_ends[:, None]
        token_indices.masked_fill_(~pruned, 0)
        input_batch.is_padding.scatter_(0, token_indices.flatten(), pruned.flatten())
        input_batch.input_ids[: input_batch.is_padding.shape[0]].masked_fill_(
            input_batch.is_padding, 0
        )


def make_adaptive_verification_manager(
    mode: str,
    attn_cg_support: "AttentionCGSupportInfo",
    req_states: "RequestState",
    query_start_loc: torch.Tensor,
    num_bonus_tokens: int,
) -> AdaptiveVerificationManager:
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
        return VarlenAdaptiveVerificationManager(
            req_states,
            query_start_loc,
            num_bonus_tokens,
        )
    if mode == "mask":
        return MaskedAdaptiveVerificationManager(
            req_states,
            num_bonus_tokens,
        )
    raise ValueError(f"Unknown confidence-based verification mode: {mode}")
