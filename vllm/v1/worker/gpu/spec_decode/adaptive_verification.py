# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptive verification for DSpark speculative decoding."""

from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch

import vllm.envs as envs
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.async_utils import StepTimingSample, stream
from vllm.v1.worker.gpu.attn_utils import (
    get_query_lens_mismatch_unsupported_backend,
    get_varlen_cudagraph_unsupported_backend,
)
from vllm.v1.worker.gpu.spec_decode.pcp_adaptive_verification import (
    PCPAdaptiveVerification,
)

logger = init_logger(__name__)
_PROFILE_REPLAYS = 5

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState
    from vllm.v1.worker.utils import AttentionGroup


def resolve_adaptive_cudagraph_mode(
    mode: CUDAGraphMode, *, piecewise_capture_available: bool
) -> CUDAGraphMode:
    """Select a separate decode route for variable-length verification."""
    assert mode.has_full_cudagraphs()
    if mode == CUDAGraphMode.FULL_DECODE_ONLY or not piecewise_capture_available:
        return CUDAGraphMode.FULL_DECODE_ONLY
    return CUDAGraphMode.FULL_AND_PIECEWISE


@triton.jit
def _count_survival_at_least(
    survival_ptr, num_slots, threshold, BLOCK_SIZE: tl.constexpr
):
    count = 0
    for start in range(0, num_slots, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        bits = tl.load(survival_ptr + offsets, mask=offsets < num_slots, other=-1)
        count += tl.sum((bits >= threshold).to(tl.int32))
    return count


@triton.jit(do_not_specialize=["num_reqs", "draft_budget"])
def _assign_draft_token_budget_kernel(
    # [max_num_reqs, num_steps]
    confidence_probs_ptr,
    confidence_probs_stride,
    # [num_reqs]
    idx_mapping_ptr,
    # [num_reqs]
    capacities_ptr,
    # [max_num_reqs * num_steps]
    survival_ptr,
    num_reqs,
    draft_budget,
    NUM_STEPS: tl.constexpr,
    BLOCK_REQS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Survival probabilities are non-negative, so their float bits order like
    # the values; ineligible slots store -1 to rank below every real draft.
    for start in range(0, num_reqs, BLOCK_REQS):
        reqs = start + tl.arange(0, BLOCK_REQS)
        mask = reqs < num_reqs
        req_state_idx = tl.load(idx_mapping_ptr + reqs, mask=mask, other=0)
        capacity = tl.load(capacities_ptr + reqs, mask=mask, other=0)
        survival = tl.full([BLOCK_REQS], 1.0, tl.float32)
        for step in tl.static_range(NUM_STEPS):
            prob = tl.load(
                confidence_probs_ptr + req_state_idx * confidence_probs_stride + step,
                mask=mask,
                other=0.0,
            )
            survival *= prob
            bits = tl.where(step < capacity, survival.to(tl.int32, bitcast=True), -1)
            tl.store(survival_ptr + reqs * NUM_STEPS + step, bits, mask=mask)
    tl.debug_barrier()

    # Binary search the largest threshold that still admits draft_budget slots.
    num_slots = num_reqs * NUM_STEPS
    lo = 0
    hi = 0x7F800000
    while lo < hi:
        mid = lo + (hi - lo + 1) // 2
        if (
            _count_survival_at_least(survival_ptr, num_slots, mid, BLOCK_SIZE)
            >= draft_budget
        ):
            lo = mid
        else:
            hi = mid - 1
    ties_left = draft_budget - _count_survival_at_least(
        survival_ptr, num_slots, lo + 1, BLOCK_SIZE
    )

    # Admit every slot above the threshold and hand out ties in request order.
    for start in range(0, num_reqs, BLOCK_REQS):
        reqs = start + tl.arange(0, BLOCK_REQS)
        mask = reqs < num_reqs
        num_above = tl.zeros([BLOCK_REQS], tl.int32)
        num_tied = tl.zeros([BLOCK_REQS], tl.int32)
        for step in tl.static_range(NUM_STEPS):
            bits = tl.load(survival_ptr + reqs * NUM_STEPS + step, mask=mask, other=-1)
            num_above += (bits > lo).to(tl.int32)
            num_tied += (bits == lo).to(tl.int32)
        tied_before = tl.cumsum(num_tied, axis=0) - num_tied
        num_taken = tl.minimum(num_tied, tl.maximum(ties_left - tied_before, 0))
        tl.store(capacities_ptr + reqs, num_above + num_taken, mask=mask)
        ties_left -= tl.sum(num_tied)


def assign_draft_token_budget(
    confidence_probs: torch.Tensor,
    idx_mapping: torch.Tensor,
    capacities: torch.Tensor,
    draft_budget: int,
    survival: torch.Tensor,
) -> None:
    """Admit the globally best `draft_budget` draft slots, in place.

    Every (request, step) slot is scored by its survival probability, the running
    product of that request's per-position confidences, and the highest scores win.
    Survival only decreases along a request, so the winners are always a prefix of
    each request's steps and only per-request counts are needed.

    `capacities` enters holding each request's scheduled draft count (which bounds its
    eligible slots) and leaves holding the admitted count. The caller only calls this
    when `draft_budget < capacities.sum()`, so every winner is a real draft slot.
    `survival` is int32 scratch with room for `num_reqs * num_steps` slots.
    """
    num_steps = confidence_probs.shape[1]
    _assign_draft_token_budget_kernel[(1,)](
        confidence_probs,
        confidence_probs.stride(0),
        idx_mapping,
        capacities,
        survival,
        idx_mapping.shape[0],
        draft_budget,
        NUM_STEPS=num_steps,
        BLOCK_REQS=256,
        BLOCK_SIZE=1024,
        num_warps=8,
    )


def build_cost_tables_from_curves(
    draft_curve: list[tuple[int, float]],
    verify_curve: list[tuple[int, float]],
    max_num_reqs: int,
    max_batch_tokens: int,
    cudagraph_limit: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build cost tables: graph-padded below the capture limit, smooth above.

    Args:
        draft_curve: (size, cost) samples for the draft model.
        verify_curve: (size, cost) samples for the verify model.
        max_num_reqs: Largest request count to build a table for.
        max_batch_tokens: Largest token count to build a table for.
        cudagraph_limit: Largest cudagraph-captured size. At or below it,
            execution pads up to the next captured size, so cost is a step
            function. Above it there is no padding, so cost is continuous.

    """

    def build_table(limit: int, curve: list[tuple[int, float]]) -> np.ndarray:
        xs, ys = np.asarray(curve, dtype=np.float64).T
        ys = np.maximum.accumulate(ys)
        values = np.arange(limit + 1)
        # Execution pads to the next captured size, so cost is a step
        # function of the padded size: smooth interpolation would invent
        # marginal per-token costs that don't exist within a pad bucket.
        idx = np.searchsorted(xs, values, side="left")
        result = ys[np.minimum(idx, len(xs) - 1)]
        # Past the capture limit nothing pads, so cost really is continuous in
        # size; snapping to the next profiled point overestimates badly when
        # the profiled points are far apart. Interpolate only between points
        # that are themselves past the limit: crossing the limit loses
        # cudagraphs entirely, which is a genuine discontinuity, so the first
        # point above it must not be blended with the last one below.
        if cudagraph_limit:
            smooth = values > cudagraph_limit
            above = xs > cudagraph_limit
            if smooth.any() and above.any():
                result[smooth] = np.interp(values[smooth], xs[above], ys[above])
        if len(xs) > 1:
            after = values > xs[-1]
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            result[after] = ys[-1] + slope * (values[after] - xs[-1])
        return result

    draft_table = np.maximum(build_table(max_num_reqs, draft_curve), 0.0)
    verify_table = np.maximum(build_table(max_batch_tokens, verify_curve), 1e-6)
    return draft_table, verify_table


class AdmittedDraftCounts:
    """Per-request admitted draft counts, in flight to the host."""

    def __init__(self, cpu: torch.Tensor, event: torch.cuda.Event | None):
        self._cpu = cpu
        self._event = event

    def wait(self) -> np.ndarray:
        if self._event is not None:
            with gpu_sync_allowed():
                self._event.synchronize()
        return self._cpu.numpy()


class AdaptiveVerificationManager:
    def __init__(
        self,
        req_states: "RequestState",
        query_start_loc: torch.Tensor,
        num_bonus_tokens: int,
        max_total_logits: int,
        copy_admitted_drafts_to_host: bool = False,
        pcp_size: int = 1,
        pcp_shard_decode_requests: bool = True,
    ):
        self.req_states = req_states
        self.pcp = (
            PCPAdaptiveVerification(pcp_size, pcp_shard_decode_requests)
            if pcp_size > 1
            else None
        )
        self.num_speculative_steps = req_states.num_speculative_steps
        device = req_states.device
        self._copy_stream = torch.cuda.Stream(device)

        self.num_bonus_tokens = num_bonus_tokens
        # Rejection sampling verifies logits in one contiguous chunk; the
        # chunked path indexes by scheduled (untrimmed) offsets and cannot
        # address the compacted layout, so the budget must fit one chunk.
        self._max_total_logits = max_total_logits
        self.query_start_loc = query_start_loc
        self.cost_tables: tuple[np.ndarray, np.ndarray] | None = None
        # Largest cudagraph-captured token count; above it nothing pads.
        self._cudagraph_limit = 0
        self._batch_budget: tuple[dict[str, int], dict[str, int], int] | None = None
        max_num_reqs = req_states.max_num_reqs
        # Current per-slot confidences
        self._confidence_probs = torch.empty(
            (max_num_reqs, self.num_speculative_steps),
            dtype=torch.float32,
            device=device,
        )
        self._batch_draft_capacity = torch.empty(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self._draft_survival = torch.empty(
            max_num_reqs * self.num_speculative_steps, dtype=torch.int32, device=device
        )
        self._num_non_draft_tokens = torch.empty_like(query_start_loc[:-1])
        self._cu_num_logits = torch.empty_like(query_start_loc)

        # Batch-sharded sampling sizes its all-to-all from host counts, so it
        # needs the admitted split on the host.
        self._copy_admitted_drafts_to_host = copy_admitted_drafts_to_host
        if copy_admitted_drafts_to_host:
            self._admitted_drafts_cpu = torch.empty(
                max_num_reqs, dtype=torch.int32, pin_memory=True
            )
            self._admitted_drafts_event = torch.cuda.Event(blocking=True)

        # Two D2H slots preserve stale inputs for budget selection.
        self._stale_confidences = [
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
        for slot in self._stale_confidences:
            slot.np.fill(1.0)

    def warmup_budget_kernel(self) -> None:
        """JIT the draft-budget kernel so the first binding step doesn't compile."""
        capacities = self._batch_draft_capacity[:1].fill_(self.num_speculative_steps)
        idx_mapping = torch.zeros(1, dtype=torch.int64, device=capacities.device)
        assign_draft_token_budget(
            self._confidence_probs, idx_mapping, capacities, 1, self._draft_survival
        )

    def add_request(self, req_idx: int) -> None:
        self._stale_confidences[self._stale_idx].np[req_idx].fill(1.0)
        self._pending_resets.append(req_idx)
        self._confidence_probs[req_idx].fill_(1.0)

    def batches_to_profile(self, capture_sizes: list[int]) -> Iterator[dict[str, int]]:
        """Dummy-run kwargs whose step timings seed the cost tables.

        Run these inside StepTimingCollector.collect(), then hand the block's
        timings to set_initial_cost_curves."""
        max_num_tokens = self.req_states.max_num_batched_tokens
        size = self._cudagraph_limit = capture_sizes[-1] if capture_sizes else 0
        # Also profile beyond the capture limit: real steps run there
        # (piecewise/eager) and linear extrapolation badly underestimates
        # them. These runs double as JIT warmup for the piecewise shapes.
        tail_sizes: set[int] = set()
        if size:
            tail_sizes.add(min(size + size // 2, max_num_tokens))
            while size < max_num_tokens:
                size = min(size * 2, max_num_tokens)
                tail_sizes.add(size)
            tail_sizes -= set(capture_sizes)
        for num_tokens in capture_sizes + sorted(tail_sizes):
            for _ in range(_PROFILE_REPLAYS):
                yield {
                    "num_tokens": num_tokens,
                    "context_len": envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN,
                }

    def set_initial_cost_curves(self, samples: list[StepTimingSample]) -> None:
        def median_curve(
            points: Iterable[tuple[int, float]],
        ) -> list[tuple[int, float]]:
            grouped: defaultdict[int, list[float]] = defaultdict(list)
            for key, value in points:
                grouped[key].append(value)
            return [(k, float(np.median(v))) for k, v in sorted(grouped.items())]

        # Draft curve: eager-target steps inflate drafter timings (the CPU is
        # still launching kernels, opening gaps between the drafter's events),
        # and request counts — unlike token counts — collide across execution
        # modes, so only graph-replay samples may price the draft curve.
        draft_curve = median_curve(
            (s.num_reqs, s.drafter_ms) for s in samples if s.full_cudagraph
        )
        verify_curve = median_curve(
            (s.num_target_tokens, s.forward_ms) for s in samples
        )
        self.set_cost_curves(draft_curve, verify_curve)

    def set_cost_curves(
        self,
        draft_curve: list[tuple[int, float]],
        verify_curve: list[tuple[int, float]],
    ) -> None:
        draft_curve, verify_curve = get_tp_group().broadcast_object(
            (draft_curve, verify_curve), src=0
        )
        if not draft_curve or not verify_curve:
            raise RuntimeError(
                "Adaptive verification could not profile step costs. Pass "
                "`enable_adaptive_verification=false` in the speculative config to "
                "verify a fixed number of drafts instead."
            )
        self.cost_tables = build_cost_tables_from_curves(
            draft_curve,
            verify_curve,
            self.req_states.max_num_reqs,
            self.req_states.max_num_batched_tokens,
            self._cudagraph_limit,
        )
        logger.debug("DSpark cost tables: %s", self.cost_tables)

    def record_confidences(
        self, confidence_probs: torch.Tensor, input_batch: "InputBatch"
    ) -> None:
        """Publish this step's raw confidences for the ranking kernel and start
        copying them to the CPU, where a later step's budget reads them."""
        num_reqs = input_batch.num_reqs
        ready_idx = self._stale_idx ^ 1
        with gpu_sync_allowed():
            self._copy_events[ready_idx].synchronize()
        if self._pending_resets:
            self._stale_confidences[ready_idx].np[self._pending_resets] = 1.0
            self._pending_resets.clear()
        # Last step's copy has landed: budgets read it, this step overwrites the
        # slot they were reading before.
        self._stale_idx, write_idx = ready_idx, self._stale_idx

        self._confidence_probs[input_batch.idx_mapping] = confidence_probs[:num_reqs]
        write_slot = self._stale_confidences[write_idx]
        write_slot.gpu.copy_(self._confidence_probs)

        current_stream = torch.cuda.current_stream(self.req_states.device)
        self._copy_stream.wait_stream(current_stream)
        with stream(self._copy_stream, current_stream):
            write_slot.copy_to_cpu()
            self._copy_events[write_idx].record()

    def get_num_tokens(
        self,
        num_tokens_per_req: dict[str, int],
        draft_tokens: dict[str, list[int]],
        *,
        has_prefill: bool = False,
    ) -> int:
        """Token count once the draft budget is trimmed to fit.

        Stashes the chosen budget in ``_batch_budget`` for the compaction and
        reallocation that follow in the same step.
        """
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
        num_non_draft_tokens = scheduled_tokens - scheduled_drafts
        slots = np.fromiter(
            map(self.req_states.req_id_to_index.__getitem__, req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        stale_confidences = self._stale_confidences[self._stale_idx].np[slots]
        survival_probability = np.cumprod(stale_confidences.astype(np.float64), axis=1)
        steps = np.arange(self.num_speculative_steps)
        valid = steps[None, :] < scheduled_drafts[:, None]
        scores = np.sort(survival_probability[valid])[::-1]
        num_non_draft_tokens_total = int(num_non_draft_tokens.sum())
        max_draft_budget = min(
            int(scheduled_drafts.sum()),
            max(0, self._max_total_logits - num_reqs * self.num_bonus_tokens),
        )
        scores = scores[:max_draft_budget]
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
        num_drafts_per_req = {
            req_id: int(num_drafts)
            for req_id, num_drafts in zip(req_ids, scheduled_drafts, strict=True)
        }
        num_non_draft_tokens_per_req = {
            req_id: int(num_tokens)
            for req_id, num_tokens in zip(req_ids, num_non_draft_tokens, strict=True)
        }
        draft_budget = int(np.argmax(num_tokens_to_estimated_accepted_tokens / costs))
        if self.pcp is not None:
            draft_budget = self.pcp.plan(
                survival_probability,
                scheduled_drafts,
                num_non_draft_tokens,
                self.cost_tables,
                max_draft_budget,
                has_prefill,
            )
        self._batch_budget = (
            num_drafts_per_req,
            num_non_draft_tokens_per_req,
            draft_budget,
        )
        return sum(num_non_draft_tokens_per_req.values()) + draft_budget

    def compact_batch(
        self,
        num_draft_tokens_per_req: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        cu_num_logits_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compact the CPU batch to the chosen draft budget.

        Returns the compacted per-request token counts and the CPU cu_num_logits_np.
        If the draft budget is 0, we can know cu_num_logits_np exactly, otherwise
        its unchanged/an-upper-bound.
        """
        batch_budget = self._batch_budget
        assert batch_budget is not None
        _, _, draft_budget = batch_budget
        if self.pcp is not None:
            assert self.pcp.admitted_drafts is not None
            assert self.pcp.num_tokens_per_req is not None
            cu_num_logits_np = np.zeros_like(cu_num_logits_np)
            np.cumsum(
                self.pcp.admitted_drafts + self.num_bonus_tokens,
                out=cu_num_logits_np[1:],
            )
            return self.pcp.num_tokens_per_req, cu_num_logits_np
        num_drafts = int(num_draft_tokens_per_req.sum())
        if draft_budget == num_drafts:
            return num_scheduled_tokens, cu_num_logits_np

        num_non_draft_tokens = num_scheduled_tokens - num_draft_tokens_per_req
        if draft_budget == 0:
            # The draft budget is 0, so we can know cu_num_logits_np exactly. This helps
            # when we would exceed the sampler logit chunk size.
            num_reqs = num_scheduled_tokens.shape[0]
            cu_num_logits_np = (
                np.arange(num_reqs + 1, dtype=cu_num_logits_np.dtype)
                * self.num_bonus_tokens
            )
            return num_non_draft_tokens, cu_num_logits_np

        is_verification_request = num_draft_tokens_per_req > 0
        num_verification_reqs = int(is_verification_request.sum())
        # sort_batch_req_ids keeps verification requests at the front.
        assert np.all(is_verification_request[:num_verification_reqs])
        # for the CPU side buffer we distribute draft tokens evenly
        draft_lens_cpu = np.zeros_like(num_non_draft_tokens)
        draft_lens_cpu[:num_verification_reqs] = draft_budget // num_verification_reqs
        draft_lens_cpu[: draft_budget % num_verification_reqs] += 1
        return num_non_draft_tokens + draft_lens_cpu, cu_num_logits_np

    def reallocate_drafts(
        self, req_ids: list[str], idx_mapping: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int, AdmittedDraftCounts | None]:
        """The admitted draft counts are None unless copy_admitted_drafts_to_host
        is set and the budget binds; otherwise the host counts are already exact.
        """
        batch_budget, self._batch_budget = self._batch_budget, None
        assert batch_budget is not None
        num_drafts_per_req, num_non_draft_tokens_per_req, draft_budget = batch_budget
        num_reqs = idx_mapping.shape[0]
        scheduled_drafts = np.fromiter(
            (num_drafts_per_req[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_non_draft_tokens = np.fromiter(
            (num_non_draft_tokens_per_req[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_tokens = int(num_non_draft_tokens.sum()) + draft_budget

        # Rank draft slots by survival probability and admit the best prefix.
        # capacities enters holding each request's valid draft count (the kernel
        # uses it to bound eligible slots) and leaves holding the admitted count.
        capacities = self._batch_draft_capacity[:num_reqs]
        if self.pcp is not None:
            assert self.pcp.admitted_drafts is not None
            async_tensor_h2d(self.pcp.admitted_drafts, out=capacities)
        elif draft_budget == 0:
            capacities.zero_()
        else:
            async_tensor_h2d(scheduled_drafts, out=capacities)
            if draft_budget < int(scheduled_drafts.sum()):
                assign_draft_token_budget(
                    self._confidence_probs,
                    idx_mapping,
                    capacities,
                    draft_budget,
                    self._draft_survival,
                )

        num_non_draft_tokens_gpu = self._num_non_draft_tokens[:num_reqs]
        async_tensor_h2d(num_non_draft_tokens, out=num_non_draft_tokens_gpu)
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

        admitted_drafts = None
        if self.pcp is not None:
            admitted_drafts = AdmittedDraftCounts(
                torch.from_numpy(self.pcp.admitted_drafts), None
            )
        elif self._copy_admitted_drafts_to_host and (
            0 < draft_budget < int(scheduled_drafts.sum())
        ):
            admitted_cpu = self._admitted_drafts_cpu[:num_reqs]
            current_stream = torch.cuda.current_stream(self.req_states.device)
            self._copy_stream.wait_stream(current_stream)
            with stream(self._copy_stream, current_stream):
                admitted_cpu.copy_(capacities, non_blocking=True)
                self._admitted_drafts_event.record()
            admitted_drafts = AdmittedDraftCounts(
                admitted_cpu, self._admitted_drafts_event
            )
        return (
            self._cu_num_logits[: num_reqs + 1],
            self.query_start_loc,
            draft_budget,
            admitted_drafts,
        )


def maybe_create_adaptive_verification_manager(
    *,
    enable_adaptive_verification: bool,
    attn_groups: list[list["AttentionGroup"]],
    req_states: "RequestState",
    query_start_loc: torch.Tensor,
    num_bonus_tokens: int,
    max_total_logits: int,
    vllm_config: "VllmConfig",
    target_layer_names: set[str] | None = None,
    additional_attn_cg_support: tuple[AttentionCGSupport, str | None] | None = None,
    copy_admitted_drafts_to_host: bool = False,
    pcp_size: int = 1,
    pcp_shard_decode_requests: bool = True,
) -> AdaptiveVerificationManager | None:
    if not enable_adaptive_verification:
        return None

    # The selector rejects unsupported backends, but models that
    # hard-wire theirs (e.g. DeepSeek-V4) never go through it.
    backend = get_query_lens_mismatch_unsupported_backend(
        attn_groups, checked_layer_names=target_layer_names
    )
    if backend is not None:
        raise ValueError(
            "Adaptive verification trims verification requests on device, which"
            f" the {backend} attention backend does not support. Pass "
            "enable_adaptive_verification=false in the speculative config, or "
            "use a backend that does."
        )

    # The runner's decode_query_len, the width varlen decode graphs capture.
    max_query_len = req_states.num_speculative_steps + num_bonus_tokens
    unsupported: tuple[str | None, int | None] | None = (
        get_varlen_cudagraph_unsupported_backend(
            attn_groups,
            vllm_config,
            max_query_len,
            checked_layer_names=target_layer_names,
        )
    )
    if unsupported is None and additional_attn_cg_support is not None:
        # Groups built outside init_attn_backend report only their support
        # level, and without a bound only ALWAYS replays varlen batches.
        additional_support, additional_backend = additional_attn_cg_support
        if additional_support != AttentionCGSupport.ALWAYS:
            unsupported = additional_backend, None
    if unsupported is not None:
        backend, bound = unsupported
        allowed = "none" if bound is None else f"at most {bound}"
        raise ValueError(
            "Adaptive verification replays decode cudagraphs whose per-request "
            f"query lengths vary up to {max_query_len}, but {backend} allows "
            f"{allowed}. Pass enable_adaptive_verification=false in the "
            "speculative config, or use a backend that does."
        )

    return AdaptiveVerificationManager(
        req_states,
        query_start_loc,
        num_bonus_tokens,
        max_total_logits=max_total_logits,
        copy_admitted_drafts_to_host=copy_admitted_drafts_to_host,
        pcp_size=pcp_size,
        pcp_shard_decode_requests=pcp_shard_decode_requests,
    )
