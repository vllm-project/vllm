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
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.async_utils import StepTimingSample, stream
from vllm.v1.worker.gpu.attn_utils import (
    get_query_lens_mismatch_unsupported_backend,
    get_varlen_cudagraph_unsupported_backend,
)
from vllm.v1.worker.gpu.spec_decode.undrafted_confidence import (
    UndraftedConfidenceEstimator,
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


def _assign_draft_token_budget(
    confidence_probs: torch.Tensor,
    idx_mapping: torch.Tensor,
    capacities: torch.Tensor,
    draft_budget: int,
    num_steps: int,
) -> None:
    """Admit the globally best `draft_budget` draft slots, in place.

    Every (request, step) slot is scored by its survival probability, the running
    product of that request's per-position confidences, and the highest scores win.
    Survival only decreases along a request, so a global top-k always admits
    continuously along steps with a request.

    `capacities` enters holding each request's scheduled draft count (which bounds its
    eligible slots) and leaves holding the admitted count. The caller only calls this
    when `draft_budget < capacities.sum()`, so every winner is a real draft slot.
    """
    survival = confidence_probs[idx_mapping].cumprod(dim=1)
    steps = torch.arange(num_steps, device=survival.device)
    # Out-of-range slots score -inf so they never outrank a real draft.
    out_of_range = steps[None, :] >= capacities[:, None]
    survival = survival.masked_fill(out_of_range, -float("inf"))
    flat = survival.flatten()
    winners = flat.topk(draft_budget).indices
    admitted = torch.zeros_like(flat, dtype=torch.bool).index_fill_(0, winners, True)
    torch.sum(admitted.view_as(survival), dim=1, dtype=capacities.dtype, out=capacities)


_assign_draft_token_budget_compiled = torch.compile(
    _assign_draft_token_budget, dynamic=True
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


class AdaptiveVerificationManager:
    def __init__(
        self,
        req_states: "RequestState",
        query_start_loc: torch.Tensor,
        num_bonus_tokens: int,
        max_total_logits: int,
        use_adaptive_draft_depth: bool = False,
    ):
        self.req_states = req_states
        self.num_speculative_steps = req_states.num_speculative_steps
        device = req_states.device
        self._copy_stream = torch.cuda.Stream(device)

        self.num_bonus_tokens = num_bonus_tokens
        # Rejection sampling verifies logits in one contiguous chunk; the
        # chunked path indexes by scheduled (untrimmed) offsets and cannot
        # address the compacted layout, so the budget must fit one chunk.
        self._max_total_logits = max_total_logits
        self.query_start_loc = query_start_loc
        # ([num_speculative_steps + 1, max_num_reqs + 1] drafter cost by depth,
        #  [max_num_batched_tokens + 1] verify cost by token count).
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
        self._num_non_draft_tokens = torch.empty_like(query_start_loc[:-1])
        self._cu_num_logits = torch.empty_like(query_start_loc)

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
        # Draft steps each slot's latest proposal ran, snapshotted with every
        # confidence copy: columns at or past it hold no fresh confidence.
        self._draft_depths = np.zeros(max_num_reqs, dtype=np.int32)
        self._stale_draft_depths = [
            np.zeros(max_num_reqs, dtype=np.int32) for _ in range(2)
        ]

        # Draft steps the latest proposal ran. The next budget only ranks those.
        self._last_draft_depth = self.num_speculative_steps
        self.use_adaptive_draft_depth = use_adaptive_draft_depth
        if self.use_adaptive_draft_depth:
            self._last_num_non_draft_tokens = 0
            # Index mapping of the latest recorded proposal.
            self._last_idx_mapping = np.zeros(0, dtype=np.int64)
            # Approximates confidence scores beyond the early draft stop.
            self.undrafted_confidence = UndraftedConfidenceEstimator(
                max_num_reqs, self.num_speculative_steps
            )

    def add_request(self, req_idx: int) -> None:
        self._stale_confidences[self._stale_idx].np[req_idx].fill(1.0)
        self._stale_draft_depths[self._stale_idx][req_idx] = 0
        self._pending_resets.append(req_idx)
        self._confidence_probs[req_idx].fill_(1.0)
        self._draft_depths[req_idx] = 0
        if self.use_adaptive_draft_depth:
            self.undrafted_confidence.reset(req_idx)

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
        context_len = envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN
        for num_tokens in capture_sizes + sorted(tail_sizes):
            for _ in range(_PROFILE_REPLAYS):
                yield {"num_tokens": num_tokens, "context_len": context_len}
            if self.use_adaptive_draft_depth:
                # Prefill-only drafts anchor the shallow end of the depth costs.
                for _ in range(_PROFILE_REPLAYS):
                    yield {
                        "num_tokens": num_tokens,
                        "context_len": context_len,
                        "num_speculative_tokens": 1,
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
        full_depth = [s for s in samples if s.draft_depth == self.num_speculative_steps]
        draft_curve = median_curve(
            (s.num_reqs, s.drafter_ms) for s in full_depth if s.full_cudagraph
        )
        verify_curve = median_curve(
            (s.num_target_tokens, s.forward_ms) for s in full_depth
        )
        if self.use_adaptive_draft_depth:
            shallow_draft_curve = median_curve(
                (s.num_reqs, s.drafter_ms)
                for s in samples
                if s.draft_depth == 1 and s.full_cudagraph
            )
            self.set_cost_curves(draft_curve, verify_curve, shallow_draft_curve)
        else:
            self.set_cost_curves(draft_curve, verify_curve)

    def set_cost_curves(
        self,
        draft_curve: list[tuple[int, float]],
        verify_curve: list[tuple[int, float]],
        shallow_draft_curve: list[tuple[int, float]] | None = None,
    ) -> None:
        draft_curve, verify_curve, shallow_draft_curve = (
            get_tp_group().broadcast_object(
                (draft_curve, verify_curve, shallow_draft_curve), src=0
            )
        )
        if not draft_curve or not verify_curve:
            raise RuntimeError(
                "Adaptive verification could not profile step costs. Pass "
                "`enable_adaptive_verification=false` in the speculative config to "
                "verify a fixed number of drafts instead."
            )
        full_cost, verify_cost = build_cost_tables_from_curves(
            draft_curve,
            verify_curve,
            self.req_states.max_num_reqs,
            self.req_states.max_num_batched_tokens,
            self._cudagraph_limit,
        )
        logger.debug("DSpark cost tables: %s", (full_cost, verify_cost))
        num_depths = self.num_speculative_steps + 1
        # Without a shallow profile every depth is priced at the full depth,
        # which is what runs.
        draft_cost_by_depth = np.broadcast_to(
            full_cost[None, :], (num_depths, len(full_cost))
        )
        if shallow_draft_curve and self.num_speculative_steps > 1:
            shallow_cost, _ = build_cost_tables_from_curves(
                shallow_draft_curve,
                verify_curve,
                self.req_states.max_num_reqs,
                self.req_states.max_num_batched_tokens,
                self._cudagraph_limit,
            )
            full_cost = np.maximum(full_cost, shallow_cost)
            # Every decode step after the prefill step costs about the same, so
            # interpolate between depth 1 (prefill only) and the full depth.
            steps = np.arange(num_depths)
            frac = np.clip((steps - 1) / (self.num_speculative_steps - 1), 0, 1)
            draft_cost_by_depth = shallow_cost[None, :] + frac[:, None] * (
                full_cost - shallow_cost
            )
        self.cost_tables = (draft_cost_by_depth, verify_cost)

    def record_confidences(
        self, confidence_probs: torch.Tensor, input_batch: "InputBatch"
    ) -> None:
        """Publish this step's raw confidences for the ranking kernel and start
        copying them to the CPU, where a later step's budget reads them."""
        num_reqs = input_batch.num_reqs
        ready_idx = self._sync_ready_slot()
        # Last step's copy has landed: budgets read it, this step overwrites the
        # slot they were reading before.
        self._stale_idx, write_idx = ready_idx, self._stale_idx

        self._confidence_probs[input_batch.idx_mapping] = confidence_probs[:num_reqs]
        if self.use_adaptive_draft_depth:
            self._draft_depths[input_batch.idx_mapping_np] = self._last_draft_depth
            self._stale_draft_depths[write_idx][:] = self._draft_depths
            self._last_idx_mapping = input_batch.idx_mapping_np.copy()
        write_slot = self._stale_confidences[write_idx]
        write_slot.gpu.copy_(self._confidence_probs)

        current_stream = torch.cuda.current_stream(self.req_states.device)
        self._copy_stream.wait_stream(current_stream)
        with stream(self._copy_stream, current_stream):
            write_slot.copy_to_cpu()
            self._copy_events[write_idx].record()

    def _sync_ready_slot(self) -> int:
        """Wait for the previous proposal's confidences to reach the CPU."""
        ready_idx = self._stale_idx ^ 1
        with gpu_sync_allowed():
            self._copy_events[ready_idx].synchronize()
        if self._pending_resets:
            self._stale_confidences[ready_idx].np[self._pending_resets] = 1.0
            self._stale_draft_depths[ready_idx][self._pending_resets] = 0
            self._pending_resets.clear()
        return ready_idx

    def _get_confidences(self, stale_idx: int, idx_mapping: np.ndarray) -> np.ndarray:
        """Stale confidences, with steps the stale proposal did not draft estimated."""
        confidences = self._stale_confidences[stale_idx].np[idx_mapping]
        if not self.use_adaptive_draft_depth:
            return confidences
        steps = np.arange(self.num_speculative_steps)
        depths = self._stale_draft_depths[stale_idx][idx_mapping]
        undrafted = steps[None, :] >= depths[:, None]
        if not undrafted.any():
            return confidences
        fill = self.undrafted_confidence.mean(idx_mapping)
        return np.where(undrafted, fill, confidences)

    def _max_logit_drafts(self, num_reqs: int) -> int:
        """Drafts that still fit the rejection sampler's single logits chunk."""
        return max(0, self._max_total_logits - num_reqs * self.num_bonus_tokens)

    def get_num_tokens(
        self,
        num_tokens_per_req: dict[str, int],
        draft_tokens: dict[str, list[int]],
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
        # Slots past the drafted depth hold no draft and can never be admitted.
        scheduled_drafts = np.minimum(scheduled_drafts, self._last_draft_depth)
        slots = np.fromiter(
            map(self.req_states.req_id_to_index.__getitem__, req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        stale_confidences = self._get_confidences(self._stale_idx, slots)
        survival_probability = np.cumprod(stale_confidences.astype(np.float64), axis=1)
        steps = np.arange(self.num_speculative_steps)
        valid = steps[None, :] < scheduled_drafts[:, None]
        scores = np.sort(survival_probability[valid])[::-1]
        num_non_draft_tokens_total = int(num_non_draft_tokens.sum())
        self._last_num_non_draft_tokens = num_non_draft_tokens_total
        max_draft_budget = min(
            int(scheduled_drafts.sum()), self._max_logit_drafts(num_reqs)
        )
        scores = scores[:max_draft_budget]
        draft_cost_by_depth, verify_cost_ms = self.cost_tables
        draft_cost_ms = draft_cost_by_depth[self._last_draft_depth]
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
        self._batch_budget = (
            num_drafts_per_req,
            num_non_draft_tokens_per_req,
            draft_budget,
        )
        return sum(num_non_draft_tokens_per_req.values()) + draft_budget

    def get_draft_depth(
        self,
        idx_mapping_np: np.ndarray,
        max_draft_depth: int,
    ) -> int:
        """Draft steps worth running for the proposal about to be drafted.

        The next step's budget ranks the previous proposal's confidences, which
        already sit in the ready slot, so the depth that budget pays for is known
        before drafting. Steps past it would never be verified. Steps that
        proposal did not draft are filled with the per-step running means.
        """
        # Startup warmup drafts before the cost tables are profiled: draft fully.
        if (
            not self.use_adaptive_draft_depth
            or max_draft_depth <= 1
            or self.cost_tables is None
        ):
            self._last_draft_depth = max_draft_depth
            return max_draft_depth

        num_reqs = len(idx_mapping_np)
        ready_idx = self._sync_ready_slot()

        # Update the estimates for draft steps beyond the early stop.
        self.undrafted_confidence.update(
            self._stale_confidences[ready_idx].np,
            self._stale_draft_depths[ready_idx],
            self._last_idx_mapping,
        )
        # Get the confidence scores up to the max number of draft steps.
        stale_confidences = self._get_confidences(ready_idx, idx_mapping_np)[
            :, :max_draft_depth
        ]

        # Find the optimal depth for expected number of tokens relative to the
        # cost of drafting + verifying them.
        draft_cost_by_depth, verify_cost_ms = self.cost_tables
        survival = np.cumprod(stale_confidences.astype(np.float64), axis=1)
        num_non_draft_tokens = self._last_num_non_draft_tokens or num_reqs
        max_budget = max(
            0,
            min(
                len(verify_cost_ms) - 1 - num_non_draft_tokens,
                self._max_logit_drafts(num_reqs),
            ),
        )
        best_rate, best_depth = -1.0, max_draft_depth
        for depth in range(1, max_draft_depth + 1):
            scores = np.sort(survival[:, :depth], axis=None)[::-1][:max_budget]
            expected_num_tokens = num_reqs + np.concatenate(([0.0], np.cumsum(scores)))
            costs = (
                draft_cost_by_depth[depth, num_reqs]
                + verify_cost_ms[
                    num_non_draft_tokens : num_non_draft_tokens + len(scores) + 1
                ]
            )
            rate = float(np.max(expected_num_tokens / costs))
            if rate > best_rate:
                best_rate, best_depth = rate, depth

        self._last_draft_depth = best_depth
        return best_depth

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
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
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
        if draft_budget == 0:
            capacities.zero_()
        else:
            async_tensor_h2d(scheduled_drafts, out=capacities)
            if draft_budget < int(scheduled_drafts.sum()):
                _assign_draft_token_budget_compiled(
                    self._confidence_probs,
                    idx_mapping,
                    capacities,
                    draft_budget,
                    self.num_speculative_steps,
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
        return (
            self._cu_num_logits[: num_reqs + 1],
            self.query_start_loc,
            draft_budget,
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
    use_adaptive_draft_depth: bool = False,
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
        use_adaptive_draft_depth=use_adaptive_draft_depth,
    )
