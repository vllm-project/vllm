# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptive verification for DSpark speculative decoding."""

from collections import defaultdict
from collections.abc import Iterable
from statistics import median
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.async_utils import stream

logger = init_logger(__name__)

_FIXED_OVERHEAD_MS = 1.0

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState


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


def get_step_cost_profile_cases(
    model_graphs: Iterable[Any],
    draft_graphs: Iterable[Any],
    decode_query_len: int,
) -> list[tuple[int, int]]:
    """Return target-token and request-count shapes for dummy timing runs."""
    target_token_counts = {
        desc.num_tokens
        for desc in model_graphs
        if desc.cg_mode == CUDAGraphMode.FULL and desc.num_active_loras == 0
    }
    cases = {
        (num_tokens, (num_tokens + decode_query_len - 1) // decode_query_len)
        for num_tokens in target_token_counts
    }
    cases.update(
        (desc.num_reqs, desc.num_reqs)
        for desc in draft_graphs
        if desc.cg_mode == CUDAGraphMode.FULL
        and desc.num_active_loras == 0
        and desc.num_reqs is not None
    )
    return sorted(cases)


class AdaptiveVerificationManager:
    def __init__(
        self,
        req_states: "RequestState",
    ):
        self.req_states = req_states
        self.num_speculative_steps = req_states.num_speculative_steps
        device = req_states.device
        self._copy_stream = torch.cuda.Stream(device)

        self.cost_tables: tuple[np.ndarray, np.ndarray] | None = None
        self._batch_budget: dict[str, int] | None = None
        max_num_reqs = req_states.max_num_reqs
        self._confidence_probs = torch.empty(
            (max_num_reqs, self.num_speculative_steps),
            dtype=torch.float32,
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

    def set_step_costs(self, samples: list[tuple[int, int, float, float]]) -> None:
        """Build cost tables from end-to-end dummy step timings."""
        forward_by_case: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
        drafter_by_case: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
        for num_tokens, num_reqs, forward_ms, drafter_ms in samples:
            case = (num_tokens, num_reqs)
            forward_by_case[case].append(forward_ms)
            drafter_by_case[case].append(drafter_ms)

        verify_points: dict[int, float] = {}
        for (num_tokens, _), timings in forward_by_case.items():
            verify_points[num_tokens] = max(
                verify_points.get(num_tokens, 0.0), median(timings)
            )
        draft_points: dict[int, float] = {}
        for (_, num_reqs), timings in drafter_by_case.items():
            draft_points[num_reqs] = max(
                draft_points.get(num_reqs, 0.0), median(timings)
            )

        draft_curve = sorted(draft_points.items())
        verify_curve = sorted(verify_points.items())
        draft_curve, verify_curve = get_tp_group().broadcast_object(
            (draft_curve, verify_curve), src=0
        )
        if not draft_curve or not verify_curve:
            logger.warning_once("DSpark could not collect dummy step timings.")
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
        num_drafts = sum(map(len, draft_tokens.values()))
        num_non_draft_tokens = sum(num_tokens_per_req.values()) - num_drafts
        return num_non_draft_tokens + sum(self._batch_budget.values()), None

    def _compute_budget(
        self,
        num_tokens_per_req: dict[str, int],
        draft_tokens: dict[str, list[int]],
        has_structured_output: bool,
    ) -> dict[str, int]:
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
        stale_probs = self._staged_probs[self._stale_idx].np[slots]
        survival_probability = np.cumprod(stale_probs.astype(np.float64), axis=1)
        steps = np.arange(self.num_speculative_steps)
        valid = steps[None, :] < valid_drafts[:, None]
        candidate_scores = survival_probability[valid]
        candidate_reqs = np.broadcast_to(np.arange(num_reqs)[:, None], valid.shape)[
            valid
        ]
        order = np.argsort(-candidate_scores, kind="stable")
        scores = candidate_scores[order]
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
            _FIXED_OVERHEAD_MS
            + draft_cost_ms[len(req_ids)]
            + verify_cost_ms[
                num_non_draft_tokens_total : num_non_draft_tokens_total
                + max_draft_budget
                + 1
            ]
        )
        draft_budget = int(np.argmax(num_tokens_to_estimated_accepted_tokens / costs))
        capacities = np.bincount(
            candidate_reqs[order[:draft_budget]], minlength=num_reqs
        )
        return {
            req_id: int(capacity)
            for req_id, capacity in zip(req_ids, capacities, strict=True)
        }

    def apply_budget(
        self,
        req_ids: list[str],
        num_draft_tokens_per_req: np.ndarray,
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        capacities_by_req = self._batch_budget
        self._batch_budget = None
        assert capacities_by_req is not None
        capacities = np.fromiter(
            (capacities_by_req[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=len(req_ids),
        )
        assert np.all(capacities <= num_draft_tokens_per_req)
        num_non_draft_tokens = num_scheduled_tokens - num_draft_tokens_per_req
        return capacities, num_non_draft_tokens + capacities


def make_adaptive_verification_manager(
    req_states: "RequestState",
) -> AdaptiveVerificationManager:
    return AdaptiveVerificationManager(req_states)
