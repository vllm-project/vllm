# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch

import vllm.envs as envs
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.async_utils import stream
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
    build_cost_tables_from_curves,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)
_PROFILE_REPLAYS = 5
_RUNTIME_CALIBRATION_SAMPLES = 2


def get_dflash_k_candidates(max_k: int) -> list[int]:
    """Return a compact set of K values whose query lengths are powers of two."""
    candidates = [0]
    k = 1
    while k < max_k:
        candidates.append(k)
        k = 2 * k + 1
    if max_k > 0 and candidates[-1] != max_k:
        candidates.append(max_k)
    return candidates


class DFlashAdaptiveKPolicy:
    def __init__(self, max_k: int, history_weight: float):
        if max_k < 1:
            raise ValueError("max_k must be positive")
        if not 0.0 < history_weight <= 1.0:
            raise ValueError("history_weight must be in (0, 1]")
        self.max_k = max_k
        self.candidates = get_dflash_k_candidates(max_k)
        self.history_weight = history_weight
        self._survival = np.ones(max_k, dtype=np.float64)
        self._draft_cost_ms: np.ndarray | None = None
        self._verify_cost_ms: np.ndarray | None = None
        self._runtime_overhead_ms: dict[int, float] = {}
        self._runtime_samples: dict[int, int] = {}

    def set_cost_tables(
        self, draft_cost_ms: np.ndarray, verify_cost_ms: np.ndarray
    ) -> None:
        if verify_cost_ms.ndim != 2:
            raise ValueError("DFlash verify costs must be indexed by batch and K")
        self._draft_cost_ms = draft_cost_ms
        self._verify_cost_ms = verify_cost_ms

    def reset_history(self) -> None:
        self._survival.fill(1.0)
        self._runtime_overhead_ms.clear()
        self._runtime_samples.clear()

    @staticmethod
    def _batch_bucket(batch_size: int) -> int:
        if batch_size < 1:
            return 0
        return 1 << (batch_size - 1).bit_length()

    def record_runtime(
        self, batch_size: int, k: int, num_sampled: int, elapsed_ms: float
    ) -> None:
        del num_sampled
        if (
            batch_size < 1
            or k == 0
            or elapsed_ms <= 0.0
            or self._verify_cost_ms is None
            or batch_size >= len(self._verify_cost_ms)
        ):
            return
        profiled_verify_ms = self._verify_cost_ms[batch_size, k]
        if not np.isfinite(profiled_verify_ms):
            return
        bucket = self._batch_bucket(batch_size)
        observed_overhead_ms = max(elapsed_ms - profiled_verify_ms, 0.0)
        previous = self._runtime_overhead_ms.get(bucket)
        self._runtime_overhead_ms[bucket] = (
            observed_overhead_ms
            if previous is None
            else (1.0 - self.history_weight) * previous
            + self.history_weight * observed_overhead_ms
        )
        self._runtime_samples[bucket] = self._runtime_samples.get(bucket, 0) + 1

    def select_k(self, batch_size: int) -> int:
        if batch_size < 1:
            return 0
        # At small batch sizes DFlash's full draft is the measured fast path:
        # the target is under-occupied and the longer accepted prefix amortizes
        # the drafter. Startup microbenchmarks do not include that end-to-end
        # under-occupancy benefit, so keep the stable full-K graph there.
        if batch_size < 16:
            return self.max_k
        if self._draft_cost_ms is None or self._verify_cost_ms is None:
            return self.max_k
        if batch_size >= len(self._draft_cost_ms):
            return 0

        bucket = self._batch_bucket(batch_size)
        shared_overhead_ms = 0.0
        if self._runtime_samples.get(bucket, 0) >= _RUNTIME_CALIBRATION_SAMPLES:
            shared_overhead_ms = self._runtime_overhead_ms[bucket]
        expected_tokens = 1.0 + np.concatenate((np.zeros(1), np.cumsum(self._survival)))
        scores = np.full(self.max_k + 1, -np.inf)
        for k in self.candidates:
            draft_cost = 0.0 if k == 0 else self._draft_cost_ms[batch_size]
            cost = draft_cost + self._verify_cost_ms[batch_size, k] + shared_overhead_ms
            scores[k] = batch_size * expected_tokens[k] / max(cost, 1e-6)
        return int(np.argmax(scores))

    def record_outcomes(
        self, num_sampled: np.ndarray, num_draft_tokens: np.ndarray
    ) -> None:
        accepted = np.maximum(num_sampled.astype(np.int64) - 1, 0)
        drafted = num_draft_tokens.astype(np.int64)
        for position in range(self.max_k):
            eligible = drafted > position
            if not eligible.any():
                continue
            observed = np.mean(accepted[eligible] > position)
            self._survival[position] = (1.0 - self.history_weight) * self._survival[
                position
            ] + self.history_weight * observed
        np.minimum.accumulate(self._survival, out=self._survival)


class DFlashAdaptiveKManager(AdaptiveVerificationManager):
    """Graph-aware batch-level draft-length control for DFlash."""

    def __init__(
        self,
        req_states: "RequestState",
        query_start_loc: torch.Tensor,
        num_bonus_tokens: int,
        max_total_logits: int,
        history_weight: float = 0.2,
        decision_interval: int = 2,
    ) -> None:
        super().__init__(
            req_states,
            query_start_loc,
            num_bonus_tokens,
            max_total_logits,
        )
        self.policy = DFlashAdaptiveKPolicy(
            self.num_speculative_steps, history_weight=history_weight
        )
        if decision_interval < 1:
            raise ValueError("decision_interval must be positive")
        self.decision_interval = decision_interval

        device = req_states.device
        self._outcome_buffers = [
            CpuGpuBuffer(
                req_states.max_num_reqs,
                dtype=torch.int32,
                device=device,
            )
            for _ in range(2)
        ]
        self._copy_events = [torch.cuda.Event(blocking=True) for _ in range(2)]
        self._runtime_start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(2)
        ]
        self._runtime_end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(2)
        ]
        self._pending_runtime: list[tuple[int, int] | None] = [None, None]
        self._pending_draft_counts: list[np.ndarray | None] = [None, None]
        self._write_idx = 0
        self._selected_k_by_batch: dict[int, int] = {}
        self._selection_uses_by_batch: dict[int, int] = {}
        self._global_k_cap = self.num_speculative_steps
        self.current_k = self.num_speculative_steps
        self._batch_is_unmodified = False

    def batches_to_profile(self, capture_sizes: list[int]) -> Iterator[dict[str, int]]:
        """Profile real ``(batch, K + 1)`` verification shapes.

        Equal total token counts can have very different hybrid-attention cost
        (for example, 32 requests x 16 tokens versus 128 x 4).  The generic
        one-dimensional profiler intentionally cannot distinguish them.
        """
        self._capture_sizes = set(capture_sizes)
        max_reqs = self.req_states.max_num_reqs
        max_tokens = self.req_states.max_num_batched_tokens
        query_lens = [k + 1 for k in self.policy.candidates]
        batch_sizes = {1, max_reqs}
        for query_len in query_lens:
            batch_sizes.update(
                size // query_len
                for size in capture_sizes
                if size % query_len == 0 and 0 < size // query_len <= max_reqs
            )
        for batch_size in sorted(batch_sizes):
            for query_len in query_lens:
                num_tokens = batch_size * query_len
                if num_tokens > max_tokens:
                    continue
                for _ in range(_PROFILE_REPLAYS):
                    yield {
                        "num_tokens": num_tokens,
                        "uniform_decode_query_len": query_len,
                        "profile_verify": True,
                        "context_len": (
                            envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN
                        ),
                    }

    def set_initial_cost_curves(self, samples: list) -> None:
        grouped: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
        draft_grouped: defaultdict[int, list[float]] = defaultdict(list)
        max_query_len = self.num_speculative_steps + 1
        for sample in samples:
            if sample.num_reqs < 1 or sample.num_target_tokens % sample.num_reqs:
                continue
            query_len = sample.num_target_tokens // sample.num_reqs
            grouped[(query_len, sample.num_reqs)].append(sample.forward_ms)
            if query_len == max_query_len and sample.full_cudagraph:
                draft_grouped[sample.num_reqs].append(sample.drafter_ms)

        curves = {
            query_len: [
                (batch_size, float(np.median(values)))
                for (shape_query_len, batch_size), values in sorted(grouped.items())
                if shape_query_len == query_len
            ]
            for query_len in (k + 1 for k in self.policy.candidates)
        }
        draft_curve = [
            (batch_size, float(np.median(values)))
            for batch_size, values in sorted(draft_grouped.items())
        ]
        draft_curve, curves = get_tp_group().broadcast_object(
            (draft_curve, curves), src=0
        )
        if not draft_curve or any(not curve for curve in curves.values()):
            raise RuntimeError(
                "DFlash adaptive K could not profile every verification shape. "
                "Pass enable_adaptive_verification=false to use a fixed K."
            )

        max_reqs = self.req_states.max_num_reqs
        draft_table, _ = build_cost_tables_from_curves(
            draft_curve,
            [(1, 1.0)],
            max_reqs,
            max_reqs,
            cudagraph_limit=max_reqs,
        )
        verify_table = np.full(
            (max_reqs + 1, self.num_speculative_steps + 1),
            np.inf,
            dtype=np.float64,
        )
        for k in self.policy.candidates:
            query_len = k + 1
            captured_batches = [
                size // query_len
                for size in self._capture_sizes
                if size % query_len == 0
            ]
            capture_limit = min(max(captured_batches, default=0), max_reqs)
            _, costs = build_cost_tables_from_curves(
                [(1, 0.0)],
                curves[query_len],
                max_reqs,
                max_reqs,
                cudagraph_limit=capture_limit,
            )
            verify_table[:, k] = costs

        self.cost_tables = (draft_table, verify_table)
        self.policy.set_cost_tables(draft_table, verify_table)
        for idx in range(len(self._outcome_buffers)):
            self._consume_outcomes(idx)
        self.policy.reset_history()
        self._selected_k_by_batch.clear()
        self._selection_uses_by_batch.clear()
        self._global_k_cap = self.num_speculative_steps
        self.current_k = self.num_speculative_steps

    def get_num_tokens(
        self,
        num_tokens_per_req: dict[str, int],
        draft_tokens: dict[str, list[int]],
    ) -> int:
        """Trim the current target verification batch to the selected K."""
        req_ids = list(num_tokens_per_req)
        scheduled_drafts = np.fromiter(
            (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
            dtype=np.int32,
            count=len(req_ids),
        )
        num_non_draft_tokens = np.fromiter(
            (
                num_tokens_per_req[req_id] - scheduled_drafts[idx]
                for idx, req_id in enumerate(req_ids)
            ),
            dtype=np.int32,
            count=len(req_ids),
        )
        k = self.select_k(int(np.count_nonzero(scheduled_drafts)))
        batch_size = int(np.count_nonzero(scheduled_drafts))
        if k > 0 and batch_size > 0:
            idx = self._write_idx
            # Do not overwrite timing metadata if the double-buffer slot is
            # still carrying a result from two steps ago. This wait is only on
            # slot reuse; the normal policy poll above remains non-blocking.
            self._consume_outcomes(idx)
            self._runtime_start_events[idx].record()
            self._pending_runtime[idx] = (batch_size, k)
        admitted_drafts = np.minimum(scheduled_drafts, k)
        self.batch_query_len = int(
            np.max(num_non_draft_tokens + admitted_drafts, initial=1)
        )
        num_drafts_per_req = {
            req_id: int(num_drafts)
            for req_id, num_drafts in zip(req_ids, admitted_drafts, strict=True)
        }
        num_non_draft_tokens_per_req = {
            req_id: int(num_tokens)
            for req_id, num_tokens in zip(req_ids, num_non_draft_tokens, strict=True)
        }
        draft_budget = int(admitted_drafts.sum())
        self._batch_budget = (
            num_drafts_per_req,
            num_non_draft_tokens_per_req,
            draft_budget,
        )
        self._batch_is_unmodified = draft_budget == int(scheduled_drafts.sum())
        return int(num_non_draft_tokens.sum()) + draft_budget

    def consume_unmodified_batch(self) -> bool:
        if not self._batch_is_unmodified:
            return False
        self._batch_is_unmodified = False
        self._batch_budget = None
        return True

    def _consume_outcomes(self, idx: int, *, wait: bool = True) -> bool:
        draft_counts = self._pending_draft_counts[idx]
        if draft_counts is None:
            return True
        if not wait and not self._copy_events[idx].query():
            return False
        if wait:
            with gpu_sync_allowed():
                self._copy_events[idx].synchronize()
        self.policy.record_outcomes(
            self._outcome_buffers[idx].np[: len(draft_counts)], draft_counts
        )
        runtime = self._pending_runtime[idx]
        if runtime is not None:
            self.policy.record_runtime(
                *runtime,
                int(self._outcome_buffers[idx].np[: len(draft_counts)].sum()),
                self._runtime_start_events[idx].elapsed_time(
                    self._runtime_end_events[idx]
                ),
            )
            self._pending_runtime[idx] = None
        self._pending_draft_counts[idx] = None
        return True

    def select_k(self, batch_size: int) -> int:
        for idx in range(len(self._outcome_buffers)):
            self._consume_outcomes(idx, wait=False)
        if batch_size < 1:
            self.current_k = self._global_k_cap
            return self.current_k
        if self._global_k_cap == 0:
            self.current_k = 0
            return 0
        bucket = DFlashAdaptiveKPolicy._batch_bucket(batch_size)
        previous = self._selected_k_by_batch.get(bucket)
        selection_uses = self._selection_uses_by_batch.get(bucket, 0)
        if previous is not None and selection_uses < self.decision_interval:
            self._selection_uses_by_batch[bucket] = selection_uses + 1
            self.current_k = min(previous, self._global_k_cap)
            return self.current_k
        k = self.policy.select_k(batch_size)
        # Requests shrink within a decode batch.  Re-enabling a longer draft
        # after shortening it creates shape churn and can reuse stale drafter
        # state after K=0.  A graph bucket therefore only moves toward shorter
        # verification shapes for the lifetime of the manager.
        if previous is not None:
            k = min(previous, k)
        k = min(k, self._global_k_cap)
        self._global_k_cap = min(self._global_k_cap, k)
        self.current_k = k
        if previous != k:
            logger.info("DFlash adaptive K: batch_size=%d, K=%d", batch_size, k)
        self._selected_k_by_batch[bucket] = k
        self._selection_uses_by_batch[bucket] = 1
        return k

    def proposal_k(self, batch_size: int) -> int:
        bucket = DFlashAdaptiveKPolicy._batch_bucket(batch_size)
        if bucket not in self._selected_k_by_batch:
            return self.select_k(batch_size)
        self.current_k = min(self._selected_k_by_batch[bucket], self._global_k_cap)
        return self.current_k

    def record_outcomes(
        self,
        num_sampled: torch.Tensor,
        input_batch: "InputBatch",
    ) -> None:
        draft_counts = input_batch.num_draft_tokens_per_req
        if draft_counts is None or not np.any(draft_counts):
            return

        idx = self._write_idx
        self._consume_outcomes(idx)
        draft_counts = draft_counts.copy()
        num_reqs = len(draft_counts)
        slot = self._outcome_buffers[idx]
        slot.gpu[:num_reqs].copy_(num_sampled[:num_reqs])
        self._runtime_end_events[idx].record()

        current_stream = torch.cuda.current_stream(self.req_states.device)
        self._copy_stream.wait_stream(current_stream)
        with stream(self._copy_stream, current_stream):
            slot.copy_to_cpu(num_reqs)
            self._copy_events[idx].record()
        self._pending_draft_counts[idx] = draft_counts
        self._write_idx = (idx + 1) % len(self._outcome_buffers)
