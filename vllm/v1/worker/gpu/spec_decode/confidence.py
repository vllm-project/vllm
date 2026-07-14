# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Confidence-based verification for DSpark speculative decoding."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

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
    meta_ptr,  # [2 * (num_reqs + 1)] int32: new query_start_loc | cu_num_logits
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
            tl.load(meta_ptr + 2 * num_reqs + 1),
        )
        return

    start = tl.load(meta_ptr + req_idx)
    end = tl.load(meta_ptr + req_idx + 1)
    cu_start = tl.load(meta_ptr + num_reqs + 1 + req_idx)
    cu_end = tl.load(meta_ptr + num_reqs + 1 + req_idx + 1)
    tl.store(query_start_loc_ptr + req_idx, start)
    tl.store(cu_num_logits_ptr + req_idx, cu_start)

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


def _get_draft_token_capacities(
    idx_mapping_np: np.ndarray,
    draft_token_capacity_np: np.ndarray,
    valid_draft_tokens_per_req: np.ndarray | None = None,
) -> np.ndarray:
    capacities = draft_token_capacity_np[idx_mapping_np]
    if valid_draft_tokens_per_req is not None:
        capacities = np.minimum(capacities, valid_draft_tokens_per_req)
    return capacities


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


THETA_MARGIN = 1.05


def allocate_draft_token_capacity(
    survival: np.ndarray,
    budget_frac: float = 1.0,
    sps_row: np.ndarray | None = None,
    min_survival: float = 0.0,
) -> np.ndarray:
    """Allocate prefixes globally by survival score (DSpark Algorithm 1).

    An SPS curve chooses the admission count maximizing expected accepted
    tokens per step. ``min_survival`` selects threshold mode instead.
    """
    num_reqs, num_steps = survival.shape
    if min_survival > 0.0:
        return (survival >= min_survival).sum(axis=1).astype(np.int32)

    flat = survival.reshape(-1)
    order = np.argsort(-flat, kind="stable")
    scores = flat[order]
    total = num_reqs * num_steps
    max_admissions = min(int(total * budget_frac) + 1, total)
    num_candidates = min(int((scores > 0.0).sum()), max_admissions)
    capacities = np.zeros(num_reqs, dtype=np.int32)
    if num_candidates == 0:
        return capacities

    if sps_row is None:
        admitted = order[:num_candidates]
    else:
        tau = num_reqs + np.cumsum(scores[:num_candidates])
        theta = tau * sps_row[num_reqs + 1 : num_reqs + num_candidates + 1]
        best = int(np.argmax(theta))
        # Predicted survival and the profiled step-rate table both carry
        # error (tail-censored calibration, interpolated rows); pruning
        # trades real acceptance for predicted step time, so act only when
        # the predicted gain over admitting every candidate clears a margin.
        if theta[best] < THETA_MARGIN * theta[num_candidates - 1]:
            best = num_candidates - 1
        elif num_reqs * sps_row[num_reqs] >= theta[best]:
            return capacities
        admitted = order[: best + 1]

    # Survival is non-increasing along each row, so admission in sorted order
    # always admits prefixes; per-request capacity is the admitted count.
    np.add.at(capacities, admitted // num_steps, 1)
    return capacities


def build_sps_table(
    sps_curve: list[tuple[int, float]],
    max_num_reqs: int,
    max_batch_tokens: int,
) -> np.ndarray:
    """Interpolate SPS breakpoints by verification batch size."""
    xs = np.array([b for b, _ in sps_curve], dtype=np.float64)
    ys = np.array([s for _, s in sps_curve], dtype=np.float64)
    row = np.interp(np.arange(max_batch_tokens + 1), xs, ys)
    return np.broadcast_to(row, (max_num_reqs + 1, row.size)).copy()


def build_additive_sps_table(
    step_time_grid: dict[int, list[tuple[int, float]]],
    max_num_reqs: int,
    max_batch_tokens: int,
) -> tuple[np.ndarray, list[tuple[int, float]], list[tuple[int, float]], float]:
    """Fit ``step_ms(R, B) = draft_ms(R) + verify_ms(B)``."""
    measured_reqs = sorted(step_time_grid)
    draft_ms: list[float] = []
    verify_by_tokens: dict[int, float] = {}
    residuals = []
    for num_reqs in measured_reqs:
        curve = sorted(step_time_grid[num_reqs])
        offsets = [
            step_ms - verify_by_tokens[num_tokens]
            for num_tokens, step_ms in curve
            if num_tokens in verify_by_tokens
        ]
        if draft_ms and not offsets:
            raise ValueError("SPS profiling rows do not share verification sizes")
        draft_cost = 0.0 if not draft_ms else float(np.median(offsets))
        draft_cost = max(draft_ms[-1] if draft_ms else 0.0, draft_cost)
        draft_ms.append(draft_cost)
        for num_tokens, step_ms in curve:
            verify_by_tokens.setdefault(num_tokens, max(step_ms - draft_cost, 1e-6))
            residuals.append(draft_cost + verify_by_tokens[num_tokens] - step_ms)

    measured_tokens = sorted(verify_by_tokens)
    verify_ms = np.array(
        [verify_by_tokens[num_tokens] for num_tokens in measured_tokens]
    )
    draft_ms_np = np.array(draft_ms)
    rmse_ms = float(np.sqrt(np.mean(np.square(residuals))))

    def interpolate(values: np.ndarray, xs: list[int], ys: np.ndarray) -> np.ndarray:
        result = np.interp(values, xs, ys)
        if len(xs) > 1:
            after = values > xs[-1]
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            result[after] = ys[-1] + slope * (values[after] - xs[-1])
        return result

    all_reqs = np.arange(max_num_reqs + 1)
    all_tokens = np.arange(max_batch_tokens + 1)
    draft_table = np.maximum(interpolate(all_reqs, measured_reqs, draft_ms_np), 0.0)
    verify_table = np.maximum(interpolate(all_tokens, measured_tokens, verify_ms), 1e-6)
    sps_table = 1000.0 / (draft_table[:, None] + verify_table[None, :])
    draft_curve = list(zip(measured_reqs, draft_ms))
    verify_curve = list(zip(measured_tokens, verify_ms.tolist()))
    return sps_table, draft_curve, verify_curve, rmse_ms


class ConfidenceManager:
    def __init__(
        self,
        max_num_tokens: int,
        req_states: "RequestState",
        device: torch.device,
        speculative_config: "SpeculativeConfig",
    ):
        self.device = device
        self.req_states = req_states
        self.num_speculative_steps = req_states.num_speculative_steps
        self.draft_token_capacity_np = np.full(
            req_states.max_num_reqs,
            self.num_speculative_steps,
            dtype=np.int32,
        )
        self.copy_stream = torch.cuda.Stream(device)
        self.varlen_spec_decode = False

        self.min_survival_probability = speculative_config.dspark_confidence_threshold
        self.capacity_budget_frac = speculative_config.dspark_budget_frac
        sps_curve = speculative_config.dspark_sps_curve
        self.wants_auto_sps_curve = sps_curve == "auto"
        max_batch_tokens = req_states.max_num_reqs * (1 + self.num_speculative_steps)
        self.sps_table_np: np.ndarray | None = None
        if isinstance(sps_curve, list):
            self.sps_table_np = build_sps_table(
                sps_curve, req_states.max_num_reqs, max_batch_tokens
            )
        # With "auto", allocation admits everything until the post-capture
        # profiling installs the measured table via set_sps_profile.
        self.online_sts: DSparkOnlineSTS | None = None
        if speculative_config.dspark_online_sts:
            self.online_sts = DSparkOnlineSTS(self.num_speculative_steps)

        # Two D2H slots preserve the prior confidences needed by STS.
        shape = (2, req_states.max_num_reqs)
        self._confidence_logits_cpu = torch.empty(
            (*shape, self.num_speculative_steps),
            dtype=torch.float32,
            pin_memory=True,
        )
        self._num_sampled_cpu = torch.empty(shape, dtype=torch.int32, pin_memory=True)
        self._confidence_logits_np = self._confidence_logits_cpu.numpy()
        self._num_sampled_np = self._num_sampled_cpu.numpy()
        self._num_verified_np = np.empty(shape, dtype=np.int64)
        self._req_ids: list[list[str]] = [[], []]
        self._num_bonus = [0, 0]
        self._copy_events = [torch.cuda.Event(blocking=True) for _ in range(2)]
        self._staged_idx: int | None = None
        self._last_idx: int | None = None
        self._next_idx = 0
        self.forced_capacity: int | None = None

    def add_request(self, req_idx: int) -> None:
        self.draft_token_capacity_np[req_idx] = self.num_speculative_steps

    def set_sps_profile(
        self, step_time_grid: dict[int, list[tuple[int, float]]]
    ) -> tuple[list[tuple[int, float]], list[tuple[int, float]], float]:
        max_batch_tokens = self.req_states.max_num_reqs * (
            1 + self.num_speculative_steps
        )
        self.sps_table_np, draft_curve, verify_curve, rmse_ms = (
            build_additive_sps_table(
                step_time_grid, self.req_states.max_num_reqs, max_batch_tokens
            )
        )
        return draft_curve, verify_curve, rmse_ms

    def profile_sps_curve(
        self,
        model_runner: "GPUModelRunner",
        worker_execute_model: Callable[[Any], Any],
        worker_sample_tokens: Callable[[Any], Any],
        req_ids: list[str],
        prompt_len: int,
        decode_query_len: int,
        num_spec_steps: int,
        num_kv_cache_groups: int,
        warmup_iters: int = 3,
        timed_chunks: int = 5,
    ) -> None:
        """Profile additive draft and verification step costs."""
        import time

        from vllm import SamplingParams
        from vllm.distributed.parallel_state import get_tp_group
        from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

        max_reqs = min(
            model_runner.max_num_reqs,
            model_runner.max_num_tokens // decode_query_len,
            len(req_ids),
        )
        req_counts = []
        count = 1
        while count < max_reqs:
            req_counts.append(count)
            count *= 2
        if count == max_reqs:
            req_counts.append(max_reqs)
        draft_counts = sorted(
            {0, min(1, num_spec_steps), min(3, num_spec_steps), num_spec_steps}
        )
        if model_runner.sampler is not None:
            params = SamplingParams()
            for req_id in req_ids:
                model_runner.sampler.add_request(
                    model_runner.req_states.req_id_to_index[req_id],
                    prompt_len,
                    params,
                )
            model_runner.sampler.apply_staged_writes()

        def time_steps(step: Callable[[], None]) -> float:
            for _ in range(warmup_iters):
                step()
            chunk_ms = []
            for _ in range(timed_chunks):
                torch.accelerator.synchronize()
                start = time.perf_counter()
                for _ in range(3):
                    step()
                torch.accelerator.synchronize()
                chunk_ms.append((time.perf_counter() - start) * 1000.0 / 3)
            return float(np.median(chunk_ms))

        def make_step(num_reqs: int) -> Callable[[], None]:
            cached = CachedRequestData.make_empty()
            cached.req_ids = list(req_ids[:num_reqs])
            cached.num_computed_tokens = [prompt_len] * num_reqs
            cached.num_output_tokens = [1] * num_reqs
            cached.new_block_ids = [None] * num_reqs
            output = SchedulerOutput.make_empty()
            output.scheduled_cached_reqs = cached
            output.num_scheduled_tokens = {
                req_id: decode_query_len for req_id in cached.req_ids
            }
            if num_spec_steps > 0:
                output.scheduled_spec_decode_tokens = {
                    req_id: [0] * num_spec_steps for req_id in cached.req_ids
                }
            output.total_num_scheduled_tokens = decode_query_len * num_reqs
            output.num_common_prefix_blocks = [0] * num_kv_cache_groups

            def step() -> None:
                worker_execute_model(output)
                worker_sample_tokens(None)

            return step

        try:
            step_ms = []
            for num_reqs in req_counts:
                step = make_step(num_reqs)
                for capacity in draft_counts:
                    self.forced_capacity = capacity
                    step_ms.append(time_steps(step))
        finally:
            self.forced_capacity = None

        timings = torch.tensor(step_ms, dtype=torch.float64, device=self.device)
        tp_group = get_tp_group()
        if tp_group.world_size > 1:
            tp_group.broadcast(timings, src=0)
        step_ms = timings.cpu().tolist()

        ms_iter = iter(step_ms)
        step_time_grid = {
            num_reqs: [
                (num_reqs * (1 + capacity), next(ms_iter)) for capacity in draft_counts
            ]
            for num_reqs in req_counts
        }
        draft_curve, verify_curve, rmse_ms = self.set_sps_profile(step_time_grid)
        if self.online_sts is not None:
            self.online_sts.reset()
        logger.info(
            "DSpark auto-profiled draft cost (requests -> ms): %s",
            [(r, round(ms, 3)) for r, ms in draft_curve],
        )
        logger.info(
            "DSpark auto-profiled verification cost (tokens -> ms): %s",
            [(b, round(ms, 3)) for b, ms in verify_curve],
        )
        logger.info(
            "DSpark additive SPS fit RMSE: %.3f ms",
            rmse_ms,
        )

    def stage_confidences(
        self,
        confidence_logits: torch.Tensor,
        num_sampled: torch.Tensor,
        input_batch: "InputBatch",
    ) -> None:
        """Stage confidence logits and sampler outcomes for host allocation."""
        num_reqs = input_batch.num_reqs
        num_bonus = self._get_num_bonus_tokens(input_batch)
        num_verified_np = (np.diff(input_batch.cu_num_logits_np) - num_bonus).astype(
            np.int64
        )
        if self._staged_idx is not None:
            self._update_staged_confidences(self._staged_idx)

        idx = self._next_idx
        self._next_idx ^= 1
        self._num_verified_np[idx, :num_reqs] = num_verified_np
        self._req_ids[idx] = list(input_batch.req_ids)
        self._num_bonus[idx] = num_bonus
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with stream(self.copy_stream, current_stream):
            self._confidence_logits_cpu[idx, :num_reqs].copy_(
                confidence_logits[:num_reqs], non_blocking=True
            )
            self._num_sampled_cpu[idx, :num_reqs].copy_(
                num_sampled[:num_reqs], non_blocking=True
            )
            num_sampled.record_stream(self.copy_stream)
            self._copy_events[idx].record()
        self._staged_idx = idx

    def wait_for_staged_copy(self) -> None:
        """Protect the persistent confidence buffer before its next replay."""
        if self._staged_idx is not None:
            torch.cuda.current_stream(self.device).wait_event(
                self._copy_events[self._staged_idx]
            )

    def get_num_tokens(
        self,
        num_tokens_per_req: dict[str, int],
        draft_tokens: dict[str, list[int]],
        has_structured_output_requests: bool = False,
    ) -> int:
        raise NotImplementedError

    def _update_staged_confidences(self, idx: int) -> None:
        """Consume one staged slot and update request capacities."""
        with gpu_sync_allowed():
            self._copy_events[idx].synchronize()

        req_ids = self._req_ids[idx]
        num_reqs = len(req_ids)
        confidence_logits_np = self._confidence_logits_np[idx, :num_reqs]
        accepted_np = (
            self._num_sampled_np[idx, :num_reqs].astype(np.int64) - self._num_bonus[idx]
        )
        prev_idx = self._last_idx
        if self.online_sts is not None and prev_idx is not None:
            # Outcomes in this slot verify confidences from the prior slot.
            prev_ids = self._req_ids[prev_idx]
            prev_row = {req_id: i for i, req_id in enumerate(prev_ids)}
            rows = np.array(
                [prev_row.get(req_id, -1) for req_id in req_ids],
                dtype=np.int64,
            )
            sel = rows >= 0
            if sel.any():
                self.online_sts.update(
                    self._confidence_logits_np[prev_idx, : len(prev_ids)][rows[sel]],
                    accepted_np[sel],
                    self._num_verified_np[idx, :num_reqs][sel],
                )
        self._last_idx = idx

        temperatures: np.ndarray | float = 1.0
        if self.online_sts is not None:
            temperatures = self.online_sts.temperatures
        survival = compute_prefix_survival(confidence_logits_np, temperatures)
        sps_row = None
        if self.sps_table_np is not None:
            sps_row = self.sps_table_np[num_reqs]
        capacities = allocate_draft_token_capacity(
            survival,
            self.capacity_budget_frac,
            sps_row,
            self.min_survival_probability,
        )
        if self.forced_capacity is not None:
            np.minimum(capacities, self.forced_capacity, out=capacities)

        req_id_to_index = self.req_states.req_id_to_index
        slots = np.array(
            [req_id_to_index.get(req_id, -1) for req_id in req_ids],
            dtype=np.int64,
        )
        live = slots >= 0
        self.draft_token_capacity_np[slots[live]] = capacities[live]

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

    def _get_draft_token_capacities(
        self,
        input_batch: "InputBatch",
        draft_tokens: dict[str, list[int]],
    ) -> np.ndarray:
        assert input_batch.num_draft_tokens_per_req is not None
        valid = input_batch.num_draft_tokens_per_req
        if input_batch.has_structured_output_reqs:
            valid = _count_valid_draft_tokens(
                [draft_tokens.get(req_id, ()) for req_id in input_batch.req_ids],
                input_batch.num_reqs,
            )
        return _get_draft_token_capacities(
            input_batch.idx_mapping_np,
            self.draft_token_capacity_np,
            valid,
        )

    def trim_batch(
        self,
        input_batch: "InputBatch",
        draft_tokens: dict[str, list[int]],
    ) -> None:
        raise NotImplementedError


class VarlenConfidenceManager(ConfidenceManager):
    def __init__(
        self,
        max_num_tokens: int,
        req_states: "RequestState",
        device: torch.device,
        speculative_config: "SpeculativeConfig",
    ):
        super().__init__(max_num_tokens, req_states, device, speculative_config)
        self.varlen_spec_decode = True
        # Persistent inputs and outputs for the fused batch rewrite.
        max_num_reqs = req_states.max_num_reqs
        max_num_logits = max_num_reqs * (1 + self.num_speculative_steps)
        self._meta_np = np.empty(2 * (max_num_reqs + 1), dtype=np.int32)
        self._meta = torch.empty(
            2 * (max_num_reqs + 1), dtype=torch.int32, device=device
        )
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
        self._planned_draft_tokens_per_req: np.ndarray | None = None

    def get_num_tokens(
        self,
        num_tokens_per_req: dict[str, int],
        draft_tokens: dict[str, list[int]],
        has_structured_output_requests: bool = False,
    ) -> int:
        num_reqs = len(num_tokens_per_req)
        req_ids = sorted(
            num_tokens_per_req,
            key=num_tokens_per_req.get,  # type: ignore[arg-type]
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
            # Otherwise the scheduler only sees -1 placeholders (real draft ids
            # stay on the GPU), so every scheduled slot counts.
            valid_num_draft_tokens_per_req = num_draft_tokens_per_req
        idx_mapping_np = np.fromiter(
            (self.req_states.req_id_to_index[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        self._planned_draft_tokens_per_req = _get_draft_token_capacities(
            idx_mapping_np,
            self.draft_token_capacity_np,
            valid_num_draft_tokens_per_req,
        )
        total_num_draft_tokens = int(self._planned_draft_tokens_per_req.sum())
        return int(
            num_scheduled_tokens.sum()
            - num_draft_tokens_per_req.sum()
            + total_num_draft_tokens
        )

    def warmup(self, input_buffers: "InputBuffers") -> None:
        # JIT-compile the fused rewrite kernel.
        query_len = 1 + self.num_speculative_steps
        self._meta_np[:4] = [0, query_len, 0, query_len]
        async_copy_to_gpu(self._meta_np[:4], out=self._meta[:4])
        idx_mapping = torch.zeros(1, dtype=torch.int32, device=self.device)
        _rewrite_compact_batch_kernel[(2,)](
            self._meta,
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
        num_scheduled_tokens: np.ndarray,
        num_draft_tokens_per_req: np.ndarray,
        num_bonus_tokens: int,
    ) -> None:
        num_reqs = input_batch.num_reqs
        num_tokens = int(num_scheduled_tokens.sum())
        input_batch.num_scheduled_tokens = num_scheduled_tokens
        input_batch.num_tokens = num_tokens
        input_batch.num_draft_tokens_per_req = num_draft_tokens_per_req
        input_batch.num_draft_tokens = int(num_draft_tokens_per_req.sum())

        # Upload query and logits offsets together.
        meta = self._meta_np[: 2 * (num_reqs + 1)]
        query_start_loc_np = meta[: num_reqs + 1]
        cu_num_logits_np = meta[num_reqs + 1 :]
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])
        cu_num_logits_np[0] = 0
        np.cumsum(num_draft_tokens_per_req + num_bonus_tokens, out=cu_num_logits_np[1:])
        num_logits = int(cu_num_logits_np[-1])
        async_copy_to_gpu(meta, out=self._meta[: meta.shape[0]])

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
            self._meta,
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
            return

        num_bonus_tokens = self._get_num_bonus_tokens(input_batch)
        num_draft_tokens_per_req = self._planned_draft_tokens_per_req
        self._planned_draft_tokens_per_req = None
        if num_draft_tokens_per_req is None:
            num_draft_tokens_per_req = self._get_draft_token_capacities(
                input_batch, draft_tokens
            )
        if int(num_draft_tokens_per_req.sum()) != input_batch.num_draft_tokens:
            num_scheduled_tokens = (
                input_batch.num_scheduled_tokens
                - input_batch.num_draft_tokens_per_req
                + num_draft_tokens_per_req
            )
            self._rewrite_compact_batch(
                input_batch,
                num_scheduled_tokens,
                num_draft_tokens_per_req,
                num_bonus_tokens,
            )


class MaskedConfidenceManager(ConfidenceManager):
    def __init__(
        self,
        max_num_tokens: int,
        req_states: "RequestState",
        device: torch.device,
        speculative_config: "SpeculativeConfig",
    ):
        super().__init__(max_num_tokens, req_states, device, speculative_config)
        self.forward_skip_mask_np = np.zeros(max_num_tokens, dtype=np.bool_)
        self._mask_edges_np = np.empty(max_num_tokens + 1, dtype=np.int32)

    def _prepare_forward_skip_mask(
        self,
        input_batch: "InputBatch",
        num_bonus_tokens: int,
        draft_tokens: dict[str, list[int]],
    ) -> bool:
        assert input_batch.num_draft_tokens_per_req is not None
        num_padded = input_batch.num_tokens_after_padding
        mask = self.forward_skip_mask_np[:num_padded]
        mask[:] = False
        capacities = self._get_draft_token_capacities(input_batch, draft_tokens)
        # Mark [start + bonus + kept, start + bonus + drafts) per request:
        # +1 at each prune start, -1 past each prune end, then a cumsum.
        starts = input_batch.query_start_loc_np[: input_batch.num_reqs]
        prune_start = starts + num_bonus_tokens + capacities
        prune_end = starts + num_bonus_tokens + input_batch.num_draft_tokens_per_req
        edges = self._mask_edges_np[: num_padded + 1]
        edges[:] = 0
        np.add.at(edges, prune_start, 1)
        np.add.at(edges, prune_end, -1)
        np.cumsum(edges[:num_padded], out=edges[:num_padded])
        np.greater(edges[:num_padded], 0, out=mask)
        has_forward_skip_mask = bool(mask[: input_batch.num_tokens].any())
        if has_forward_skip_mask:
            async_copy_to_gpu(
                mask[: input_batch.num_tokens],
                out=input_batch.is_padding[: input_batch.num_tokens],
            )
        return has_forward_skip_mask

    def trim_batch(
        self,
        input_batch: "InputBatch",
        draft_tokens: dict[str, list[int]],
    ) -> None:
        if (
            input_batch.num_draft_tokens == 0
            or input_batch.num_draft_tokens_per_req is None
        ):
            return

        num_bonus_tokens = self._get_num_bonus_tokens(input_batch)
        if self._prepare_forward_skip_mask(input_batch, num_bonus_tokens, draft_tokens):
            input_batch.input_ids[: input_batch.num_tokens].masked_fill_(
                input_batch.is_padding[: input_batch.num_tokens],
                0,
            )


def make_confidence_manager(
    mode: str,
    attn_cg_support: "AttentionCGSupportInfo",
    max_num_tokens: int,
    req_states: "RequestState",
    device: torch.device,
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
        return VarlenConfidenceManager(
            max_num_tokens,
            req_states,
            device,
            speculative_config,
        )
    if mode == "mask":
        return MaskedConfidenceManager(
            max_num_tokens,
            req_states,
            device,
            speculative_config,
        )
    raise ValueError(f"Unknown confidence-based verification mode: {mode}")
