# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import (
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    get_num_sampled_and_rejected,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
from vllm.v1.worker.gpu.spec_decode.decompaction import (
    SamplerDecompactionBuffers,
    SamplerDecompactionMetadata,
    prepare_sampler_decompaction_from_counts,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState


@dataclass
class _SamplerDecompactionState:
    num_scheduled_tokens_before_capacity: np.ndarray
    scheduled_draft_tokens_per_req: np.ndarray
    pruned_draft_tokens_per_req: np.ndarray
    num_bonus_tokens: int


class VerificationCapacityRequestStates:
    def __init__(self, req_states: "RequestState"):
        self.max_num_reqs = req_states.max_num_reqs
        self.num_speculative_steps = req_states.num_speculative_steps
        self.req_id_to_index: dict[str, int] = {}

        self.draft_token_capacity_np = np.full(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=np.int32,
        )
        self.num_computed_tokens = req_states.num_computed_tokens.gpu
        self.prefill_len = req_states.prefill_len.gpu
        self.next_prefill_tokens = req_states.next_prefill_tokens
        self.all_token_ids = req_states.all_token_ids.gpu
        self.last_sampled_tokens = req_states.last_sampled_tokens
        self.draft_tokens = req_states.draft_tokens

    def add_request(self, req_id: str, req_idx: int) -> None:
        self.req_id_to_index[req_id] = req_idx
        self.draft_token_capacity_np[req_idx] = self.num_speculative_steps

    def remove_request(self, req_id: str) -> None:
        req_idx = self.req_id_to_index.pop(req_id, None)
        if req_idx is not None:
            self.draft_token_capacity_np[req_idx] = self.num_speculative_steps


def get_draft_token_capacity(
    speculator: object | None,
    num_reqs: int,
) -> torch.Tensor | None:
    draft_token_capacity = getattr(speculator, "draft_token_capacity", None)
    if draft_token_capacity is None:
        return None
    if not getattr(speculator, "use_draft_token_capacity", True):
        return None
    return draft_token_capacity[:num_reqs]


def get_scheduled_draft_token_counts(
    req_ids: list[str] | tuple[str, ...],
    draft_tokens: dict[str, list[int]],
) -> np.ndarray:
    return np.fromiter(
        (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
        dtype=np.int32,
        count=len(req_ids),
    )


def apply_draft_token_capacity(
    num_scheduled_tokens: np.ndarray,
    scheduled_draft_tokens_per_req: np.ndarray,
    idx_mapping_np: np.ndarray,
    draft_token_capacity_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_draft_tokens_per_req = np.minimum(
        scheduled_draft_tokens_per_req,
        draft_token_capacity_np[idx_mapping_np],
    )
    pruned_draft_tokens_per_req = (
        scheduled_draft_tokens_per_req - num_draft_tokens_per_req
    )
    return (
        num_scheduled_tokens - pruned_draft_tokens_per_req,
        num_draft_tokens_per_req,
        pruned_draft_tokens_per_req,
    )


def get_effective_scheduled_token_counts(
    scheduler_output: SchedulerOutput,
    req_id_to_index: dict[str, int],
    draft_token_capacity_np: np.ndarray,
) -> tuple[int, int]:
    num_tokens_per_req = scheduler_output.num_scheduled_tokens
    req_ids = tuple(num_tokens_per_req)
    num_reqs = len(req_ids)
    num_scheduled_tokens = np.fromiter(
        num_tokens_per_req.values(), dtype=np.int32, count=num_reqs
    )
    draft_tokens = scheduler_output.scheduled_spec_decode_tokens
    if draft_tokens:
        scheduled_draft_tokens_per_req = get_scheduled_draft_token_counts(
            req_ids, draft_tokens
        )
        idx_mapping_np = np.fromiter(
            map(req_id_to_index.get, req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_scheduled_tokens, _, _ = apply_draft_token_capacity(
            num_scheduled_tokens,
            scheduled_draft_tokens_per_req,
            idx_mapping_np,
            draft_token_capacity_np,
        )
    return int(num_scheduled_tokens.sum()), int(num_scheduled_tokens.max())


class CapacityBasedVerificationManager:
    def __init__(
        self,
        max_num_tokens: int,
        req_states: "RequestState",
        device: torch.device,
    ):
        self.device = device
        self.req_states = VerificationCapacityRequestStates(req_states)
        self.copy_stream = torch.cuda.Stream(device)
        self.copy_event = torch.Event()
        self.sampler_decompaction_buffers = SamplerDecompactionBuffers.make(
            max_num_tokens,
            device,
        )

        self.req_ids: list[str] = []
        self.idx_mapping_np: np.ndarray | None = None
        self.copied_draft_token_capacity_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0
        self.copy_event_pending = False
        self._sampler_decompaction_state: _SamplerDecompactionState | None = None
        self.sampler_decompaction: SamplerDecompactionMetadata | None = None

    def clear(self) -> None:
        self.req_ids = []
        self.idx_mapping_np = None
        self.copied_draft_token_capacity_np = None
        self.num_draft_tokens = 0
        self.copy_event_pending = False
        self._sampler_decompaction_state = None
        self.sampler_decompaction = None

    def add_request(self, req_id: str, req_idx: int) -> None:
        self.req_states.add_request(req_id, req_idx)

    def remove_request(self, req_id: str) -> None:
        self.req_states.remove_request(req_id)

    def _trim_counts(
        self,
        input_batch: "InputBatch",
        num_bonus_tokens: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert input_batch.num_draft_tokens_per_req is not None
        num_scheduled_tokens_before_capacity = input_batch.num_scheduled_tokens.copy()
        scheduled_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
        (
            num_scheduled_tokens,
            num_draft_tokens_per_req,
            pruned_draft_tokens_per_req,
        ) = apply_draft_token_capacity(
            input_batch.num_scheduled_tokens,
            scheduled_draft_tokens_per_req,
            input_batch.idx_mapping_np,
            self.req_states.draft_token_capacity_np,
        )
        self._sampler_decompaction_state = _SamplerDecompactionState(
            num_scheduled_tokens_before_capacity=num_scheduled_tokens_before_capacity,
            scheduled_draft_tokens_per_req=scheduled_draft_tokens_per_req,
            pruned_draft_tokens_per_req=pruned_draft_tokens_per_req,
            num_bonus_tokens=num_bonus_tokens,
        )
        return (
            num_scheduled_tokens,
            num_draft_tokens_per_req,
            pruned_draft_tokens_per_req,
        )

    def _sync_copy(self) -> None:
        if self.copy_event_pending:
            self.copy_event.synchronize()
            self.copy_event_pending = False

    def _copy_ready(self) -> bool:
        if not self.copy_event_pending:
            return True
        if not self.copy_event.query():
            return False
        self.copy_event_pending = False
        return True

    def _get_draft_token_capacities(self) -> np.ndarray | None:
        if self.copied_draft_token_capacity_np is None:
            return None
        return np.clip(self.copied_draft_token_capacity_np, 0, self.num_draft_tokens)

    def set_draft_token_capacities(
        self,
        req_ids: list[str],
        idx_mapping_np: np.ndarray,
        draft_token_capacity: torch.Tensor,
    ) -> None:
        self.req_ids = list(req_ids)
        self.idx_mapping_np = idx_mapping_np
        self.num_draft_tokens = self.req_states.num_speculative_steps
        self.copied_draft_token_capacity_np = None
        self.copy_event_pending = False

        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            self.copied_draft_token_capacity_np = async_copy_to_np(draft_token_capacity)
            draft_token_capacity.record_stream(self.copy_stream)
            self.copy_event.record()
            self.copy_event_pending = True

    def update_draft_token_capacities(
        self,
        input_batch: "InputBatch",
        draft_token_capacity: torch.Tensor | None,
    ) -> None:
        self.try_update_draft_token_capacities()
        if draft_token_capacity is None:
            self.clear()
            return
        self.set_draft_token_capacities(
            input_batch.req_ids,
            input_batch.idx_mapping_np,
            draft_token_capacity,
        )

    def try_update_draft_token_capacities(self) -> bool:
        if self.copied_draft_token_capacity_np is None:
            return False
        if not self._copy_ready():
            return False
        draft_token_capacities = self._get_draft_token_capacities()
        assert draft_token_capacities is not None
        assert self.idx_mapping_np is not None
        active = np.isin(self.req_ids, tuple(self.req_states.req_id_to_index))
        self.req_states.draft_token_capacity_np[self.idx_mapping_np[active]] = (
            draft_token_capacities[active]
        )
        return True

    def get_effective_scheduled_token_counts(
        self,
        scheduler_output: SchedulerOutput,
    ) -> tuple[int, int]:
        return get_effective_scheduled_token_counts(
            scheduler_output,
            self.req_states.req_id_to_index,
            self.req_states.draft_token_capacity_np,
        )

    def _prepare_sampler_decompaction(
        self,
        input_batch: "InputBatch",
    ) -> SamplerDecompactionMetadata | None:
        state = self._sampler_decompaction_state
        if state is None:
            return None
        if not np.any(state.pruned_draft_tokens_per_req > 0):
            return None
        return prepare_sampler_decompaction_from_counts(
            input_batch.cu_num_logits,
            input_batch.query_start_loc,
            input_batch.idx_mapping,
            input_batch.positions,
            self.req_states.last_sampled_tokens,
            self.req_states.draft_tokens,
            state.num_scheduled_tokens_before_capacity,
            state.scheduled_draft_tokens_per_req,
            state.num_bonus_tokens,
            input_batch.num_reqs,
            input_batch.num_reqs_after_padding,
            self.req_states.max_num_reqs,
            self.device,
            self.sampler_decompaction_buffers,
            state.num_bonus_tokens,
        )

    def _rewrite_compact_batch(
        self,
        input_batch: "InputBatch",
        num_scheduled_tokens: np.ndarray,
        num_draft_tokens_per_req: np.ndarray,
        num_bonus_tokens: int,
    ) -> None:
        input_batch.num_scheduled_tokens = num_scheduled_tokens
        input_batch.num_tokens = int(num_scheduled_tokens.sum())
        input_batch.num_draft_tokens_per_req = num_draft_tokens_per_req
        input_batch.num_draft_tokens = int(num_draft_tokens_per_req.sum())

        num_logits = num_draft_tokens_per_req + num_bonus_tokens
        input_batch.cu_num_logits_np = np.empty(
            input_batch.num_reqs + 1, dtype=np.int32
        )
        input_batch.cu_num_logits_np[0] = 0
        np.cumsum(num_logits, out=input_batch.cu_num_logits_np[1:])
        input_batch.cu_num_logits = async_copy_to_gpu(
            input_batch.cu_num_logits_np,
            device=self.device,
        )
        (
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
        ) = expand_idx_mapping(
            input_batch.idx_mapping,
            int(input_batch.cu_num_logits_np[-1]),
            input_batch.cu_num_logits,
            max(1, int(num_logits.max())),
        )

        query_start_loc_np = np.empty(self.req_states.max_num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(
            num_scheduled_tokens,
            out=query_start_loc_np[1 : input_batch.num_reqs + 1],
        )
        query_start_loc_np[input_batch.num_reqs + 1 :] = input_batch.num_tokens
        input_batch.query_start_loc_np = query_start_loc_np[
            : input_batch.num_reqs_after_padding + 1
        ]
        async_copy_to_gpu(
            input_batch.query_start_loc_np,
            out=input_batch.query_start_loc,
        )

        if np.any(input_batch.is_prefilling_np):
            prepare_prefill_inputs(
                input_batch.input_ids,
                self.req_states.next_prefill_tokens,
                input_batch.idx_mapping,
                input_batch.query_start_loc,
                self.req_states.all_token_ids,
                self.req_states.prefill_len,
                self.req_states.num_computed_tokens,
            )
        prepare_pos_seq_lens(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            self.req_states.num_computed_tokens,
            input_batch.positions,
            input_batch.seq_lens,
        )
        input_batch.logits_indices = combine_sampled_and_draft_tokens(
            input_batch.input_ids,
            input_batch.idx_mapping,
            self.req_states.last_sampled_tokens,
            input_batch.query_start_loc,
            input_batch.seq_lens,
            self.req_states.prefill_len,
            self.req_states.draft_tokens,
            input_batch.cu_num_logits,
            int(input_batch.cu_num_logits_np[-1]),
            num_bonus_tokens,
        )

        seq_lens_cpu_upper_bound_np = np.zeros(
            input_batch.num_reqs_after_padding,
            dtype=np.int32,
        )
        np.add(
            input_batch.num_computed_tokens_np,
            num_scheduled_tokens,
            out=seq_lens_cpu_upper_bound_np[: input_batch.num_reqs],
        )
        input_batch.seq_lens_cpu_upper_bound = torch.from_numpy(
            seq_lens_cpu_upper_bound_np
        )

    def trim_batch(self, input_batch: "InputBatch") -> "InputBatch":
        if (
            input_batch.num_draft_tokens == 0
            or input_batch.num_draft_tokens_per_req is None
        ):
            self._sampler_decompaction_state = None
            self.sampler_decompaction = None
            return input_batch

        num_logits = np.diff(input_batch.cu_num_logits_np)
        num_bonus_tokens_per_req = num_logits - input_batch.num_draft_tokens_per_req
        num_bonus_tokens = int(num_bonus_tokens_per_req[0])
        assert np.all(num_bonus_tokens_per_req == num_bonus_tokens)
        (
            num_scheduled_tokens,
            num_draft_tokens_per_req,
            pruned_draft_tokens_per_req,
        ) = self._trim_counts(input_batch, num_bonus_tokens)

        if np.any(pruned_draft_tokens_per_req > 0):
            self._rewrite_compact_batch(
                input_batch,
                num_scheduled_tokens,
                num_draft_tokens_per_req,
                num_bonus_tokens,
            )
        self.sampler_decompaction = self._prepare_sampler_decompaction(input_batch)
        return input_batch

    def get_num_rejected_for_next_step(
        self,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: "InputBatch",
    ) -> torch.Tensor:
        if self.sampler_decompaction is None:
            return num_rejected
        _, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.req_states.prefill_len,
        )
        return num_rejected

    def get_postprocess_query_start_loc(
        self,
        input_batch: "InputBatch",
    ) -> torch.Tensor:
        if self.sampler_decompaction is None:
            return input_batch.query_start_loc
        return self.sampler_decompaction.query_start_loc


@triton.jit
def _compute_prefix_survival_probabilities_kernel(
    confidence_logits_ptr,
    survival_probs_ptr,
    NUM_SPECULATIVE_STEPS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    survival_prob = tl.full((), 1.0, tl.float32)
    for step in tl.static_range(0, NUM_SPECULATIVE_STEPS):
        confidence_logit = tl.load(
            confidence_logits_ptr + req_idx * NUM_SPECULATIVE_STEPS + step
        ).to(tl.float32)
        confidence_prob = 1.0 / (1.0 + tl.exp(-confidence_logit))
        survival_prob *= confidence_prob
        tl.store(
            survival_probs_ptr + req_idx * NUM_SPECULATIVE_STEPS + step,
            survival_prob,
        )


@triton.jit
def _allocate_draft_token_capacity_kernel(
    survival_probs_ptr,
    capacity_ptr,
    num_reqs,
    min_survival_probability,
    max_admissions,
    REQ_BLOCK: tl.constexpr,
    NUM_SPECULATIVE_STEPS: tl.constexpr,
    MAX_ADMISSIONS: tl.constexpr,
    USE_BUDGET: tl.constexpr,
):
    offsets = tl.arange(0, REQ_BLOCK)
    active = offsets < num_reqs
    threshold = min_survival_probability

    if USE_BUDGET:
        lengths = tl.full((REQ_BLOCK,), 0, tl.int32)
        kth_score = tl.full((), 0.0, tl.float32)
        for admission_idx in tl.range(0, MAX_ADMISSIONS):
            has_next = (
                active
                & (admission_idx < max_admissions)
                & (lengths < NUM_SPECULATIVE_STEPS)
            )
            next_scores = tl.load(
                survival_probs_ptr + offsets * NUM_SPECULATIVE_STEPS + lengths,
                mask=has_next,
                other=-1.0,
            )
            best_score, best_idx = tl.max(next_scores, axis=0, return_indices=True)
            admit = best_score >= 0.0
            lengths += tl.where(admit & (offsets == best_idx), 1, 0)
            kth_score = tl.where(admit, best_score, kth_score)
        threshold = kth_score

    capacities = tl.full((REQ_BLOCK,), 0, tl.int32)
    for step in tl.static_range(0, NUM_SPECULATIVE_STEPS):
        scores = tl.load(
            survival_probs_ptr + offsets * NUM_SPECULATIVE_STEPS + step,
            mask=active,
            other=-1.0,
        )
        capacities += tl.where(scores >= threshold, 1, 0)

    tl.store(capacity_ptr + offsets, capacities, mask=active)


def compute_draft_token_capacity_from_confidence(
    confidence_logits: torch.Tensor,
    draft_token_capacity: torch.Tensor,
    min_survival_probability: float,
    num_reqs: int,
    num_speculative_steps: int,
    survival_probs: torch.Tensor | None = None,
    budget_frac: float = 1.0,
) -> None:
    if num_reqs == 0 or num_speculative_steps == 0:
        return
    if survival_probs is None:
        survival_probs = torch.empty_like(confidence_logits)
    _compute_prefix_survival_probabilities_kernel[(num_reqs,)](
        confidence_logits,
        survival_probs,
        NUM_SPECULATIVE_STEPS=num_speculative_steps,
    )
    max_admissions = num_reqs * num_speculative_steps
    use_budget = min_survival_probability <= 0.0
    if use_budget:
        max_admissions = min(
            int(max_admissions * budget_frac) + 1,
            max_admissions,
        )
        if max_admissions == num_reqs * num_speculative_steps:
            draft_token_capacity[:num_reqs].fill_(num_speculative_steps)
            return

    req_block = triton.next_power_of_2(max(num_reqs, 1))
    _allocate_draft_token_capacity_kernel[(1,)](
        survival_probs,
        draft_token_capacity,
        num_reqs,
        min_survival_probability,
        max_admissions,
        REQ_BLOCK=req_block,
        NUM_SPECULATIVE_STEPS=num_speculative_steps,
        MAX_ADMISSIONS=num_reqs * num_speculative_steps,
        USE_BUDGET=use_budget,
    )
