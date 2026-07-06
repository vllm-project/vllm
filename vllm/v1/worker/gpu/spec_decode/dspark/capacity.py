# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


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
