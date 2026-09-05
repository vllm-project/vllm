# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Entropy-gated deferred verification for FLy speculative decoding.

FLy defers a high-entropy rejection when the next ``W`` draft tokens would be
accepted natively. ModelRunnerV1 encodes these decisions in a pre-pass by
replacing target argmaxes or uniform samples. ModelRunnerV2 applies them
inside its rejection kernel.
"""

import torch

import vllm.envs as envs
from vllm.triton_utils import tl, triton


def compute_fly_entropy(
    values: torch.Tensor, *, from_logits: bool = False
) -> torch.Tensor:
    """Compute FLy's top-k entropy from processed target probabilities or logits."""

    if values.ndim != 2:
        raise ValueError("FLy expects 2-D target probabilities or logits")
    if values.shape[-1] == 0:
        raise ValueError("FLy requires a non-empty target vocabulary")

    entropy_top_k = envs.VLLM_FLY_ENTROPY_TOP_K
    if entropy_top_k <= 0:
        raise ValueError("VLLM_FLY_ENTROPY_TOP_K must be greater than zero")
    top_k = min(entropy_top_k, values.shape[-1])
    values = values.to(torch.float32)
    top_values = torch.topk(values, k=top_k, dim=-1).values
    if from_logits:
        top_log_probs = top_values - values.logsumexp(dim=-1, keepdim=True)
        top_probs = top_log_probs.exp()
    else:
        top_probs = top_values
        top_log_probs = top_probs.log()

    entropy_terms = torch.where(
        top_probs > 0,
        top_probs * top_log_probs,
        torch.zeros_like(top_probs),
    )
    return -entropy_terms.sum(dim=-1)


# Encode FLy overrides in inputs consumed by the unchanged native kernels.
@triton.jit
def apply_fly_greedy_acceptance_kernel(
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    target_argmax_ptr,
    target_logits_ptr,
    fly_entropy_ptr,
    is_greedy_ptr,
    vocab_size,
    fly_entropy_threshold,
    FLY_WINDOW_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        return

    start_idx = (
        tl.zeros([], dtype=cu_num_draft_tokens_ptr.dtype.element_ty)
        if req_idx == 0
        else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    )
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    for pos in range(num_draft_tokens):
        token_idx = start_idx + pos
        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)
        target_token_id = tl.load(target_argmax_ptr + token_idx)
        can_defer = (
            draft_token_id != target_token_id
            and draft_token_id >= 0
            and draft_token_id < vocab_size
            and pos + FLY_WINDOW_SIZE < num_draft_tokens
            and tl.load(fly_entropy_ptr + token_idx) >= fly_entropy_threshold
        )
        draft_logit = tl.load(
            target_logits_ptr + token_idx * vocab_size + draft_token_id,
            mask=can_defer,
            other=float("-inf"),
        )
        can_defer = can_defer and draft_logit > float("-inf")
        for offset in range(1, FLY_WINDOW_SIZE + 1):
            future_idx = token_idx + offset
            in_bounds = pos + offset < num_draft_tokens
            future_draft_id = tl.load(
                draft_token_ids_ptr + future_idx, mask=in_bounds, other=-1
            )
            future_target_id = tl.load(
                target_argmax_ptr + future_idx, mask=in_bounds, other=-2
            )
            can_defer = can_defer and future_draft_id == future_target_id
        tl.store(target_argmax_ptr + token_idx, draft_token_id, mask=can_defer)


# Setting u=0 forces native p/q acceptance after p>0 and q>0 are verified.
@triton.jit
def apply_fly_random_acceptance_kernel(
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    uniform_probs_ptr,
    fly_entropy_ptr,
    is_greedy_ptr,
    vocab_size,
    fly_entropy_threshold,
    NO_DRAFT_PROBS: tl.constexpr,
    FLY_WINDOW_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    if tl.load(is_greedy_ptr + req_idx):
        return

    start_idx = (
        tl.zeros([], dtype=cu_num_draft_tokens_ptr.dtype.element_ty)
        if req_idx == 0
        else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    )
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    for pos in range(num_draft_tokens):
        token_idx = start_idx + pos
        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)
        valid_draft = draft_token_id >= 0 and draft_token_id < vocab_size
        if NO_DRAFT_PROBS:
            draft_prob = 1.0
        else:
            draft_prob = tl.load(
                draft_probs_ptr + token_idx * vocab_size + draft_token_id,
                mask=valid_draft,
                other=0.0,
            )
        target_prob = tl.load(
            target_probs_ptr + token_idx * vocab_size + draft_token_id,
            mask=valid_draft,
            other=0.0,
        )
        uniform_prob = tl.load(uniform_probs_ptr + token_idx)
        native_accepted = (
            valid_draft and draft_prob > 0 and target_prob / draft_prob >= uniform_prob
        )
        can_defer = (
            not native_accepted
            and valid_draft
            and draft_prob > 0
            and target_prob > 0
            and pos + FLY_WINDOW_SIZE < num_draft_tokens
            and tl.load(fly_entropy_ptr + token_idx) >= fly_entropy_threshold
        )
        for offset in range(1, FLY_WINDOW_SIZE + 1):
            future_idx = token_idx + offset
            in_bounds = pos + offset < num_draft_tokens
            future_draft_id = tl.load(
                draft_token_ids_ptr + future_idx, mask=in_bounds, other=-1
            )
            future_valid = (
                in_bounds and future_draft_id >= 0 and future_draft_id < vocab_size
            )
            if NO_DRAFT_PROBS:
                future_draft_prob = 1.0
            else:
                future_draft_prob = tl.load(
                    draft_probs_ptr + future_idx * vocab_size + future_draft_id,
                    mask=future_valid,
                    other=0.0,
                )
            future_target_prob = tl.load(
                target_probs_ptr + future_idx * vocab_size + future_draft_id,
                mask=future_valid,
                other=0.0,
            )
            future_uniform_prob = tl.load(
                uniform_probs_ptr + future_idx, mask=in_bounds, other=1.0
            )
            future_accepted = (
                future_valid
                and future_draft_prob > 0
                and future_target_prob / future_draft_prob >= future_uniform_prob
            )
            can_defer = can_defer and future_accepted
        tl.store(uniform_probs_ptr + token_idx, 0.0, mask=can_defer)
