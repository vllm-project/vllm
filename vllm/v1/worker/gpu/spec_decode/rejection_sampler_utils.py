# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import zip_inputs
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    VllmTritonJitKernel,
    kernel_launcher,
    triton_warmup_inputs,
    triton_scalar_specialization_rep,
)
from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_block_argmax, tl_rand32


@triton.jit
def _compute_max_and_sumexp(logits):
    max = tl.max(logits, axis=0)
    sumexp = tl.where(
        max > float("-inf"),
        tl.sum(tl.exp(logits - max)),
        0.0,
    )
    return max, sumexp


@triton.jit
def _compute_global_logsumexp(
    local_max_ptr,
    local_max_stride,
    local_sumexp_ptr,
    local_sumexp_stride,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
    blocks_mask = blocks < vocab_num_blocks
    maxes = tl.load(
        local_max_ptr + logit_idx * local_max_stride + blocks,
        mask=blocks_mask,
        other=float("-inf"),
    )
    sumexps = tl.load(
        local_sumexp_ptr + logit_idx * local_sumexp_stride + blocks,
        mask=blocks_mask,
        other=0.0,
    )
    global_max = tl.max(maxes, axis=0)
    global_lse = global_max + tl.log(tl.sum(sumexps * tl.exp(maxes - global_max)))
    return global_lse


@triton.jit
def _compute_global_residual_mass(
    local_residual_mass_ptr,
    local_residual_mass_stride,
    prefix_joint_ratio,
    target_logits_ptr,
    target_logits_stride,
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    draft_token,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    if HAS_DRAFT_LOGITS:
        blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
        mask = blocks < vocab_num_blocks
        partials = tl.load(
            local_residual_mass_ptr + logit_idx * local_residual_mass_stride + blocks,
            mask=mask,
            other=0.0,
        )
        return tl.sum(partials, axis=0)
    else:
        # One-hot draft. M_s is a point mass at this draft token
        # so the residual mass reduces to the closed form:
        #   p * (1 - M_b(draft_token)).
        target_lse = _compute_global_logsumexp(
            target_local_max_ptr,
            target_local_max_stride,
            target_local_sumexp_ptr,
            target_local_sumexp_stride,
            logit_idx,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        )
        target_logit = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + draft_token,
        ).to(tl.float32)
        m_b = tl.exp(target_logit - target_lse)
        return prefix_joint_ratio * (1.0 - m_b)


@triton.jit
def _compute_global_target_argmax(
    target_local_max_ptr,
    target_local_max_stride,
    target_local_argmax_ptr,
    target_local_argmax_stride,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
    blocks_mask = blocks < vocab_num_blocks
    local_max = tl.load(
        target_local_max_ptr + logit_idx * target_local_max_stride + blocks,
        mask=blocks_mask,
        other=float("-inf"),
    )
    # See _insert_resampled_kernel: NaN breaks tl.argmax index bounds.
    local_max = tl.where(local_max != local_max, float("-inf"), local_max)
    max_block_idx = tl.argmax(local_max, axis=0)
    return tl.load(
        target_local_argmax_ptr + logit_idx * target_local_argmax_stride + max_block_idx
    ).to(tl.int64)


@triton.jit
def _compute_global_logprobs_and_logsumexp(
    token,
    mask,
    logit_idx,
    req_state_idx,
    draft_step,
    temp,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    target_logit = tl.load(
        target_logits_ptr + logit_idx * target_logits_stride + token,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    target_lse = _compute_global_logsumexp(
        target_local_max_ptr,
        target_local_max_stride,
        target_local_sumexp_ptr,
        target_local_sumexp_stride,
        logit_idx,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS,
    )
    target_log_prob = target_logit - target_lse
    if HAS_DRAFT_LOGITS:
        # draft_logits is stored pre-temperature, so apply scale first.
        draft_logit = (
            tl.load(
                draft_logits_ptr
                + req_state_idx * draft_logits_stride_0
                + draft_step * draft_logits_stride_1
                + token,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
            / temp
        )
        draft_lse = _compute_global_logsumexp(
            draft_local_max_ptr,
            draft_local_max_stride,
            draft_local_sumexp_ptr,
            draft_local_sumexp_stride,
            logit_idx,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        )
        draft_log_prob = draft_logit - draft_lse
    else:
        # One-hot draft: q(token) = 1, log_q = 0.
        draft_log_prob = 0.0
        draft_lse = 0.0
    return target_log_prob, draft_log_prob, target_lse, draft_lse


@triton.jit
def _compute_local_logits_stats_kernel(
    # [num_logits, num_blocks]
    target_local_argmax_ptr,
    target_local_argmax_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_logits, num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    expanded_local_pos_ptr,
    # [max_num_reqs]
    temp_ptr,
    vocab_size,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    logit_idx = tl.program_id(0).to(tl.int64)
    draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)

    if draft_step_idx >= num_speculative_steps:
        # Bonus token. Max/argmax and summed exponentials are not needed.
        return

    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx).to(tl.int64)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    block_idx = tl.program_id(1)
    block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < vocab_size

    if temp == 0.0:
        # Greedy sampling. Only the target max/argmax are needed.
        target_logits = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        value, idx = tl.max(target_logits, axis=0, return_indices=True)
        token_id = block_idx * BLOCK_SIZE + idx
        tl.store(
            target_local_argmax_ptr
            + logit_idx * target_local_argmax_stride
            + block_idx,
            token_id,
        )
        tl.store(
            target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
            value,
        )
    else:
        # Get local target max and summed exponentials.
        target_logits = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        target_max, target_sumexp = _compute_max_and_sumexp(target_logits)
        tl.store(
            target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
            target_max,
        )
        tl.store(
            target_local_sumexp_ptr
            + logit_idx * target_local_sumexp_stride
            + block_idx,
            target_sumexp,
        )
        if HAS_DRAFT_LOGITS:
            # Get local draft max and summed exponentials. draft_logits is
            # stored pre-temperature, so apply scale first.
            draft_logits = (
                tl.load(
                    draft_logits_ptr
                    + req_state_idx * draft_logits_stride_0
                    + draft_step_idx * draft_logits_stride_1
                    + block_offsets,
                    mask=mask,
                    other=float("-inf"),
                ).to(tl.float32)
                / temp
            )
            draft_max, draft_sumexp = _compute_max_and_sumexp(draft_logits)
            tl.store(
                draft_local_max_ptr + logit_idx * draft_local_max_stride + block_idx,
                draft_max,
            )
            tl.store(
                draft_local_sumexp_ptr
                + logit_idx * draft_local_sumexp_stride
                + block_idx,
                draft_sumexp,
            )


@triton.jit
def _compute_cumulative_log_p_kernel(
    # [num_logits]
    cumulative_log_p_ptr,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [num_logits]
    draft_sampled_ptr,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_logits, num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if temp == 0.0:
        return

    log_p = tl.zeros((), tl.float32)
    for step in range(num_draft_tokens):
        logit_idx = start_idx + step
        draft_token = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
        # -1 placeholder tokens can never be accepted. Skip their reductions
        # and carry the last valid cumulative value.
        if draft_token >= 0:
            target_logprob, draft_logprob, _, _ = (
                _compute_global_logprobs_and_logsumexp(
                    draft_token,
                    True,  # mask
                    logit_idx,
                    req_state_idx,
                    step,
                    temp,
                    target_logits_ptr,
                    target_logits_stride,
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_sumexp_ptr,
                    target_local_sumexp_stride,
                    draft_logits_ptr,
                    draft_logits_stride_0,
                    draft_logits_stride_1,
                    draft_local_max_ptr,
                    draft_local_max_stride,
                    draft_local_sumexp_ptr,
                    draft_local_sumexp_stride,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                    HAS_DRAFT_LOGITS,
                )
            )
            log_p = tl.minimum(log_p + (target_logprob - draft_logprob), 0.0)
        tl.store(cumulative_log_p_ptr + logit_idx, log_p)


@triton.jit
def _compute_local_residual_mass_kernel(
    # [num_logits, num_blocks]
    local_residual_mass_ptr,
    local_residual_mass_stride,
    # [num_logits]
    cumulative_log_p_ptr,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_logits, num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_logits]
    draft_sampled_ptr,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    expanded_local_pos_ptr,
    # [max_num_reqs]
    temp_ptr,
    vocab_size,
    num_speculative_steps,
    vocab_num_blocks,
    BLOCK_SIZE: tl.constexpr,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    logit_idx = tl.program_id(0).to(tl.int64)
    draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)
    if draft_step_idx == 0 or draft_step_idx >= num_speculative_steps:
        # The acceptance threshold, h, looks one position ahead and sums
        # over: max(p_i * M_b(x|x_{<i}) - M_s(x|x_{<i}), 0). Tokens at the
        # first and last (bonus) positions aren't needed for this computation.
        return

    if tl.load(draft_sampled_ptr + logit_idx + 1) < 0:
        # -1 placeholder token. The rejection kernel treats the preceding token
        # as the end of the block, so this position's residual mass is unused.
        return

    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx).to(tl.int64)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if temp == 0.0:
        return

    block_idx = tl.program_id(1)
    block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < vocab_size
    target_log_probs, draft_log_probs, _, _ = _compute_global_logprobs_and_logsumexp(
        block_offsets,
        mask,
        logit_idx,
        req_state_idx,
        draft_step_idx,
        temp,
        target_logits_ptr,
        target_logits_stride,
        target_local_max_ptr,
        target_local_max_stride,
        target_local_sumexp_ptr,
        target_local_sumexp_stride,
        draft_logits_ptr,
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_local_max_ptr,
        draft_local_max_stride,
        draft_local_sumexp_ptr,
        draft_local_sumexp_stride,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS,
        True,  # HAS_DRAFT_LOGITS
    )

    # Compute the residual mass: max(p_i * M_b(x|x_{<i}) - M_s(x|x_{<i}), 0)
    p = tl.exp(tl.load(cumulative_log_p_ptr + logit_idx - 1).to(tl.float32))
    m_b = tl.exp(target_log_probs)
    m_s = tl.exp(draft_log_probs)
    partial = tl.sum(tl.maximum(p * m_b - m_s, 0.0), axis=0)
    tl.store(
        local_residual_mass_ptr + logit_idx * local_residual_mass_stride + block_idx,
        partial,
    )


@triton.jit
def _rejection_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    rejected_steps_ptr,
    # [num_reqs]
    target_rejected_logsumexp_ptr,
    # [num_reqs]
    draft_rejected_logsumexp_ptr,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_logits, num_blocks]
    target_local_argmax_ptr,
    target_local_argmax_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [num_logits]
    draft_sampled_ptr,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_logits, num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    # [max_num_reqs]
    seed_ptr,
    # [num_logits]
    pos_ptr,
    # [num_speculative_steps]
    synthetic_conditional_rates_ptr,
    # [num_logits]
    cumulative_log_p_ptr,
    # [num_logits, num_blocks]
    local_residual_mass_ptr,
    local_residual_mass_stride,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,
    USE_BLOCK_VERIFICATION: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1
    seed = tl.load(seed_ptr + req_state_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_greedy = temp == 0.0

    accepted_length = tl.zeros((), tl.int64)
    target_lse = 0.0
    draft_lse = 0.0
    verifying = True
    for i in range(num_draft_tokens):
        logit_idx = start_idx + i
        draft_sampled = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
        # -1 is used for placeholder draft token ids that should be rejected.
        is_valid_draft = draft_sampled >= 0
        # Avoid possible OOB ptr access.
        draft_sampled = tl.maximum(0, draft_sampled)
        if not is_greedy:
            # A -1 placeholder ends verification. Greedy is excluded because it
            # stores the target argmax upon first rejection, so it rejects the
            # placeholder via `accepted` instead.
            verifying &= is_valid_draft

        if verifying:
            pos = tl.load(pos_ptr + logit_idx)
            u = tl_rand32(seed, pos, includes_zero=False)
            if is_greedy:
                # Greedy sampling. Accept IFF draft matches target argmax.
                # NOTE: Target argmax is stored directly so that resampling
                # can be skipped upon rejection.
                target_argmax = _compute_global_target_argmax(
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_argmax_ptr,
                    target_local_argmax_stride,
                    logit_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                )
                if SYNTHETIC_MODE:
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted = u < rate
                else:
                    accepted = target_argmax == draft_sampled
                accepted &= is_valid_draft
                verifying = accepted
                accepted_length += accepted
                tl.store(
                    sampled_ptr + req_idx * sampled_stride + i,
                    draft_sampled if accepted else target_argmax,
                )
            elif USE_BLOCK_VERIFICATION:
                # Block verification (Sun et al., 2024): https://arxiv.org/abs/2403.10444
                prefix_joint_ratio = tl.exp(
                    tl.load(cumulative_log_p_ptr + logit_idx).to(tl.float32)
                )
                next_draft_token = tl.load(
                    draft_sampled_ptr + logit_idx + 2,
                    mask=i < num_draft_tokens - 1,
                    other=-1,
                ).to(tl.int64)
                if next_draft_token >= 0:
                    residual_mass = _compute_global_residual_mass(
                        local_residual_mass_ptr,
                        local_residual_mass_stride,
                        prefix_joint_ratio,
                        target_logits_ptr,
                        target_logits_stride,
                        target_local_max_ptr,
                        target_local_max_stride,
                        target_local_sumexp_ptr,
                        target_local_sumexp_stride,
                        next_draft_token,
                        logit_idx + 1,
                        vocab_num_blocks,
                        PADDED_VOCAB_NUM_BLOCKS,
                        HAS_DRAFT_LOGITS,
                    )
                    denom = residual_mass + 1.0 - prefix_joint_ratio
                    h = tl.where(denom > 0.0, residual_mass / denom, 1.0)
                else:
                    h = prefix_joint_ratio
                accepted_length = tl.where(u <= h, i + 1, accepted_length)
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            else:
                # Speculative decoding (Leviathan et al., 2023): https://arxiv.org/abs/2211.17192
                target_logprob, draft_logprob, target_lse, draft_lse = (
                    _compute_global_logprobs_and_logsumexp(
                        draft_sampled,
                        True,  # mask
                        logit_idx,
                        req_state_idx,
                        i,
                        temp,
                        target_logits_ptr,
                        target_logits_stride,
                        target_local_max_ptr,
                        target_local_max_stride,
                        target_local_sumexp_ptr,
                        target_local_sumexp_stride,
                        draft_logits_ptr,
                        draft_logits_stride_0,
                        draft_logits_stride_1,
                        draft_local_max_ptr,
                        draft_local_max_stride,
                        draft_local_sumexp_ptr,
                        draft_local_sumexp_stride,
                        vocab_num_blocks,
                        PADDED_VOCAB_NUM_BLOCKS,
                        HAS_DRAFT_LOGITS,
                    )
                )
                if SYNTHETIC_MODE:
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted = u < rate
                else:
                    # Probability ratio test: p(x) > u * q(x)
                    # Equivalent log form: log_p(x) > log(u) + log_q(x)
                    accepted = target_logprob > tl.log(u) + draft_logprob
                verifying = accepted
                accepted_length += accepted
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)

    tl.store(rejected_steps_ptr + req_idx, accepted_length)
    if USE_BLOCK_VERIFICATION and not is_greedy and accepted_length < num_draft_tokens:
        # Compute the target and draft log exponential sums for the
        # rejected token.
        rejected_idx = start_idx + accepted_length
        target_lse = _compute_global_logsumexp(
            target_local_max_ptr,
            target_local_max_stride,
            target_local_sumexp_ptr,
            target_local_sumexp_stride,
            rejected_idx,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        )
        if HAS_DRAFT_LOGITS:
            draft_lse = _compute_global_logsumexp(
                draft_local_max_ptr,
                draft_local_max_stride,
                draft_local_sumexp_ptr,
                draft_local_sumexp_stride,
                rejected_idx,
                vocab_num_blocks,
                PADDED_VOCAB_NUM_BLOCKS,
            )
    tl.store(target_rejected_logsumexp_ptr + req_idx, target_lse)
    tl.store(draft_rejected_logsumexp_ptr + req_idx, draft_lse)


@triton.jit
def _resample_kernel(
    # [num_reqs, num_blocks]
    resampled_local_argmax_ptr,
    resampled_local_argmax_stride,
    # [num_reqs, num_blocks]
    resampled_local_max_ptr,
    resampled_local_max_stride,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_reqs]
    target_rejected_logsumexp_ptr,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_reqs]
    draft_rejected_logsumexp_ptr,
    # [num_reqs]
    rejected_step_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    draft_sampled_ptr,
    # [max_num_reqs]
    temp_ptr,
    # [max_num_reqs]
    seed_ptr,
    # [num_logits]
    pos_ptr,
    # [num_logits]
    cumulative_log_p_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    USE_FP64: tl.constexpr,
    USE_BLOCK_VERIFICATION: tl.constexpr,
):
    req_idx = tl.program_id(0)
    resample_idx = tl.load(rejected_step_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + resample_idx
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx).to(tl.int64)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1
    if temp == 0.0 and not is_bonus:
        # Greedy + non-bonus token. No resampling needed because
        # the target argmax is already in the sampled tensor.
        return

    rejected_draft_token = tl.load(
        draft_sampled_ptr + resample_token_idx + 1,
        mask=not is_bonus,
        other=0,
    )
    is_valid_rejected_draft = rejected_draft_token >= 0

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    target_logits = tl.load(
        target_logits_ptr + resample_token_idx * target_logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)

    # Compute the residual logits to resample the rejected token from.
    if is_bonus or not is_valid_rejected_draft:
        # Bonus token (no rejections) or -1 placeholder token. In either case,
        # directly use the target logits.
        residual_logits = target_logits
    elif HAS_DRAFT_LOGITS:
        # draft_logits is stored pre-temperature, so apply scale first.
        draft_logits = (
            tl.load(
                draft_logits_ptr
                + req_state_idx * draft_logits_stride_0
                + resample_idx * draft_logits_stride_1
                + block,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
            / temp
        )
        target_lse = tl.load(target_rejected_logsumexp_ptr + req_idx)
        draft_lse = tl.load(draft_rejected_logsumexp_ptr + req_idx)
        target_log_probs = target_logits - target_lse
        if USE_BLOCK_VERIFICATION:
            # Block residual is:
            #   max(p_tau * M_b(x) - M_s(x), 0) / Z.
            # Scale the target logprobs by log(p_tau). p_0 = 1, so skip
            # shifting when nothing was accepted (tau == 0).
            log_p_tau = 0.0
            if resample_idx > 0:
                log_p_tau = tl.load(cumulative_log_p_ptr + resample_token_idx - 1).to(
                    tl.float32
                )
            target_log_probs += log_p_tau
        draft_log_probs = draft_logits - draft_lse
        # Compute the residual:
        #   r(x) = max(p(x) - q(x), 0)
        # Gumbel sampling needs logits, so we compute it in log space:
        #   log(r(x)) = log(max(exp(log_p(x)) - exp(log_q(x)), 0))
        # The more numerically stable form is:
        #   log(max(exp(a) - exp(b), 0)) = a + log(max(1 - exp(b - a), 0))
        ratio = tl.exp(draft_log_probs - target_log_probs)
        residual_logits = tl.where(
            ratio < 1.0,
            target_log_probs + tldevice.log1p(-ratio),
            float("-inf"),
        ).to(tl.float32)
    else:
        # One-hot draft. The residual is just the target distribution with
        # the rejected draft token probability zeroed out.
        # NOTE: During block verification, the residual becomes:
        #   0                   if x == rejected_draft_token
        #   p_tau * M_b(x) / Z  otherwise
        # Therefore p_tau is a constant that cancels under normalization,
        # and does not need to be applied.
        residual_logits = tl.where(
            block != rejected_draft_token,
            target_logits,
            float("-inf"),
        ).to(tl.float32)

    # Resample the rejected/bonus token.
    value, idx = gumbel_block_argmax(
        residual_logits,
        block,
        mask,
        resample_token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seed_ptr,
        pos_ptr,
        None,  # logits_cache_ptr
        0,  # logits_cache_stride_0
        0,  # logits_cache_stride_1
        None,  # logits_cache_col_ptr
        vocab_size,
        IS_DRAFTING=False,
        APPLY_TEMPERATURE=False,
        USE_FP64=USE_FP64,
    )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(
        resampled_local_argmax_ptr
        + req_idx * resampled_local_argmax_stride
        + block_idx,
        token_id,
    )
    tl.store(
        resampled_local_max_ptr + req_idx * resampled_local_max_stride + block_idx,
        value,
    )


@triton.jit
def _insert_resampled_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs, num_blocks]
    resampled_local_argmax_ptr,
    resampled_local_argmax_stride,
    # [num_reqs, num_blocks]
    resampled_local_max_ptr,
    resampled_local_max_stride,
    resample_num_blocks,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    expanded_idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    PADDED_RESAMPLE_NUM_BLOCKS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + num_sampled
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx)

    # Increment the number of sampled tokens.
    tl.store(num_sampled_ptr + req_idx, num_sampled + 1)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1
    if temp == 0.0 and not is_bonus:
        # Greedy + non-bonus token. The target argmax is already
        # in the sampled tensor.
        return

    # Insert the resampled token.
    block = tl.arange(0, PADDED_RESAMPLE_NUM_BLOCKS)
    mask = block < resample_num_blocks
    resampled_local_max = tl.load(
        resampled_local_max_ptr + req_idx * resampled_local_max_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    # NaN max values (from NaN target logits) make tl.argmax return an
    # out-of-range block index (into the padded region), causing an OOB read
    # of resampled_local_argmax. Map NaN to -inf so argmax stays in range.
    resampled_local_max = tl.where(
        resampled_local_max != resampled_local_max,
        float("-inf"),
        resampled_local_max,
    )
    resampled_max_block_idx = tl.argmax(resampled_local_max, axis=0)
    resampled = tl.load(
        resampled_local_argmax_ptr
        + req_idx * resampled_local_argmax_stride
        + resampled_max_block_idx,
    )
    tl.store(
        sampled_ptr + req_idx * sampled_stride + num_sampled,
        resampled,
    )


class ComputeLocalLogitsStatsKernel(
    VllmTritonJitKernel["ComputeLocalLogitsStatsKernel.CompileKey"]
):
    kernel = staticmethod(_compute_local_logits_stats_kernel)

    @dataclass(frozen=True)
    class CompileKey:
        target_dtype: torch.dtype
        draft_dtype: torch.dtype | None
        target_logits_stride: int
        draft_logits_stride_0: int
        draft_logits_stride_1: int
        vocab_size: int
        num_speculative_steps: int
        block_size: int
        has_draft_logits: bool

    def dispatch(
        self,
        *,
        target_dtype: torch.dtype,
        draft_dtype: torch.dtype | None,
        target_logits_stride: int,
        draft_logits_stride_0: int,
        draft_logits_stride_1: int,
        vocab_size: int,
        num_speculative_steps: int,
        block_size: int,
        has_draft_logits: bool,
    ) -> CompileKey:
        return self.CompileKey(
            target_dtype=target_dtype,
            draft_dtype=draft_dtype,
            target_logits_stride=triton_scalar_specialization_rep(
                target_logits_stride
            ),
            draft_logits_stride_0=triton_scalar_specialization_rep(
                draft_logits_stride_0
            ),
            draft_logits_stride_1=triton_scalar_specialization_rep(
                draft_logits_stride_1
            ),
            vocab_size=triton_scalar_specialization_rep(vocab_size),
            num_speculative_steps=triton_scalar_specialization_rep(
                num_speculative_steps
            ),
            block_size=block_size,
            has_draft_logits=has_draft_logits,
        )

    def get_warmup_keys(
        self, *, model_dtype: torch.dtype, vocab_size: int, num_speculative_steps: int
    ) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(
            zip_inputs(
                dict(
                    target_dtype=model_dtype,
                    draft_dtype=None,
                    draft_logits_stride_0=0,
                    draft_logits_stride_1=0,
                    has_draft_logits=False,
                ),
                dict(
                    target_dtype=torch.float32,
                    draft_dtype=None,
                    draft_logits_stride_0=0,
                    draft_logits_stride_1=0,
                    has_draft_logits=False,
                ),
                dict(
                    target_dtype=model_dtype,
                    draft_dtype=model_dtype,
                    draft_logits_stride_0=num_speculative_steps * vocab_size,
                    draft_logits_stride_1=vocab_size,
                    has_draft_logits=True,
                ),
                dict(
                    target_dtype=torch.float32,
                    draft_dtype=model_dtype,
                    draft_logits_stride_0=num_speculative_steps * vocab_size,
                    draft_logits_stride_1=vocab_size,
                    has_draft_logits=True,
                ),
            ),
            target_logits_stride=vocab_size,
            vocab_size=vocab_size,
            num_speculative_steps=num_speculative_steps,
            block_size=8192,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        int32_ptr = TritonWarmupTensor(torch.int32)
        int64_ptr = TritonWarmupTensor(torch.int64)
        float32_ptr = TritonWarmupTensor(torch.float32)
        target_ptr = TritonWarmupTensor(compile_key.target_dtype)
        draft_ptr = (
            None
            if compile_key.draft_dtype is None
            else TritonWarmupTensor(compile_key.draft_dtype)
        )
        return triton_warmup_inputs(
            self.kernel,
            int64_ptr,
            1,
            float32_ptr,
            1,
            float32_ptr,
            1,
            float32_ptr,
            1,
            float32_ptr,
            1,
            target_ptr,
            compile_key.target_logits_stride,
            draft_ptr,
            compile_key.draft_logits_stride_0,
            compile_key.draft_logits_stride_1,
            int32_ptr,
            int32_ptr,
            float32_ptr,
            compile_key.vocab_size,
            compile_key.num_speculative_steps,
            BLOCK_SIZE=compile_key.block_size,
            HAS_DRAFT_LOGITS=compile_key.has_draft_logits,
            grid=(1, 1),
        )

    @kernel_launcher
    def __call__(
        self, grid: tuple[int, ...], *args: Any, **kwargs: Any
    ) -> tuple[tuple[int, ...], dict[str, Any]]:
        return grid, {**dict(zip(self._kernel_arg_names, args)), **kwargs}


_COMPUTE_LOCAL_LOGITS_STATS_KERNEL = ComputeLocalLogitsStatsKernel()


class ComputeCumulativeLogPKernel(
    VllmTritonJitKernel["ComputeCumulativeLogPKernel.CompileKey"]
):
    kernel = staticmethod(_compute_cumulative_log_p_kernel)

    @dataclass(frozen=True)
    class CompileKey:
        target_dtype: torch.dtype
        draft_dtype: torch.dtype | None
        target_logits_stride: int
        draft_logits_stride_0: int
        draft_logits_stride_1: int
        vocab_num_blocks: int
        padded_vocab_num_blocks: int
        has_draft_logits: bool

    def dispatch(self, **compile_key_fields: Any) -> CompileKey:
        return self.CompileKey(**compile_key_fields)

    def get_warmup_keys(
        self, *, model_dtype: torch.dtype, vocab_size: int, num_speculative_steps: int
    ) -> list[CompileKey]:
        vocab_num_blocks = triton.cdiv(vocab_size, 8192)
        return self._trace_dispatch(self.dispatch)(
            zip_inputs(
                dict(
                    target_dtype=model_dtype,
                    draft_dtype=None,
                    draft_logits_stride_0=0,
                    draft_logits_stride_1=0,
                    has_draft_logits=False,
                ),
                dict(
                    target_dtype=torch.float32,
                    draft_dtype=None,
                    draft_logits_stride_0=0,
                    draft_logits_stride_1=0,
                    has_draft_logits=False,
                ),
                dict(
                    target_dtype=model_dtype,
                    draft_dtype=model_dtype,
                    draft_logits_stride_0=num_speculative_steps * vocab_size,
                    draft_logits_stride_1=vocab_size,
                    has_draft_logits=True,
                ),
                dict(
                    target_dtype=torch.float32,
                    draft_dtype=model_dtype,
                    draft_logits_stride_0=num_speculative_steps * vocab_size,
                    draft_logits_stride_1=vocab_size,
                    has_draft_logits=True,
                ),
            ),
            target_logits_stride=vocab_size,
            vocab_num_blocks=vocab_num_blocks,
            padded_vocab_num_blocks=triton.next_power_of_2(vocab_num_blocks),
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        int32_ptr = TritonWarmupTensor(torch.int32)
        int64_ptr = TritonWarmupTensor(torch.int64)
        float32_ptr = TritonWarmupTensor(torch.float32)
        draft_ptr = (
            None
            if compile_key.draft_dtype is None
            else TritonWarmupTensor(compile_key.draft_dtype)
        )
        return triton_warmup_inputs(
            self.kernel,
            float32_ptr,
            TritonWarmupTensor(compile_key.target_dtype),
            compile_key.target_logits_stride,
            float32_ptr,
            1,
            float32_ptr,
            1,
            int64_ptr,
            draft_ptr,
            compile_key.draft_logits_stride_0,
            compile_key.draft_logits_stride_1,
            float32_ptr,
            1,
            float32_ptr,
            1,
            int32_ptr,
            int32_ptr,
            float32_ptr,
            compile_key.vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=compile_key.padded_vocab_num_blocks,
            HAS_DRAFT_LOGITS=compile_key.has_draft_logits,
            grid=(1,),
            num_warps=1,
        )

    @kernel_launcher
    def __call__(
        self, grid: tuple[int, ...], *args: Any, **kwargs: Any
    ) -> tuple[tuple[int, ...], dict[str, Any]]:
        return grid, {**dict(zip(self._kernel_arg_names, args)), **kwargs}


_COMPUTE_CUMULATIVE_LOG_P_KERNEL = ComputeCumulativeLogPKernel()


class ComputeLocalResidualMassKernel(
    VllmTritonJitKernel["ComputeLocalResidualMassKernel.CompileKey"]
):
    kernel = staticmethod(_compute_local_residual_mass_kernel)

    @dataclass(frozen=True)
    class CompileKey:
        target_dtype: torch.dtype
        draft_dtype: torch.dtype
        target_logits_stride: int
        draft_logits_stride_0: int
        draft_logits_stride_1: int
        vocab_size: int
        num_speculative_steps: int
        vocab_num_blocks: int
        block_size: int
        padded_vocab_num_blocks: int

    def dispatch(self, **compile_key_fields: Any) -> CompileKey:
        return self.CompileKey(**compile_key_fields)

    def get_warmup_keys(
        self, *, model_dtype: torch.dtype, vocab_size: int, num_speculative_steps: int
    ) -> list[CompileKey]:
        vocab_num_blocks = triton.cdiv(vocab_size, 8192)
        return self._trace_dispatch(self.dispatch)(
            zip_inputs(
                dict(target_dtype=model_dtype, draft_dtype=model_dtype),
                dict(target_dtype=torch.float32, draft_dtype=model_dtype),
            ),
            target_logits_stride=vocab_size,
            draft_logits_stride_0=num_speculative_steps * vocab_size,
            draft_logits_stride_1=vocab_size,
            vocab_size=vocab_size,
            num_speculative_steps=num_speculative_steps,
            vocab_num_blocks=vocab_num_blocks,
            block_size=8192,
            padded_vocab_num_blocks=triton.next_power_of_2(vocab_num_blocks),
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        int32_ptr = TritonWarmupTensor(torch.int32)
        int64_ptr = TritonWarmupTensor(torch.int64)
        float32_ptr = TritonWarmupTensor(torch.float32)
        return triton_warmup_inputs(
            self.kernel,
            float32_ptr,
            1,
            float32_ptr,
            TritonWarmupTensor(compile_key.target_dtype),
            compile_key.target_logits_stride,
            float32_ptr,
            1,
            float32_ptr,
            1,
            TritonWarmupTensor(compile_key.draft_dtype),
            compile_key.draft_logits_stride_0,
            compile_key.draft_logits_stride_1,
            float32_ptr,
            1,
            float32_ptr,
            1,
            int64_ptr,
            int32_ptr,
            int32_ptr,
            float32_ptr,
            compile_key.vocab_size,
            compile_key.num_speculative_steps,
            compile_key.vocab_num_blocks,
            BLOCK_SIZE=compile_key.block_size,
            PADDED_VOCAB_NUM_BLOCKS=compile_key.padded_vocab_num_blocks,
            grid=(1, 1),
        )

    @kernel_launcher
    def __call__(
        self, grid: tuple[int, ...], *args: Any, **kwargs: Any
    ) -> tuple[tuple[int, ...], dict[str, Any]]:
        return grid, {**dict(zip(self._kernel_arg_names, args)), **kwargs}


_COMPUTE_LOCAL_RESIDUAL_MASS_KERNEL = ComputeLocalResidualMassKernel()


class RejectionKernel(VllmTritonJitKernel["RejectionKernel.CompileKey"]):
    kernel = staticmethod(_rejection_kernel)

    @dataclass(frozen=True)
    class CompileKey:
        target_dtype: torch.dtype
        draft_dtype: torch.dtype | None
        target_logits_stride: int
        draft_logits_stride_0: int
        draft_logits_stride_1: int
        vocab_num_blocks: int
        padded_vocab_num_blocks: int
        has_draft_logits: bool
        synthetic_mode: bool
        use_block_verification: bool

    def dispatch(self, **compile_key_fields: Any) -> CompileKey:
        return self.CompileKey(**compile_key_fields)

    def get_warmup_keys(
        self,
        *,
        model_dtype: torch.dtype,
        vocab_size: int,
        num_speculative_steps: int,
        synthetic_mode: bool,
        use_block_verification: bool,
    ) -> list[CompileKey]:
        vocab_num_blocks = triton.cdiv(vocab_size, 8192)
        return self._trace_dispatch(self.dispatch)(
            zip_inputs(
                dict(
                    target_dtype=model_dtype,
                    draft_dtype=None,
                    draft_logits_stride_0=0,
                    draft_logits_stride_1=0,
                    has_draft_logits=False,
                ),
                dict(
                    target_dtype=torch.float32,
                    draft_dtype=None,
                    draft_logits_stride_0=0,
                    draft_logits_stride_1=0,
                    has_draft_logits=False,
                ),
                dict(
                    target_dtype=model_dtype,
                    draft_dtype=model_dtype,
                    draft_logits_stride_0=num_speculative_steps * vocab_size,
                    draft_logits_stride_1=vocab_size,
                    has_draft_logits=True,
                ),
                dict(
                    target_dtype=torch.float32,
                    draft_dtype=model_dtype,
                    draft_logits_stride_0=num_speculative_steps * vocab_size,
                    draft_logits_stride_1=vocab_size,
                    has_draft_logits=True,
                ),
            ),
            target_logits_stride=vocab_size,
            vocab_num_blocks=vocab_num_blocks,
            padded_vocab_num_blocks=triton.next_power_of_2(vocab_num_blocks),
            synthetic_mode=synthetic_mode,
            use_block_verification=use_block_verification,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        int32_ptr = TritonWarmupTensor(torch.int32)
        int64_ptr = TritonWarmupTensor(torch.int64)
        float32_ptr = TritonWarmupTensor(torch.float32)
        draft_ptr = (
            None
            if compile_key.draft_dtype is None
            else TritonWarmupTensor(compile_key.draft_dtype)
        )
        return triton_warmup_inputs(
            self.kernel,
            int64_ptr,
            1,
            int32_ptr,
            float32_ptr,
            float32_ptr,
            TritonWarmupTensor(compile_key.target_dtype),
            compile_key.target_logits_stride,
            int64_ptr,
            1,
            float32_ptr,
            1,
            float32_ptr,
            1,
            int64_ptr,
            draft_ptr,
            compile_key.draft_logits_stride_0,
            compile_key.draft_logits_stride_1,
            float32_ptr,
            1,
            float32_ptr,
            1,
            int32_ptr,
            int32_ptr,
            float32_ptr,
            int64_ptr,
            int64_ptr,
            float32_ptr if compile_key.synthetic_mode else None,
            float32_ptr if compile_key.use_block_verification else None,
            (
                float32_ptr
                if compile_key.use_block_verification
                and compile_key.has_draft_logits
                else None
            ),
            1 if compile_key.has_draft_logits else 0,
            compile_key.vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=compile_key.padded_vocab_num_blocks,
            HAS_DRAFT_LOGITS=compile_key.has_draft_logits,
            SYNTHETIC_MODE=compile_key.synthetic_mode,
            USE_BLOCK_VERIFICATION=compile_key.use_block_verification,
            grid=(1,),
            num_warps=1,
        )

    @kernel_launcher
    def __call__(
        self, grid: tuple[int, ...], *args: Any, **kwargs: Any
    ) -> tuple[tuple[int, ...], dict[str, Any]]:
        return grid, {**dict(zip(self._kernel_arg_names, args)), **kwargs}


_REJECTION_KERNEL = RejectionKernel()


class ResampleKernel(VllmTritonJitKernel["ResampleKernel.CompileKey"]):
    kernel = staticmethod(_resample_kernel)

    @dataclass(frozen=True)
    class CompileKey:
        target_dtype: torch.dtype
        draft_dtype: torch.dtype | None
        resampled_max_dtype: torch.dtype
        target_logits_stride: int
        draft_logits_stride_0: int
        draft_logits_stride_1: int
        vocab_size: int
        block_size: int
        has_draft_logits: bool
        use_fp64: bool
        use_block_verification: bool

    def dispatch(self, **compile_key_fields: Any) -> CompileKey:
        return self.CompileKey(**compile_key_fields)

    def get_warmup_keys(
        self,
        *,
        model_dtype: torch.dtype,
        vocab_size: int,
        num_speculative_steps: int,
        use_fp64: bool,
        use_block_verification: bool,
    ) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(
            zip_inputs(
                dict(
                    target_dtype=model_dtype,
                    draft_dtype=None,
                    draft_logits_stride_0=0,
                    draft_logits_stride_1=0,
                    has_draft_logits=False,
                ),
                dict(
                    target_dtype=torch.float32,
                    draft_dtype=None,
                    draft_logits_stride_0=0,
                    draft_logits_stride_1=0,
                    has_draft_logits=False,
                ),
                dict(
                    target_dtype=model_dtype,
                    draft_dtype=model_dtype,
                    draft_logits_stride_0=num_speculative_steps * vocab_size,
                    draft_logits_stride_1=vocab_size,
                    has_draft_logits=True,
                ),
                dict(
                    target_dtype=torch.float32,
                    draft_dtype=model_dtype,
                    draft_logits_stride_0=num_speculative_steps * vocab_size,
                    draft_logits_stride_1=vocab_size,
                    has_draft_logits=True,
                ),
            ),
            resampled_max_dtype=torch.float64 if use_fp64 else torch.float32,
            target_logits_stride=vocab_size,
            vocab_size=vocab_size,
            block_size=1024,
            use_fp64=use_fp64,
            use_block_verification=use_block_verification,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        int32_ptr = TritonWarmupTensor(torch.int32)
        int64_ptr = TritonWarmupTensor(torch.int64)
        float32_ptr = TritonWarmupTensor(torch.float32)
        draft_ptr = (
            None
            if compile_key.draft_dtype is None
            else TritonWarmupTensor(compile_key.draft_dtype)
        )
        return triton_warmup_inputs(
            self.kernel,
            int64_ptr,
            1,
            TritonWarmupTensor(compile_key.resampled_max_dtype),
            1,
            TritonWarmupTensor(compile_key.target_dtype),
            compile_key.target_logits_stride,
            float32_ptr,
            draft_ptr,
            compile_key.draft_logits_stride_0,
            compile_key.draft_logits_stride_1,
            float32_ptr,
            int32_ptr,
            int32_ptr,
            int32_ptr,
            int64_ptr,
            float32_ptr,
            int64_ptr,
            int64_ptr,
            float32_ptr if compile_key.use_block_verification else None,
            compile_key.vocab_size,
            BLOCK_SIZE=compile_key.block_size,
            HAS_DRAFT_LOGITS=compile_key.has_draft_logits,
            USE_FP64=compile_key.use_fp64,
            USE_BLOCK_VERIFICATION=compile_key.use_block_verification,
            grid=(1, 1),
        )

    @kernel_launcher
    def __call__(
        self, grid: tuple[int, ...], *args: Any, **kwargs: Any
    ) -> tuple[tuple[int, ...], dict[str, Any]]:
        return grid, {**dict(zip(self._kernel_arg_names, args)), **kwargs}


_RESAMPLE_KERNEL = ResampleKernel()


class InsertResampledKernel(
    VllmTritonJitKernel["InsertResampledKernel.CompileKey"]
):
    kernel = staticmethod(_insert_resampled_kernel)

    @dataclass(frozen=True)
    class CompileKey:
        resampled_max_dtype: torch.dtype
        sampled_stride: int
        resampled_local_argmax_stride: int
        resampled_local_max_stride: int
        resample_num_blocks: int
        padded_resample_num_blocks: int

    def dispatch(self, **compile_key_fields: Any) -> CompileKey:
        return self.CompileKey(**compile_key_fields)

    def get_warmup_keys(
        self,
        *,
        num_speculative_steps: int,
        vocab_size: int,
        use_fp64: bool,
    ) -> list[CompileKey]:
        resample_num_blocks = triton.cdiv(vocab_size, 1024)
        return self._trace_dispatch(self.dispatch)(
            resampled_max_dtype=torch.float64 if use_fp64 else torch.float32,
            sampled_stride=num_speculative_steps + 1,
            resampled_local_argmax_stride=resample_num_blocks,
            resampled_local_max_stride=resample_num_blocks,
            resample_num_blocks=resample_num_blocks,
            padded_resample_num_blocks=triton.next_power_of_2(
                resample_num_blocks
            ),
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        int32_ptr = TritonWarmupTensor(torch.int32)
        int64_ptr = TritonWarmupTensor(torch.int64)
        return triton_warmup_inputs(
            self.kernel,
            int64_ptr,
            compile_key.sampled_stride,
            int32_ptr,
            int64_ptr,
            compile_key.resampled_local_argmax_stride,
            TritonWarmupTensor(compile_key.resampled_max_dtype),
            compile_key.resampled_local_max_stride,
            compile_key.resample_num_blocks,
            int32_ptr,
            int32_ptr,
            TritonWarmupTensor(torch.float32),
            PADDED_RESAMPLE_NUM_BLOCKS=compile_key.padded_resample_num_blocks,
            grid=(1,),
        )

    @kernel_launcher
    def __call__(
        self, grid: tuple[int, ...], *args: Any, **kwargs: Any
    ) -> tuple[tuple[int, ...], dict[str, Any]]:
        return grid, {**dict(zip(self._kernel_arg_names, args)), **kwargs}


_INSERT_RESAMPLED_KERNEL = InsertResampledKernel()


def rejection_sample(
    # [num_logits, V]
    target_logits: torch.Tensor,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits: torch.Tensor | None,
    # [num_logits]
    draft_sampled: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_logits]
    pos: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_local_pos: torch.Tensor,
    # [max_num_reqs]
    temperature: torch.Tensor,
    # [max_num_reqs]
    seed: torch.Tensor,
    num_speculative_steps: int,
    # [num_speculative_steps]
    synthetic_conditional_rates: torch.Tensor | None = None,
    use_fp64: bool = False,
    use_block_verification: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert target_logits.ndim == 2 and target_logits.stride(-1) == 1
    assert draft_logits is None or (
        draft_logits.ndim == 3 and draft_logits.stride(-1) == 1
    )
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape
    draft_logits_stride_0 = 0
    draft_logits_stride_1 = 0
    if has_draft_logits := draft_logits is not None:
        draft_logits_stride_0 = draft_logits.stride(0)
        draft_logits_stride_1 = draft_logits.stride(1)
        # In some cases (e.g. MiMo v2.5 Pro + DFlash) the target model's
        # vocab size is larger than the draft's due to padding.
        vocab_size = min(vocab_size, draft_logits.size(-1))

    # Compute the per-vocab-block logits stats, such as target argmax
    # (for greedy requests), and target max + softmax exponential
    # (for non-greedy requests).
    VOCAB_BLOCK_SIZE = 8192
    vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
    padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)
    target_local_argmax = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.int64
    )
    target_local_max = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    target_local_sumexp = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    draft_local_max = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    draft_local_sumexp = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    _COMPUTE_LOCAL_LOGITS_STATS_KERNEL((num_logits, vocab_num_blocks),
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        target_logits,
        target_logits.stride(0),
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        vocab_size,
        num_speculative_steps,
        BLOCK_SIZE=VOCAB_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=has_draft_logits,
    )

    # Precompute the running joint ratio and residual mass for block
    # verification.
    if use_block_verification:
        assert synthetic_conditional_rates is None, (
            "Block verification is incompatible with synthetic acceptance rates."
        )

        # Compute the log of the running joint ratio, p_i.
        # cumulative_log_p[start + i] = log(p_{i+1}), the cumulative ratio after
        # the (i+1)-th draft token.
        cumulative_log_p = target_logits.new_empty(num_logits, dtype=torch.float32)
        _COMPUTE_CUMULATIVE_LOG_P_KERNEL((num_reqs,),
            cumulative_log_p,
            target_logits,
            target_logits.stride(0),
            target_local_max,
            target_local_max.stride(0),
            target_local_sumexp,
            target_local_sumexp.stride(0),
            draft_sampled,
            draft_logits,
            draft_logits_stride_0,
            draft_logits_stride_1,
            draft_local_max,
            draft_local_max.stride(0),
            draft_local_sumexp,
            draft_local_sumexp.stride(0),
            cu_num_logits,
            idx_mapping,
            temperature,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            HAS_DRAFT_LOGITS=has_draft_logits,
            num_warps=1,
        )

        # Compute the per-vocab-block partials of the residual mass, later reduced
        # to the total by _compute_global_residual_mass. Only launched for full
        # draft logits distributions. One-hot drafts used a closed-form residual
        # mass instead.
        if has_draft_logits:
            local_residual_mass = target_logits.new_empty(
                num_logits, vocab_num_blocks, dtype=torch.float32
            )
            _COMPUTE_LOCAL_RESIDUAL_MASS_KERNEL((num_logits, vocab_num_blocks),
                local_residual_mass,
                local_residual_mass.stride(0),
                cumulative_log_p,
                target_logits,
                target_logits.stride(0),
                target_local_max,
                target_local_max.stride(0),
                target_local_sumexp,
                target_local_sumexp.stride(0),
                draft_logits,
                draft_logits_stride_0,
                draft_logits_stride_1,
                draft_local_max,
                draft_local_max.stride(0),
                draft_local_sumexp,
                draft_local_sumexp.stride(0),
                draft_sampled,
                expanded_idx_mapping,
                expanded_local_pos,
                temperature,
                vocab_size,
                num_speculative_steps,
                vocab_num_blocks,
                BLOCK_SIZE=VOCAB_BLOCK_SIZE,
                PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            )
        else:
            local_residual_mass = None
    else:
        cumulative_log_p = None
        local_residual_mass = None

    # Sample up until the first rejected/bonus token, and store
    # the step.
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    target_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    draft_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    _REJECTION_KERNEL((num_reqs,),
        sampled,
        sampled.stride(0),
        num_sampled,
        target_rejected_logsumexp,
        draft_rejected_logsumexp,
        target_logits,
        target_logits.stride(0),
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_sampled,
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        cu_num_logits,
        idx_mapping,
        temperature,
        seed,
        pos,
        synthetic_conditional_rates,
        cumulative_log_p,
        local_residual_mass,
        local_residual_mass.stride(0) if local_residual_mass is not None else 0,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
        HAS_DRAFT_LOGITS=has_draft_logits,
        SYNTHETIC_MODE=synthetic_conditional_rates is not None,
        USE_BLOCK_VERIFICATION=use_block_verification,
        num_warps=1,
    )

    # Resample the rejected/bonus tokens.
    RESAMPLE_BLOCK_SIZE = 1024
    resample_num_blocks = triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE)
    padded_resample_num_blocks = triton.next_power_of_2(resample_num_blocks)
    resampled_local_argmax = target_logits.new_empty(
        num_reqs, resample_num_blocks, dtype=torch.int64
    )
    resampled_local_max = target_logits.new_empty(
        num_reqs,
        resample_num_blocks,
        dtype=torch.float64 if use_fp64 else torch.float32,
    )
    _RESAMPLE_KERNEL((num_reqs, resample_num_blocks),
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        target_logits,
        target_logits.stride(0),
        target_rejected_logsumexp,
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_rejected_logsumexp,
        num_sampled,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        seed,
        pos,
        cumulative_log_p,
        vocab_size,
        BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=has_draft_logits,
        USE_FP64=use_fp64,
        USE_BLOCK_VERIFICATION=use_block_verification,
    )

    # Insert the resampled tokens into the output sampled.
    _INSERT_RESAMPLED_KERNEL((num_reqs,),
        sampled,
        sampled.stride(0),
        num_sampled,
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        resample_num_blocks,
        cu_num_logits,
        expanded_idx_mapping,
        temperature,
        PADDED_RESAMPLE_NUM_BLOCKS=padded_resample_num_blocks,
    )
    return sampled, num_sampled
