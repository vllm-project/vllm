# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

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
    draft_sampled_ptr,
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
        draft_token = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
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
        draft_logit = tl.load(
            draft_logits_ptr
            + req_state_idx * draft_logits_stride_0
            + draft_step * draft_logits_stride_1
            + token,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
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
            # Get local draft max and summed exponentials.
            draft_logits = tl.load(
                draft_logits_ptr
                + req_state_idx * draft_logits_stride_0
                + draft_step_idx * draft_logits_stride_1
                + block_offsets,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
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
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if temp == 0.0:
        return

    log_p = 0.0
    for step in range(num_draft_tokens):
        logit_idx = start_idx + step
        draft_token = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
        target_logprob, draft_logprob, _, _ = _compute_global_logprobs_and_logsumexp(
            draft_token,
            True,  # mask
            logit_idx,
            req_state_idx,
            step,
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
    logit_idx = tl.program_id(0)
    draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)
    if draft_step_idx == 0 or draft_step_idx >= num_speculative_steps:
        # The acceptance threshold, h, looks one position ahead and sums
        # over: max(p_i * M_b(x|x_{<i}) - M_s(x|x_{<i}), 0). Tokens at the
        # first and last (bonus) positions aren't needed for this computation.
        return

    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx)
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
    accepted = True
    for i in range(num_draft_tokens):
        logit_idx = start_idx + i
        draft_sampled = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
        pos = tl.load(pos_ptr + logit_idx)
        u = tl_rand32(seed, pos, includes_zero=False)
        if USE_BLOCK_VERIFICATION and not is_greedy:
            # Block verification (Sun et al., 2024): https://arxiv.org/abs/2403.10444
            prefix_joint_ratio = tl.exp(
                tl.load(cumulative_log_p_ptr + logit_idx).to(tl.float32)
            )
            if i < num_draft_tokens - 1:
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
                    draft_sampled_ptr,
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
        elif accepted:
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
                    # -1 is used for padded draft token ids that should be rejected.
                    accepted &= (u < rate) & (draft_sampled >= 0)
                else:
                    accepted &= target_argmax == draft_sampled
                tl.store(
                    sampled_ptr + req_idx * sampled_stride + i,
                    draft_sampled if accepted else target_argmax,
                )
            else:
                # Speculative decoding (Leviathan et al., 2023): https://arxiv.org/abs/2211.17192
                # -1 is used for padded draft token ids that should be rejected.
                is_valid_draft = draft_sampled >= 0
                # Avoid possible OOB ptr access.
                draft_sampled = tl.maximum(0, draft_sampled)
                target_logprob, draft_logprob, target_lse, draft_lse = (
                    _compute_global_logprobs_and_logsumexp(
                        draft_sampled,
                        True,  # mask
                        logit_idx,
                        req_state_idx,
                        i,
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
                    accepted &= u < rate
                else:
                    # Probability ratio test: p(x) > u * q(x)
                    # Equivalent log form: log_p(x) > log(u) + log_q(x)
                    accepted &= target_logprob > tl.log(u) + draft_logprob
                accepted &= is_valid_draft
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            accepted_length += accepted
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

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    target_logits = tl.load(
        target_logits_ptr + resample_token_idx * target_logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)

    # Compute the residual logits to resample the rejected token from.
    if is_bonus:
        # Bonus token (no rejections). Directly use the target logits.
        residual_logits = target_logits
    elif HAS_DRAFT_LOGITS:
        draft_logits = tl.load(
            draft_logits_ptr
            + req_state_idx * draft_logits_stride_0
            + resample_idx * draft_logits_stride_1
            + block,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
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
        rejected_draft_token = tl.load(draft_sampled_ptr + resample_token_idx + 1)
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
        None,  # processed_logits_ptr
        0,  # processed_logits_stride
        None,  # processed_logits_col_ptr
        vocab_size,
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
    _compute_local_logits_stats_kernel[(num_logits, vocab_num_blocks)](
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
        _compute_cumulative_log_p_kernel[(num_reqs,)](
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
            _compute_local_residual_mass_kernel[(num_logits, vocab_num_blocks)](
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
    _rejection_kernel[(num_reqs,)](
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
    _resample_kernel[(num_reqs, resample_num_blocks)](
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
    _insert_resampled_kernel[(num_reqs,)](
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
