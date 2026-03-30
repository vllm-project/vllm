# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_block_argmax, tl_rand64


@triton.jit
def _compute_block_max_and_sumexp(logits):
    block_max = tl.max(logits, axis=0)
    block_sumexp = tl.where(
        block_max > float("-inf"),
        tl.sum(tl.exp(logits - block_max)),
        0.0,
    )
    return block_max, block_sumexp


@triton.jit
def _compute_global_lse(
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
def _prepare_inputs_kernel(
    # [num_logits, num_blocks]
    target_local_argmax_ptr,
    target_local_argmax_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [num_logits, V]
    out_draft_logits_ptr,
    out_draft_logits_stride,
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
):
    logit_idx = tl.program_id(0)
    draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)

    if draft_step_idx >= num_speculative_steps:
        # Bonus token. Skip gathering draft logits, max/argmax,
        # and summed exponentials.
        return

    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx)
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
        # Gather draft max and summed exponentials.
        draft_logits = tl.load(
            draft_logits_ptr
            + req_state_idx * draft_logits_stride_0
            + draft_step_idx * draft_logits_stride_1
            + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        draft_max, draft_sumexp = _compute_block_max_and_sumexp(draft_logits)
        tl.store(
            draft_local_max_ptr + logit_idx * draft_local_max_stride + block_idx,
            draft_max,
        )
        tl.store(
            draft_local_sumexp_ptr + logit_idx * draft_local_sumexp_stride + block_idx,
            draft_sumexp,
        )
        # Gather draft logits.
        tl.store(
            out_draft_logits_ptr + logit_idx * out_draft_logits_stride + block_offsets,
            draft_logits,
            mask=mask,
        )
        # Gather target max and summed exponentials.
        target_logits = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        target_max, target_sumexp = _compute_block_max_and_sumexp(target_logits)
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


@triton.jit
def _probabilistic_rejection_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    rejected_steps_ptr,
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
    # [num_logits, V]
    draft_logits_ptr,
    draft_logits_stride,
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
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    seed = tl.load(seed_ptr + req_state_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    rejected_step = 0
    accepted = True
    for i in range(num_tokens - 1):
        if accepted:
            logit_idx = start_idx + i
            draft_sampled = tl.load(draft_sampled_ptr + logit_idx + 1)
            if temp == 0.0:
                # Greedy sampling. Only accept the sampled draft token if
                # it exactly matches the target argmax.
                target_blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
                target_blocks_mask = target_blocks < vocab_num_blocks
                target_local_max = tl.load(
                    target_local_max_ptr
                    + logit_idx * target_local_max_stride
                    + target_blocks,
                    mask=target_blocks_mask,
                    other=float("-inf"),
                )
                max_target_block_idx = tl.argmax(target_local_max, axis=0)
                target_argmax = tl.load(
                    target_local_argmax_ptr
                    + logit_idx * target_local_argmax_stride
                    + max_target_block_idx
                )
                accepted &= target_argmax == draft_sampled
            else:
                target_logit = tl.load(
                    target_logits_ptr + logit_idx * target_logits_stride + draft_sampled
                ).to(tl.float64)
                draft_logit = tl.load(
                    draft_logits_ptr + logit_idx * draft_logits_stride + draft_sampled
                ).to(tl.float64)
                target_lse = _compute_global_lse(
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_sumexp_ptr,
                    target_local_sumexp_stride,
                    logit_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                ).to(tl.float64)
                draft_lse = _compute_global_lse(
                    draft_local_max_ptr,
                    draft_local_max_stride,
                    draft_local_sumexp_ptr,
                    draft_local_sumexp_stride,
                    logit_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                ).to(tl.float64)
                target_log_prob = target_logit - target_lse
                draft_log_prob = draft_logit - draft_lse
                pos = tl.load(pos_ptr + logit_idx)
                u = tl_rand64(seed, pos, includes_zero=False)
                # Probability ratio test: p(x) > u * q(x)
                # Equivalent log form: log_p(x) > log(u) + log_q(x)
                accepted &= target_log_prob > tl.log(u) + draft_log_prob
            tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            rejected_step += accepted
    tl.store(rejected_steps_ptr + req_idx, rejected_step)


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
    # [num_logits, lse_num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, lse_num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [num_logits, V]
    draft_logits_ptr,
    draft_logits_stride,
    # [num_logits, lse_num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_logits, lse_num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_reqs]
    rejected_step_ptr,
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
    vocab_size,
    vocab_num_blocks,
    BLOCK_SIZE: tl.constexpr,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    rejected_token_idx = start_idx + tl.load(rejected_step_ptr + req_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < vocab_size

    # Compute the residual logits to resample the rejected token
    # from. In the case of no rejections (bonus token), we directly
    # use the target logits.
    if temp == 0.0 or (rejected_token_idx == end_idx - 1):
        # Greedy sampling / bonus token. In either case, use the
        # target logits directly to reduce numerical error.
        residual_logits = tl.load(
            target_logits_ptr
            + rejected_token_idx * target_logits_stride
            + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
    else:
        target_logits = tl.load(
            target_logits_ptr
            + rejected_token_idx * target_logits_stride
            + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float64)
        draft_logits = tl.load(
            draft_logits_ptr + rejected_token_idx * draft_logits_stride + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float64)
        target_lse = _compute_global_lse(
            target_local_max_ptr,
            target_local_max_stride,
            target_local_sumexp_ptr,
            target_local_sumexp_stride,
            rejected_token_idx,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        ).to(tl.float64)
        draft_lse = _compute_global_lse(
            draft_local_max_ptr,
            draft_local_max_stride,
            draft_local_sumexp_ptr,
            draft_local_sumexp_stride,
            rejected_token_idx,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        ).to(tl.float64)
        target_log_probs = target_logits - target_lse
        draft_log_probs = draft_logits - draft_lse
        # Compute the residual: max(p(x) - q(x), 0)
        # Equivalent log form: log(max(exp(log_p(x)) - exp(log_q(x)), 0))
        # The more numerically stable form is:
        # log(max(exp(a) - exp(b), 0)) = a + log(max(1 - exp(b - a), 0))
        ratio = tl.exp(draft_log_probs - target_log_probs)
        residual_logits = tl.where(
            ratio < 1.0,
            target_log_probs + tl.log(1.0 - ratio),
            float("-inf"),
        ).to(tl.float32)

    # Resample the rejected/bonus token.
    value, token_id = gumbel_block_argmax(
        residual_logits,
        mask,
        block_idx,
        req_state_idx,
        rejected_token_idx,
        temp_ptr,
        seed_ptr,
        pos_ptr,
        None,
        0,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=False,
    )
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
    PADDED_RESAMPLE_NUM_BLOCKS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    resample_idx = tl.load(num_sampled_ptr + req_idx)
    block_offsets = tl.arange(0, PADDED_RESAMPLE_NUM_BLOCKS)
    block_mask = block_offsets < resample_num_blocks
    resampled_local_max = tl.load(
        resampled_local_max_ptr + req_idx * resampled_local_max_stride + block_offsets,
        mask=block_mask,
        other=float("-inf"),
    )
    resampled_max_block_idx = tl.argmax(resampled_local_max, axis=0)
    resampled = tl.load(
        resampled_local_argmax_ptr
        + req_idx * resampled_local_argmax_stride
        + resampled_max_block_idx,
    )
    tl.store(
        sampled_ptr + req_idx * sampled_stride + resample_idx,
        resampled,
    )
    tl.store(num_sampled_ptr + req_idx, resample_idx + 1)


def probabilistic_rejection_sample(
    # [num_logits, V]
    target_logits: torch.Tensor,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape

    # Gather draft logits, compute target argmax for greedy, and
    # compute per-block LSE and max for non-greedy requests.
    VOCAB_BLOCK_SIZE = 8192
    vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
    padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)
    gathered_draft_logits = target_logits.new_empty(target_logits.shape)
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
    _prepare_inputs_kernel[(num_logits, vocab_num_blocks)](
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        gathered_draft_logits,
        gathered_draft_logits.stride(0),
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        target_logits,
        target_logits.stride(0),
        draft_logits,
        draft_logits.stride(0),
        draft_logits.stride(1),
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        vocab_size,
        num_speculative_steps,
        BLOCK_SIZE=VOCAB_BLOCK_SIZE,
    )

    # Sample up until the first rejected/bonus token, and store
    # the step.
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs)
    _probabilistic_rejection_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_logits,
        target_logits.stride(0),
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_sampled,
        gathered_draft_logits,
        gathered_draft_logits.stride(0),
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        cu_num_logits,
        idx_mapping,
        temperature,
        seed,
        pos,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
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
        num_reqs, resample_num_blocks, dtype=torch.float64
    )
    _resample_kernel[(num_reqs, resample_num_blocks)](
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        target_logits,
        target_logits.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        gathered_draft_logits,
        gathered_draft_logits.stride(0),
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        num_sampled,
        cu_num_logits,
        idx_mapping,
        temperature,
        seed,
        pos,
        vocab_size,
        vocab_num_blocks,
        BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
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
        PADDED_RESAMPLE_NUM_BLOCKS=padded_resample_num_blocks,
    )
    return sampled, num_sampled
