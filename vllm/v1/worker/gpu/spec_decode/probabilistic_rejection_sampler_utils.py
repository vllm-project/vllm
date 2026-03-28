# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_block_argmax, tl_rand64


@triton.jit
def _gather_draft_logits_and_target_argmax_kernel(
    # [num_logits, num_blocks]
    target_local_argmax_ptr,
    target_local_argmax_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, V]
    out_draft_logits_ptr,
    out_draft_logits_stride,
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
        # Bonus token. No draft logits to gather or target argmax needed.
        # The bonus token will be resampled later in _gumbel_resample_kernel.
        return

    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    block_idx = tl.program_id(1)
    block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < vocab_size

    if temp == 0.0:
        # Greedy sampling. Get the target logits argmax.
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
        draft_logits = tl.load(
            draft_logits_ptr
            + req_state_idx * draft_logits_stride_0
            + draft_step_idx * draft_logits_stride_1
            + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        tl.store(
            out_draft_logits_ptr + logit_idx * out_draft_logits_stride + block_offsets,
            draft_logits,
            mask=mask,
        )


@triton.jit
def _probabilistic_rejection_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    rejected_steps_ptr,
    # [num_logits, V]
    target_probs_ptr,
    target_probs_stride,
    # [num_logits, target_num_blocks]
    target_local_argmax_ptr,
    target_local_argmax_stride,
    # [num_logits, target_num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits]
    draft_sampled_ptr,
    # [num_logits, V]
    draft_probs_ptr,
    draft_probs_stride,
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
    target_num_blocks,
    PADDED_TARGET_NUM_BLOCKS: tl.constexpr,
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
                target_blocks = tl.arange(0, PADDED_TARGET_NUM_BLOCKS)
                target_blocks_mask = target_blocks < target_num_blocks
                local_target_max = tl.load(
                    target_local_max_ptr
                    + logit_idx * target_local_max_stride
                    + target_blocks,
                    mask=target_blocks_mask,
                    other=float("-inf"),
                )
                max_target_block_idx = tl.argmax(local_target_max, axis=0)
                target_argmax = tl.load(
                    target_local_argmax_ptr
                    + logit_idx * target_local_argmax_stride
                    + max_target_block_idx
                )
                accepted &= target_argmax == draft_sampled
            else:
                target_prob = tl.load(
                    target_probs_ptr + logit_idx * target_probs_stride + draft_sampled
                ).to(tl.float64)
                draft_prob = tl.load(
                    draft_probs_ptr + logit_idx * draft_probs_stride + draft_sampled
                ).to(tl.float64)
                pos = tl.load(pos_ptr + logit_idx)
                u = tl_rand64(seed, pos, includes_zero=False)
                accepted &= target_prob > u * draft_prob
            tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            rejected_step += accepted
    tl.store(rejected_steps_ptr + req_idx, rejected_step)


@triton.jit
def _gumbel_resample_kernel(
    # [num_reqs, num_blocks]
    resampled_local_argmax_ptr,
    resampled_local_argmax_stride,
    # [num_reqs, num_blocks]
    resampled_local_max_ptr,
    resampled_local_max_stride,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_logits, V]
    target_probs_ptr,
    target_probs_stride,
    # [num_logits, V]
    draft_probs_ptr,
    draft_probs_stride,
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
    BLOCK_SIZE: tl.constexpr,
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
        )
    else:
        target_probs = tl.load(
            target_probs_ptr + rejected_token_idx * target_probs_stride + block_offsets,
            mask=mask,
            other=0.0,
        )
        draft_probs = tl.load(
            draft_probs_ptr + rejected_token_idx * draft_probs_stride + block_offsets,
            mask=mask,
            other=0.0,
        )
        residual_probs = tl.maximum(target_probs - draft_probs, 0.0)
        residual_logits = tl.log(residual_probs)

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

    # Gather draft logits and target argmax for greedy sampling.
    GATHER_BLOCK_SIZE = 8192
    gather_num_blocks = triton.cdiv(vocab_size, GATHER_BLOCK_SIZE)
    gathered_draft_logits = target_logits.new_empty(target_logits.shape)
    target_local_argmax = target_logits.new_empty(
        num_logits, gather_num_blocks, dtype=torch.int64
    )
    target_local_max = target_logits.new_empty(
        num_logits, gather_num_blocks, dtype=torch.float32
    )
    _gather_draft_logits_and_target_argmax_kernel[(num_logits, gather_num_blocks)](
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        gathered_draft_logits,
        gathered_draft_logits.stride(0),
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
        BLOCK_SIZE=GATHER_BLOCK_SIZE,
    )

    # Compute target and draft probs.
    target_probs = torch.softmax(target_logits, dim=-1)
    draft_probs = torch.softmax(gathered_draft_logits, dim=-1)

    # Sample up until the first rejected/bonus token, and store
    # the step.
    # [num_reqs, num_speculative_steps + 1]
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    # [num_reqs]
    rejected_steps = sampled.new_empty(num_reqs)
    _probabilistic_rejection_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        rejected_steps,
        target_probs,
        target_probs.stride(0),
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        draft_sampled,
        draft_probs,
        draft_probs.stride(0),
        cu_num_logits,
        idx_mapping,
        temperature,
        seed,
        pos,
        gather_num_blocks,
        PADDED_TARGET_NUM_BLOCKS=triton.next_power_of_2(gather_num_blocks),
        num_warps=1,
    )

    # Resample the rejected/bonus tokens.
    RESAMPLE_BLOCK_SIZE = 1024
    resample_num_blocks = triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE)
    # [num_reqs, resample_num_blocks]
    resampled_local_argmax = target_logits.new_empty(
        num_reqs, resample_num_blocks, dtype=torch.int64
    )
    # [num_reqs, resample_num_blocks]
    resampled_local_max = target_logits.new_empty(
        num_reqs, resample_num_blocks, dtype=torch.float64
    )
    _gumbel_resample_kernel[(num_reqs, resample_num_blocks)](
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        target_logits,
        target_logits.stride(0),
        target_probs,
        target_probs.stride(0),
        draft_probs,
        draft_probs.stride(0),
        rejected_steps,
        cu_num_logits,
        idx_mapping,
        temperature,
        seed,
        pos,
        vocab_size,
        BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,
    )
    max_block_idx = resampled_local_max.argmax(dim=-1, keepdim=True)
    resampled = resampled_local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    sampled.scatter_(1, rejected_steps.unsqueeze(1), resampled.unsqueeze(1))
    return sampled, rejected_steps + 1
