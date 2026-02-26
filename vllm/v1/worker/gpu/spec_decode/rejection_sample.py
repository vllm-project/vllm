# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _rejection_sample_kernel(
    sampled_ptr,  # [num_reqs, num_speculative_steps + 1]
    sampled_stride,
    num_sampled_ptr,  # [num_reqs]
    input_ids_ptr,  # [num_draft_tokens + num_reqs]
    cu_num_logits_ptr,  # [num_reqs + 1]
    target_probs_ptr,  # [num_draft_tokens + num_reqs, vocab_size]
    target_probs_stride,
    draft_probs_ptr,  # [num_draft_tokens, vocab_size]
    draft_probs_stride,
    recovered_ids_ptr,  # [num_draft_tokens + num_reqs]
    seeds_ptr,  # [num_reqs]
    pos_ptr,  # [num_reqs]
    idx_mapping_ptr,  # [max_num_reqs]
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    pos = tl.load(pos_ptr + req_idx)
    seed = tl.load(seeds_ptr + req_state_idx)
    num_tokens = end_idx - start_idx

    num_sampled = 0
    rejected = False
    for i in range(num_tokens - 1):
        if not rejected:
            draft_id = tl.load(input_ids_ptr + start_idx + i + 1)
            target_prob = tl.load(
                target_probs_ptr + (start_idx + i) * target_probs_stride + draft_id
            )
            draft_prob = tl.load(
                draft_probs_ptr
                + (start_idx + i - req_idx) * draft_probs_stride
                + draft_id
            )
            u = tl.rand(seed=seed, offset=pos + i)
            if target_prob >= u * draft_prob:  # Accept
                token_id = draft_id
            else:  # Reject
                token_id = tl.load(recovered_ids_ptr + start_idx + i).to(tl.int32)
                rejected = True
            tl.store(sampled_ptr + req_idx * sampled_stride + i, token_id)
            num_sampled += 1
    if not rejected:
        bonus_id = tl.load(recovered_ids_ptr + start_idx + num_tokens - 1)
        tl.store(sampled_ptr + req_idx * sampled_stride + num_tokens - 1, bonus_id)
        num_sampled += 1
    tl.store(num_sampled_ptr + req_idx, num_sampled)


def rejection_sample(
    # [num_draft_tokens + num_reqs]
    input_ids: torch.Tensor,
    # [num_draft_tokens + num_reqs]
    recovered_ids: torch.Tensor,
    # [num_draft_tokens + num_reqs, vocab_size]
    target_probs: torch.Tensor,
    # [num_draft_tokens, vocab_size],
    draft_probs: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_reqs]
    seeds: torch.Tensor,
    # [num_reqs]
    pos: torch.Tensor,
    # [max_num_reqs]
    idx_mapping: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = cu_num_logits.shape[0] - 1
    sampled = torch.empty(
        num_reqs,
        num_speculative_steps + 1,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    num_sampled = torch.empty(
        num_reqs,
        dtype=torch.int32,
        device=input_ids.device,
    )
    _rejection_sample_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        input_ids,
        cu_num_logits,
        target_probs,
        target_probs.stride(0),
        draft_probs,
        draft_probs.stride(0),
        recovered_ids,
        seeds,
        pos,
        idx_mapping,
        num_warps=1,
    )
    return sampled, num_sampled


@triton.jit
def _sample_recovered_and_bonus_tokens_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    target_probs_ptr,
    target_probs_stride,
    draft_probs_ptr,
    draft_probs_stride,
    cu_num_logits_ptr,
    seeds_ptr,
    pos_ptr,
    idx_mapping_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + token_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_state_idx + 1)
    is_bonus_token_idx = token_idx == end_idx - 1

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    target_probs = tl.load(
        target_probs_ptr + token_idx * target_probs_stride + block,
        mask=mask,
        other=0.0,
    )
    if not is_bonus_token_idx:
        draft_probs = tl.load(
            draft_probs_ptr + (token_idx - req_state_idx) * draft_probs_stride + block,
            mask=mask,
            other=0.0,
        )
        target_probs -= draft_probs

    # Calculate the seed for exponential noise.
    seed = tl.load(seeds_ptr + req_state_idx)
    pos = tl.load(pos_ptr + token_idx)
    gumbel_seed = tl.randint(seed, pos)
    # Generate exponential noise in FP32.
    u = tl.rand(gumbel_seed, block)
    u = tl.maximum(u, 1e-7)
    exp_noise = -tl.log(u)

    value, idx = tl.max(target_probs / exp_noise, axis=0, return_indices=True)
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


def sample_recovered_and_bonus_tokens(
    target_probs: torch.Tensor,  # [num_draft_tokens + num_reqs, vocab_size]
    draft_probs: torch.Tensor,  # [num_draft_tokens, vocab_size]
    cu_num_logits: torch.Tensor,  # [num_reqs + 1]
    idx_mapping: torch.Tensor,  # [num_draft_tokens + num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_reqs]
) -> torch.Tensor:
    num_tokens, vocab_size = target_probs.shape
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = torch.empty(
        num_tokens,
        num_blocks,
        dtype=torch.int64,
        device=target_probs.device,
    )
    local_max = torch.empty(
        num_tokens,
        num_blocks,
        dtype=torch.float32,
        device=target_probs.device,
    )
    _sample_recovered_and_bonus_tokens_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        target_probs,
        target_probs.stride(0),
        draft_probs,
        draft_probs.stride(0),
        cu_num_logits,
        seed,
        pos,
        idx_mapping,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # NOTE(woosuk): Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return sampled
