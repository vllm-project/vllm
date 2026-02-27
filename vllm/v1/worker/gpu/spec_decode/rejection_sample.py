# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _strict_rejection_sample_kernel(
    sampled_ptr,  # [num_reqs, num_speculative_steps + 1]
    sampled_stride,
    num_sampled_ptr,  # [num_reqs]
    target_sampled_ptr,  # [num_draft_tokens + num_reqs]
    input_ids_ptr,  # [num_draft_tokens + num_reqs]
    cu_num_logits_ptr,  # [num_reqs + 1]
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx

    num_sampled = 0
    rejected = False
    for i in range(num_tokens - 1):
        if not rejected:
            target_sampled = tl.load(target_sampled_ptr + start_idx + i)
            draft_sampled = tl.load(input_ids_ptr + start_idx + i + 1)
            tl.store(sampled_ptr + req_idx * sampled_stride + i, target_sampled)
            num_sampled += 1
            if target_sampled != draft_sampled:
                rejected = True
    if not rejected:
        target_sampled = tl.load(target_sampled_ptr + start_idx + num_tokens - 1)
        tl.store(
            sampled_ptr + req_idx * sampled_stride + num_tokens - 1, target_sampled
        )
        num_sampled += 1
    tl.store(num_sampled_ptr + req_idx, num_sampled)


def strict_rejection_sample(
    # [num_draft_tokens + num_reqs]
    target_sampled: torch.Tensor,
    # [num_draft_tokens + num_reqs]
    input_ids: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = cu_num_logits.shape[0] - 1
    sampled = torch.empty(
        num_reqs,
        num_speculative_steps + 1,
        dtype=target_sampled.dtype,
        device=target_sampled.device,
    )
    num_sampled = torch.empty(
        num_reqs,
        dtype=torch.int32,
        device=target_sampled.device,
    )
    _strict_rejection_sample_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_sampled,
        input_ids,
        cu_num_logits,
        num_warps=1,
    )
    return sampled, num_sampled


@triton.jit
def _probabilistic_rejection_sample_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    resampled_ptr,
    target_sampled_ptr,
    draft_sampled_ptr,
    target_probs_ptr,
    draft_probs_ptr,
    vocab_size,
    rand_ptr,
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    logits_start_idx = tl.load(cu_num_logits_ptr + req_idx)
    start_idx = logits_start_idx - req_idx
    num_draft_tokens = tl.load(cu_num_logits_ptr + req_idx + 1) - logits_start_idx - 1

    num_sampled = 0
    rejected = False
    for i in range(num_draft_tokens):
        if not rejected:
            draft_sampled = tl.load(draft_sampled_ptr + logits_start_idx + i + 1)
            resampled = tl.load(resampled_ptr + start_idx + i)
            target_prob = tl.load(
                target_probs_ptr + (start_idx + i) * vocab_size + draft_sampled
            )
            draft_prob = tl.load(
                draft_probs_ptr + (start_idx + i) * vocab_size + draft_sampled
            )
            r = tl.load(rand_ptr + start_idx + i)
            accept_prob = tl.minimum(1.0, target_prob / draft_prob)
            rejected |= (draft_prob <= 0) | (r > accept_prob)
            sampled_token = tl.where(rejected, resampled, draft_sampled)
            tl.store(sampled_ptr + req_idx * sampled_stride + i, sampled_token)
            num_sampled += 1
    if not rejected:
        # All draft tokens were accepted. Append the bonus token sampled from the
        # target model.
        bonus_sampled = tl.load(target_sampled_ptr + logits_start_idx + num_draft_tokens)
        tl.store(
            sampled_ptr + req_idx * sampled_stride + num_draft_tokens, bonus_sampled
        )
        num_sampled += 1
    tl.store(num_sampled_ptr + req_idx, num_sampled)


def probabilistic_rejection_sample(
    # [num_draft_tokens + num_reqs]
    target_sampled: torch.Tensor,
    # [num_draft_tokens + num_reqs]
    draft_sampled: torch.Tensor,
    # [num_draft_tokens + num_reqs, V]
    target_logits: torch.Tensor,
    # [num_reqs, num_speculative_steps, V]
    draft_logits: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape
    device = target_sampled.device

    # Compute target probs.
    mask = torch.ones(num_logits, dtype=torch.bool, device=device)
    bonus_token_indices = cu_num_logits[1:] - 1
    mask[bonus_token_indices] = False
    target_probs = torch.softmax(target_logits[mask], dim=-1)

    # Compute draft probs.
    num_draft_tokens = bonus_token_indices - cu_num_logits[:-1]
    mask = torch.arange(num_speculative_steps, device=device).unsqueeze(
        0
    ) < num_draft_tokens.unsqueeze(1)
    draft_probs = torch.softmax(draft_logits[mask], dim=-1)

    # Compute distribution to resample from after draft token rejection.
    resample_probs = torch.clamp(target_probs - draft_probs, min=0.0)
    norms = resample_probs.sum(dim=-1, keepdim=True)
    resample_probs = torch.where(norms > 1e-8, resample_probs / norms, target_probs)
    resampled = torch.multinomial(resample_probs, num_samples=1).squeeze(-1)

    # [num_draft_tokens]
    rand = torch.rand(
        num_logits - num_reqs,
        dtype=draft_probs.dtype,
        device=device,
    )

    # [num_reqs, num_speculative_steps + 1]
    sampled = torch.empty(
        num_reqs,
        num_speculative_steps + 1,
        dtype=target_sampled.dtype,
        device=device,
    )
    # [num_reqs]
    num_sampled = torch.empty(
        num_reqs,
        dtype=torch.int32,
        device=device,
    )
    _probabilistic_rejection_sample_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        resampled,
        target_sampled,
        draft_sampled,
        target_probs,
        draft_probs,
        vocab_size,
        rand,
        cu_num_logits,
        num_warps=1,
    )
    return sampled, num_sampled
