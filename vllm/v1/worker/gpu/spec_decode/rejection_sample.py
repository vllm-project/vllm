# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample


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
    target_sampled_ptr,
    draft_sampled_ptr,
    target_probs_ptr,
    draft_probs_ptr,
    num_speculative_steps,
    vocab_size,
    rand_ptr,
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_tokens = tl.load(cu_num_logits_ptr + req_idx + 1) - start_idx

    num_sampled = 0
    rejected = False
    for i in range(num_tokens - 1):
        if not rejected:
            draft_sampled = tl.load(draft_sampled_ptr + start_idx + i + 1)
            target_prob = tl.load(
                target_probs_ptr + (start_idx + i) * vocab_size + draft_sampled
            )
            draft_prob = tl.load(
                draft_probs_ptr + (req_idx * num_speculative_steps + i) * vocab_size + draft_sampled
            )
            r = tl.load(rand_ptr + start_idx + i)
            accept_prob = tl.minimum(1.0, target_prob / draft_prob)
            rejected |= (draft_prob <= 0) | (r > accept_prob)
            tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            num_sampled += 1
    if not rejected:
        # All draft tokens were accepted. Append the bonus token sampled from the
        # target model.
        bonus_sampled = tl.load(target_sampled_ptr + start_idx + num_tokens - 1)
        tl.store(
            sampled_ptr + req_idx * sampled_stride + num_tokens - 1, bonus_sampled
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
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [num_reqs]
    temperature: torch.Tensor,
    # [num_reqs]
    seeds: torch.Tensor,

) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape
    device = target_sampled.device
    req_idxs = torch.arange(num_reqs, device=device)

    # Compute target and draft probs.
    target_probs = torch.softmax(target_logits, dim=-1)
    draft_probs = torch.softmax(draft_logits, dim=-1)

    # [num_logits]
    rand = torch.rand(
        num_logits,
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
        target_sampled,
        draft_sampled,
        target_probs,
        draft_probs,
        num_speculative_steps,
        vocab_size,
        rand,
        cu_num_logits,
        num_warps=1,
    )

    # Get indices of the first rejected draft token.
    rejected_draft_steps = torch.clamp(num_sampled - 1, max=num_speculative_steps - 1)
    rejected_logit_idxs = cu_num_logits[:-1] + rejected_draft_steps
    # Resample from adjusted distribution.
    resample_probs = torch.clamp(
        target_probs[rejected_logit_idxs] - draft_probs[req_idxs, rejected_draft_steps],
        min=0.0
    )
    resampled = gumbel_sample(
        torch.log(resample_probs),
        idx_mapping,
        temperature,
        seeds,
        num_sampled,
        apply_temperature=False,
    )
    # Only set the non-bonus tokens.
    resample_mask = num_sampled <= num_speculative_steps
    sampled[resample_mask, rejected_draft_steps[resample_mask]] = resampled[resample_mask]
    return sampled, num_sampled
