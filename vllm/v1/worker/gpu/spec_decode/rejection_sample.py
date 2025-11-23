# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _rejection_sample_kernel(
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


def rejection_sample(
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
    _rejection_sample_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_sampled,
        input_ids,
        cu_num_logits,
        num_warps=1,
    )
    return sampled, num_sampled
