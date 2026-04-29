# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import tl_rand64


@triton.jit
def _synthetic_rejection_sample_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_draft_tokens + num_reqs]
    target_sampled_ptr,
    # [num_draft_tokens + num_reqs]
    input_ids_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_logits]
    pos_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    seeds_ptr,
    # [num_speculative_steps]
    acceptance_rates_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    seed = tl.load(seeds_ptr + req_state_idx)

    num_sampled = 0
    rejected = False
    for i in range(num_tokens - 1):
        if not rejected:
            logit_idx = start_idx + i
            pos = tl.load(pos_ptr + logit_idx)
            u = tl_rand64(seed, pos, includes_zero=False)
            acceptance_rate = tl.load(acceptance_rates_ptr + i)
            if u < acceptance_rate:
                sampled = tl.load(input_ids_ptr + logit_idx + 1).to(tl.int64)
            else:
                sampled = tl.load(target_sampled_ptr + logit_idx)
                rejected = True
            tl.store(sampled_ptr + req_idx * sampled_stride + i, sampled)
            num_sampled += 1
    if not rejected:
        target_sampled = tl.load(target_sampled_ptr + start_idx + num_tokens - 1)
        tl.store(
            sampled_ptr + req_idx * sampled_stride + num_tokens - 1, target_sampled
        )
        num_sampled += 1
    tl.store(num_sampled_ptr + req_idx, num_sampled)


def synthetic_rejection_sample(
    # [num_draft_tokens + num_reqs]
    target_sampled: torch.Tensor,
    # [num_draft_tokens + num_reqs]
    draft_sampled: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_logits]
    pos: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    seed: torch.Tensor,
    # [num_speculative_steps]
    acceptance_rates: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = cu_num_logits.shape[0] - 1
    sampled = target_sampled.new_empty(num_reqs, num_speculative_steps + 1)
    num_sampled = target_sampled.new_empty(num_reqs, dtype=torch.int32)
    _synthetic_rejection_sample_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_sampled,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        seed,
        acceptance_rates,
        num_warps=1,
    )
    return sampled, num_sampled
