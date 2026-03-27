# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import tl_rand64

MIN_ACCEPTANCE_DECAY_FACTOR = 0.85


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
    base_acceptance_rate,
    decay_factor,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    seed = tl.load(seeds_ptr + req_state_idx)

    num_sampled = 0
    acceptance_rate = base_acceptance_rate
    rejected = False
    for i in range(num_tokens - 1):
        if not rejected:
            logit_idx = start_idx + i
            pos = tl.load(pos_ptr + logit_idx)
            u = tl_rand64(seed, pos, includes_zero=False)
            if u < acceptance_rate:
                sampled = tl.load(input_ids_ptr + logit_idx + 1).to(tl.int64)
            else:
                sampled = tl.load(target_sampled_ptr + logit_idx)
                rejected = True
            tl.store(sampled_ptr + req_idx * sampled_stride + i, sampled)
            num_sampled += 1
            acceptance_rate *= decay_factor
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
    base_acceptance_rate: float,
    decay_factor: float,
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
        base_acceptance_rate,
        decay_factor,
        num_warps=1,
    )
    return sampled, num_sampled


def compute_synthetic_rejection_sampler_params(
    p_avg: float, n: int, tol: float = 1e-9
) -> tuple[float, float]:
    def mean_joint_prob(a_0: float, gamma: float, n: int):
        total = 0.0
        for i in range(n):
            total += a_0 ** (i + 1) * gamma ** (i * (i + 1) // 2)
        return total / n

    def min_valid_decay_factor(p: float, n: int, tol: float = 1e-9) -> float:
        low, high = MIN_ACCEPTANCE_DECAY_FACTOR, 1.0
        if mean_joint_prob(1, low, n) >= p:
            return low

        # Sweep for a gamma decay factor that is guaranteed
        # to yield a base acceptance rate <= 1.
        while (high - low) > tol:
            mid = (low + high) / 2
            if mean_joint_prob(1, mid, n) >= p:
                high = mid
            else:
                low = mid
        return high

    def compute_base_acceptance_rate(
        p_avg: float, gamma: float, n: int, tol: float = 1e-9
    ) -> float:
        if p_avg <= 0.0:
            return 0.0
        if p_avg >= 1.0:
            return 1.0

        # Sweep for a base acceptance rate that yields
        # the desired mean joint probability.
        low, high = 0.0, 1.0
        while (high - low) > tol:
            mid = (low + high) / 2
            if mean_joint_prob(mid, gamma, n) >= p_avg:
                high = mid
            else:
                low = mid
        return high

    decay_factor = min_valid_decay_factor(p_avg, n)
    base_rate = compute_base_acceptance_rate(p_avg, decay_factor, n)
    return base_rate, decay_factor
