# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utilities for handling LongCat Flash zero experts."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _compute_identity_kernel(
    top_k: int,
    hidden_states_ptr: tl.tensor,
    expert_scales_ptr: tl.tensor,
    num_tokens: int,
    output_ptr: tl.tensor,
    hidden_dim: int,
    scales_stride: int,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    pid = tl.program_id(0)

    batch_id = pid // (hidden_dim // BLOCK_SIZE)
    dim_offset = pid % (hidden_dim // BLOCK_SIZE) * BLOCK_SIZE

    if batch_id >= num_tokens or dim_offset >= hidden_dim:
        return

    h = tl.load(
        hidden_states_ptr
        + batch_id * hidden_dim
        + dim_offset
        + tl.arange(0, BLOCK_SIZE),
        mask=(dim_offset + tl.arange(0, BLOCK_SIZE)) < hidden_dim,
    )

    result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(top_k):
        scale = tl.load(expert_scales_ptr + batch_id * scales_stride + i)
        result += h * scale

    tl.store(
        output_ptr + batch_id * hidden_dim + dim_offset + tl.arange(0, BLOCK_SIZE),
        result,
        mask=(dim_offset + tl.arange(0, BLOCK_SIZE)) < hidden_dim,
    )


def zero_experts_compute_triton(
    expert_indices: torch.Tensor,
    expert_scales: torch.Tensor,
    num_experts: int,
    zero_expert_type: str,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the contribution of LongCat's zero experts.

    Args:
        expert_indices: Top-k expert indices selected by router.
        expert_scales: Corresponding router weights.
        num_experts: Number of real experts.
        zero_expert_type: Currently only "identity" is supported.
        hidden_states: Token hidden states prior to expert dispatch.
    """
    if zero_expert_type != "identity":
        raise ValueError(
            f"Unsupported zero_expert_type={zero_expert_type!r}. "
            "LongCat Flash currently only implements identity zero experts."
        )

    zero_expert_mask = expert_indices < num_experts
    zero_expert_scales = expert_scales.clone()
    zero_expert_scales[zero_expert_mask] = 0.0

    normal_expert_mask = expert_indices >= num_experts
    expert_indices[normal_expert_mask] = 0
    expert_scales[normal_expert_mask] = 0.0

    output = torch.zeros_like(hidden_states).to(hidden_states.device)
    hidden_dim = hidden_states.size(-1)
    num_tokens = hidden_states.size(0)

    grid = lambda meta: (num_tokens * (hidden_dim // meta["BLOCK_SIZE"]),)
    _compute_identity_kernel[grid](
        top_k=expert_indices.size(-1),
        hidden_states_ptr=hidden_states,
        expert_scales_ptr=zero_expert_scales,
        num_tokens=num_tokens,
        output_ptr=output,
        hidden_dim=hidden_dim,
        scales_stride=zero_expert_scales.stride(0),
        BLOCK_SIZE=256,
    )

    return output

