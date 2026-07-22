# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Small Indexer helper kernels."""

import torch
import triton
import triton.language as tl


@triton.jit
def _scale_indexer_weights_kernel(
    weights_ptr,  # [T, H] fp32/bf16
    q_scale_ptr,  # [T, H, 1] fp32, flattened as [T * H]
    out_ptr,  # [T, H] fp32
    n_elements,
    weights_scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    weights = tl.load(weights_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    q_scale = tl.load(q_scale_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, weights * q_scale * weights_scale, mask=mask)


def scale_indexer_weights(
    weights: torch.Tensor,
    q_scale: torch.Tensor,
    weights_scale: float,
    block_size: int = 1024,
) -> torch.Tensor:
    """Apply `weights * q_scale.squeeze(-1) * weights_scale` in one Triton launch."""
    assert weights.dim() == 2, f"weights must be [T, H], got {tuple(weights.shape)}"
    assert q_scale.shape == (
        weights.size(0),
        weights.size(1),
        1,
    ), f"q_scale shape {tuple(q_scale.shape)} incompatible with weights {tuple(weights.shape)}"
    assert weights.is_contiguous(), "weights must be contiguous"
    assert q_scale.is_contiguous(), "q_scale must be contiguous"

    n_elements = weights.numel()
    out = torch.empty_like(weights, dtype=torch.float32)
    if n_elements == 0:
        return out

    grid = (triton.cdiv(n_elements, block_size),)
    _scale_indexer_weights_kernel[grid](
        weights,
        q_scale,
        out,
        n_elements,
        weights_scale,
        BLOCK_SIZE=block_size,
    )
    return out
