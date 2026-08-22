# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small fused RMSNorm helpers for EAGLE3 draft models."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_dual_rmsnorm_cat_kernel(
    a_ptr,
    b_ptr,
    w_a_ptr,
    w_b_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    block_h: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = tl.arange(0, block_h)
    mask = offsets < hidden_size

    if group == 0:
        x = tl.load(a_ptr + row * hidden_size + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        w = tl.load(w_a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    else:
        x = tl.load(b_ptr + row * hidden_size + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        w = tl.load(w_b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / hidden_size
    y = x * tl.rsqrt(variance + eps) * w
    out_offset = row * (2 * hidden_size) + group * hidden_size + offsets
    tl.store(out_ptr + out_offset, y.to(out_ptr.dtype.element_ty), mask=mask)


def fused_dual_rmsnorm_cat(
    a: torch.Tensor,
    b: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """RMS-normalize two tensors and concatenate the normalized results.

    Returns ``cat([rmsnorm(a, w_a), rmsnorm(b, w_b)], dim=-1)`` using a
    single Triton launch. Inputs must be contiguous, same-shaped CUDA tensors.
    """
    if a.shape != b.shape:
        raise ValueError(f"input shapes must match, got {a.shape} and {b.shape}")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("fused_dual_rmsnorm_cat requires CUDA tensors")
    if a.device != b.device or a.dtype != b.dtype:
        raise ValueError("inputs must have the same device and dtype")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("inputs must be contiguous")
    hidden_size = a.shape[-1]
    if w_a.device != a.device or w_b.device != a.device:
        raise ValueError("RMSNorm weights must be on the input device")
    if not w_a.is_contiguous() or not w_b.is_contiguous():
        raise ValueError("RMSNorm weights must be contiguous")
    if w_a.shape != (hidden_size,) or w_b.shape != (hidden_size,):
        raise ValueError(
            "RMSNorm weights must match the input hidden dimension: "
            f"hidden={hidden_size}, w_a={w_a.shape}, w_b={w_b.shape}"
        )

    out = torch.empty((*a.shape[:-1], hidden_size * 2), dtype=a.dtype, device=a.device)
    if a.numel() == 0:
        return out

    rows = a.numel() // hidden_size
    block_h = triton.next_power_of_2(hidden_size)
    num_warps = 8 if block_h >= 4096 else (4 if block_h >= 1024 else 2)
    _fused_dual_rmsnorm_cat_kernel[(rows, 2)](
        a,
        b,
        w_a,
        w_b,
        out,
        hidden_size,
        float(eps),
        block_h,
        num_warps=num_warps,
    )
    return out
