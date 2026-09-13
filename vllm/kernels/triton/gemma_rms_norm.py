# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for Gemma-style RMS normalization."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gemma_rms_norm_residual_prelude_kernel(
    x_ptr,
    residual_ptr,
    x_fp32_ptr,
    residual_output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    summed = x + residual
    tl.store(x_fp32_ptr + offsets, summed, mask=mask)
    tl.store(residual_output_ptr + offsets, summed, mask=mask)


@triton.jit
def _gemma_rms_norm_affine_kernel(
    x_fp32_ptr,
    inverse_rms_ptr,
    raw_weight_ptr,
    output_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_offset = row * n_cols

    x = tl.load(x_fp32_ptr + row_offset + offsets, mask=mask, other=0.0)
    inverse_rms = tl.load(inverse_rms_ptr + row)
    raw_weight = tl.load(raw_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    normalized = x * inverse_rms
    output = normalized * (raw_weight + 1.0)
    tl.store(output_ptr + row_offset + offsets, output, mask=mask)


def gemma_rms_norm(
    x: torch.Tensor,
    raw_weight: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply Gemma RMSNorm with a fused FP32 affine tail.

    Keep the native PyTorch FP32 reduction order, then fuse normalization,
    zero-centered weight offset, FP32 affine multiplication, and the output
    cast. This removes full-size FP32 intermediates without changing the
    reduction tree used by ``forward_native``.
    """
    hidden_size = x.shape[-1]
    orig_dtype = x.dtype
    residual_output = None
    if residual is None:
        x_fp32 = x.to(torch.float32)
    else:
        x_fp32 = torch.empty(x.shape, device=x.device, dtype=torch.float32)
        residual_output = torch.empty_like(x)
        block_size = 1024
        grid = (triton.cdiv(x.numel(), block_size),)
        _gemma_rms_norm_residual_prelude_kernel[grid](
            x,
            residual,
            x_fp32,
            residual_output,
            x.numel(),
            BLOCK_SIZE=block_size,
            num_warps=8,
        )

    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    inverse_rms = torch.rsqrt(variance + epsilon)

    x_2d = x_fp32.view(-1, hidden_size)
    output = torch.empty(x_2d.shape, device=x.device, dtype=orig_dtype)
    block_size = triton.next_power_of_2(hidden_size)
    num_warps = 8 if block_size >= 4096 else 4

    _gemma_rms_norm_affine_kernel[(x_2d.shape[0],)](
        x_2d,
        inverse_rms,
        raw_weight,
        output,
        n_cols=hidden_size,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    output = output.view_as(x)
    if residual is None:
        return output
    assert residual_output is not None
    return output, residual_output
