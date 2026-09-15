# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

_MAX_BLOCK_SIZE = 65536


@triton.jit
def _internvl_shuffle_layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    stride_xb,
    stride_xh,
    stride_xw,
    stride_xc,
    eps,
    channels: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    output_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    rows_per_batch = output_height * output_width
    batch = row // rows_per_batch
    spatial = row % rows_per_batch
    output_h = spatial // output_width
    output_w = spatial % output_width

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_dim
    group = offsets // channels
    source_c = offsets % channels
    source_h = 2 * output_h + group // 2
    source_w = 2 * output_w + group % 2
    source_offsets = (
        batch * stride_xb
        + source_h * stride_xh
        + source_w * stride_xw
        + source_c * stride_xc
    )

    values = tl.load(x_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(values, axis=0) / output_dim
    centered = tl.where(mask, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / output_dim
    normalized = centered * tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        output_ptr + row * output_dim + offsets,
        normalized * weight + bias,
        mask=mask,
    )


def internvl_shuffle_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fuse InternVL v2 2x pixel shuffle with projector LayerNorm."""
    if not HAS_TRITON:
        raise RuntimeError("InternVL shuffle LayerNorm requires Triton")
    if x.ndim != 4:
        raise ValueError(f"Expected a 4D input, got shape {tuple(x.shape)}")

    batch, height, width, channels = x.shape
    output_dim = 4 * channels
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"Expected even spatial dimensions, got {(height, width)}")
    if weight.numel() != output_dim or bias.numel() != output_dim:
        raise ValueError(
            f"Expected LayerNorm width {output_dim}, got "
            f"weight={weight.numel()} and bias={bias.numel()}"
        )

    block_size = triton.next_power_of_2(output_dim)
    if block_size > _MAX_BLOCK_SIZE:
        raise ValueError(
            f"LayerNorm width {output_dim} requires unsupported block size {block_size}"
        )

    output_height = height // 2
    output_width = width // 2
    output = torch.empty(
        (batch, output_height, output_width, output_dim),
        dtype=x.dtype,
        device=x.device,
    )
    grid = (batch * output_height * output_width,)
    _internvl_shuffle_layer_norm_kernel[grid](
        x,
        weight,
        bias,
        output,
        *x.stride(),
        eps,
        channels=channels,
        output_height=output_height,
        output_width=output_width,
        output_dim=output_dim,
        BLOCK_SIZE=block_size,
        num_warps=8 if block_size >= 4096 else 4,
    )
    return output


def can_use_internvl_shuffle_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    ps_version: str,
    downsample_ratio: float,
) -> bool:
    """Return whether the fused InternVL projector operation is supported."""
    output_dim = 4 * x.shape[-1] if x.ndim == 4 else 0
    return (
        HAS_TRITON
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == x.dtype
        and bias is not None
        and bias.dtype == x.dtype
        and weight.device == x.device
        and bias.device == x.device
        and weight.is_contiguous()
        and bias.is_contiguous()
        and ps_version == "v2"
        and downsample_ratio == 0.5
        and x.ndim == 4
        and x.shape[1] % 2 == 0
        and x.shape[2] % 2 == 0
        and weight.numel() == output_dim
        and bias.numel() == output_dim
        and triton.next_power_of_2(output_dim) <= _MAX_BLOCK_SIZE
    )
