# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SenseNova Loop routing with FP32 logits and BF16 stream coefficients."""

import torch
from torch import Tensor

from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _mhc_rmsnorm_kernel(
    x,
    out,
    stride_t: tl.constexpr,
    stride_s: tl.constexpr,
    stride_h: tl.constexpr,
    hidden: tl.constexpr,
    streams: tl.constexpr,
    block: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, block)
    values = tl.load(
        x
        + token * stride_t
        + offsets // hidden * stride_s
        + offsets % hidden * stride_h,
        offsets < hidden * streams,
        0,
    ).to(tl.float32)
    inv = tl.rsqrt(tl.sum(values * values, 0) / (hidden * streams) + 1e-6)
    tl.store(
        out + token * hidden * streams + offsets,
        values * inv,
        offsets < hidden * streams,
    )


def mhc_rmsnorm_triton(x: Tensor) -> Tensor:
    """Normalize residual streams into a contiguous FP32 routing input."""
    tokens, streams, hidden = x.shape
    out = torch.empty((tokens, streams * hidden), device=x.device, dtype=torch.float32)
    if tokens:
        block = triton.next_power_of_2(streams * hidden)
        _mhc_rmsnorm_kernel[(tokens,)](
            x,
            out,
            *x.stride(),
            hidden,
            streams,
            block,
            num_warps=4 if block <= 4096 else 8,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _mhc_project_partial_kernel(
    x,
    weight,
    partial,
    width: tl.constexpr,
    outputs: tl.constexpr,
    splits: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    token = row // outputs
    output = row % outputs
    offsets = split * block + tl.arange(0, block)
    a = tl.load(x + token * width + offsets, offsets < width, 0)
    b = tl.load(weight + output * width + offsets, offsets < width, 0)
    value = tl.sum(a * b, 0)
    tl.store(partial + row * splits + split, value)


@triton.jit
def _mhc_project_reduce_kernel(
    partial,
    bias,
    out,
    outputs: tl.constexpr,
    splits: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, block)
    values = tl.load(partial + row * splits + offsets, offsets < splits, 0)
    value = tl.sum(values, 0) + tl.load(bias + row % outputs)
    tl.store(out + row, value)


def mhc_project_triton(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """Split-K FP32 projection for small decode routing matrices."""
    tokens, width = x.shape
    outputs = weight.shape[0]
    block = 2048 if tokens >= 32 else 1024
    splits = triton.cdiv(width, block)
    partial = torch.empty((tokens * outputs, splits), device=x.device, dtype=x.dtype)
    out = torch.empty((tokens, outputs), device=x.device, dtype=x.dtype)
    if tokens:
        _mhc_project_partial_kernel[(tokens * outputs, splits)](
            x,
            weight,
            partial,
            width,
            outputs,
            splits,
            block,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _mhc_project_reduce_kernel[(tokens * outputs,)](
            partial,
            bias,
            out,
            outputs,
            splits,
            triton.next_power_of_2(splits),
            num_warps=1,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _mhc_merge_kernel(
    residual,
    branch,
    post,
    mix,
    out,
    residual_stride_t: tl.constexpr,
    residual_stride_s: tl.constexpr,
    residual_stride_h: tl.constexpr,
    branch_stride_t: tl.constexpr,
    branch_stride_h: tl.constexpr,
    post_stride_t: tl.constexpr,
    post_stride_s: tl.constexpr,
    mix_stride_t: tl.constexpr,
    mix_stride_row: tl.constexpr,
    mix_stride_col: tl.constexpr,
    hidden: tl.constexpr,
    streams: tl.constexpr,
    block: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    offsets = tl.program_id(1) * block + tl.arange(0, block)
    x = tl.load(
        branch + token * branch_stride_t + offsets * branch_stride_h,
        offsets < hidden,
        0,
    ).to(tl.float32)
    for row in tl.static_range(streams):
        mixed = tl.full((block,), 0, tl.float32)
        for col in tl.static_range(streams):
            coefficient = tl.load(
                mix + token * mix_stride_t + row * mix_stride_row + col * mix_stride_col
            )
            coefficient = coefficient.to(tl.bfloat16).to(tl.float32)
            r = tl.load(
                residual
                + token * residual_stride_t
                + col * residual_stride_s
                + offsets * residual_stride_h,
                offsets < hidden,
                0,
            ).to(tl.float32)
            mixed = tl.fma(coefficient, r, mixed)
        p = (
            tl.load(post + token * post_stride_t + row * post_stride_s)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        update = (p * x).to(tl.bfloat16).to(tl.float32)
        value = mixed.to(tl.bfloat16).to(tl.float32) + update
        tl.store(
            out + (token * streams + row) * hidden + offsets, value, offsets < hidden
        )


def mhc_merge_triton(
    residual: Tensor, branch: Tensor, post: Tensor, mix: Tensor
) -> Tensor:
    """Merge output-to-input mixing matrices with BF16 intermediate rounding."""
    tokens, streams, hidden = residual.shape
    if streams != 2:
        # Preserve cuBLAS accumulation order for wider residual streams.
        return mix.to(residual.dtype) @ residual + (
            post.to(branch.dtype).unsqueeze(-1) * branch.unsqueeze(-2)
        )
    out = torch.empty(residual.shape, device=residual.device, dtype=residual.dtype)
    mix_strides = (0, *mix.stride()) if mix.ndim == 2 else mix.stride()
    if tokens:
        _mhc_merge_kernel[(tokens, triton.cdiv(hidden, 512))](
            residual,
            branch,
            post,
            mix,
            out,
            *residual.stride(),
            *branch.stride(),
            *post.stride(),
            *mix_strides,
            hidden,
            streams,
            512,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _mhc_log_sinkhorn_kernel(
    logits,
    output,
    stride_t: tl.constexpr,
    streams: tl.constexpr,
    iterations: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    row = tl.arange(0, streams)
    col = tl.arange(0, streams)
    offsets = row[:, None] * streams + col[None, :]
    mix = tl.load(logits + token * stride_t + offsets)
    for _ in range(iterations):
        maximum = tl.max(mix, 1)
        normalizer = maximum + tldevice.log(
            tl.sum(tldevice.exp(mix - maximum[:, None]), 1)
        )
        mix = mix - normalizer[:, None]
        maximum = tl.max(mix, 0)
        normalizer = maximum + tldevice.log(
            tl.sum(tldevice.exp(mix - maximum[None, :]), 0)
        )
        mix = mix - normalizer[None, :]
    tl.store(output + token * streams * streams + offsets, tldevice.exp(mix))


def mhc_log_sinkhorn_triton(logits: Tensor, iterations: int) -> Tensor:
    """Normalize FP32 mHC logits by alternating log-space row/column sums."""
    assert logits.ndim == 3 and logits.dtype == torch.float32
    tokens, streams, columns = logits.shape
    assert streams in (2, 4) and columns == streams
    assert logits.stride(-1) == 1 and logits.stride(-2) == streams
    assert iterations > 0
    output = torch.empty(logits.shape, device=logits.device, dtype=logits.dtype)
    if tokens:
        _mhc_log_sinkhorn_kernel[(tokens,)](
            logits,
            output,
            logits.stride(0),
            streams,
            iterations,
            num_warps=1,
            enable_fp_fusion=False,
        )
    return output
