# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loop message controllers, batched over streams.

A controller blends a fp32 message into a fp32 sublayer input `dest`: from
`h = down(rms([dest, message]))` (bf16), `out = (1 - tanh(forget(h))) * dest +
tanh(gain(h)) * message` in fp32, unfused as the reference. The output feeds
the sublayer's RMSNorm, applied in the same kernel.
"""

import torch

from vllm.triton_utils import tl, tldevice, triton

from .norm import rms_norm, rsqrt


@triton.jit
def _project_blend_kernel(
    Hidden,
    Up,
    Dest,
    Message,
    Blend,
    tokens,
    stride_dest_s,
    D: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    stream = tl.program_id(2).to(tl.int64)
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    rank = tl.arange(0, RANK)
    hidden = tl.load(
        Hidden + (stream * tokens + rows[:, None]) * RANK + rank[None, :],
        rows[:, None] < tokens,
        other=0.0,
    )
    up = Up + (stream * 2 * D + cols[:, None]) * RANK + rank[None, :]
    gain_w = tl.load(up, cols[:, None] < D, other=0.0)
    forget_w = tl.load(up + D * RANK, cols[:, None] < D, other=0.0)
    gain = tldevice.tanh(tl.dot(hidden, tl.trans(gain_w)))
    forget = tldevice.tanh(tl.dot(hidden, tl.trans(forget_w)))
    mask = (rows[:, None] < tokens) & (cols[None, :] < D)
    offset = rows[:, None] * D + cols[None, :]
    dest = tl.load(Dest + stream * stride_dest_s + offset, mask, other=0.0)
    message = tl.load(Message + stream * tokens * D + offset, mask, other=0.0)
    tl.store(
        Blend + stream * tokens * D + offset,
        (1.0 - forget) * dest + gain * message,
        mask,
    )


def project_blend_rms_norm(
    dest: torch.Tensor,
    message: torch.Tensor,
    hidden: torch.Tensor,
    up: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """Apply the rank-64 up projection and gates before storing the FP32 blend."""
    streams, tokens, dim = message.shape
    rank = hidden.shape[-1]
    assert hidden.shape == (streams, tokens, rank) and hidden.is_contiguous()
    assert up.shape == (streams, 2 * dim, rank) and up.is_contiguous()
    assert message.is_contiguous() and dest.shape == message.shape
    assert dest.stride(1) == dim and dest.stride(2) == 1
    blend = torch.empty_like(message)
    block_m, block_n, warps = (
        (64, 128, 8) if streams == 3 and tokens >= 2048 else (32, 64, 4)
    )
    _project_blend_kernel[
        (triton.cdiv(tokens, block_m), triton.cdiv(dim, block_n), streams)
    ](
        hidden,
        up,
        dest,
        message,
        blend,
        tokens,
        dest.stride(0),
        D=dim,
        RANK=rank,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return rms_norm(blend, norm_weight, norm_eps, torch.bfloat16)


@triton.jit
def _pair_rms_norm_kernel(
    Dest,
    Message,
    Out,
    stride_dest_s,
    T,
    eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    dest = tl.load(Dest + s * stride_dest_s + t * D + cols, mask, other=0)
    message = tl.load(Message + (s * T + t) * D + cols, mask, other=0)
    dest_rstd = rsqrt(tl.div_rn(tl.sum(dest * dest), D) + eps)
    message_rstd = rsqrt(tl.div_rn(tl.sum(message * message), D) + eps)
    out = Out + (s * T + t) * (2 * D)
    tl.store(out + cols, (dest * dest_rstd).to(Out.dtype.element_ty), mask)
    tl.store(out + D + cols, (message * message_rstd).to(Out.dtype.element_ty), mask)


def pair_rms_norm(dest: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
    """bf16 `(S, T, 2D)` `[rms(dest), rms(message)]` of fp32 `(S, T, D)`
    inputs, weightless; `dest` may broadcast over streams (stride 0)."""
    num_streams, num_tokens, d = message.shape
    assert dest.shape == message.shape and message.is_contiguous()
    assert dest.stride(1) == d and dest.stride(2) == 1
    assert dest.dtype == message.dtype == torch.float32
    out = message.new_empty((num_streams, num_tokens, 2 * d), dtype=torch.bfloat16)
    if num_tokens:
        _pair_rms_norm_kernel[(num_streams, num_tokens)](
            dest,
            message,
            out,
            dest.stride(0),
            num_tokens,
            1e-6,
            D=d,
            BLOCK=triton.next_power_of_2(d),
            num_warps=8,
        )
    return out


@triton.jit
def _blend_rms_norm_kernel(
    Dest,
    Message,
    GainForget,
    NormW,
    Out,
    stride_dest_s,
    stride_w_s,
    T,
    norm_eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    row = s * T + t
    dest = tl.load(Dest + s * stride_dest_s + t * D + cols, mask, other=0)
    message = tl.load(Message + row * D + cols, mask, other=0)
    gain = tldevice.tanh(tl.load(GainForget + row * (2 * D) + cols, mask, other=0))
    forget = tldevice.tanh(
        tl.load(GainForget + row * (2 * D) + D + cols, mask, other=0)
    )
    x = (1.0 - forget) * dest + gain * message
    rstd = rsqrt(tl.div_rn(tl.sum(x * x), D) + norm_eps)
    y = x * rstd
    y = y * tl.load(NormW + s * stride_w_s + cols, mask, other=0).to(tl.float32)
    tl.store(Out + row * D + cols, y.to(Out.dtype.element_ty), mask)


def blend_rms_norm(
    dest: torch.Tensor,
    message: torch.Tensor,
    gain_forget: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """bf16 `(S, T, D)` RMSNorm (row `s` of `norm_weight` `(S, D)`) of
    `(1 - tanh(forget)) * dest + tanh(gain) * message`, with fp32 `(S, T, D)`
    `dest` (may broadcast over streams) and `message`, and fp32 `(S, T, 2D)`
    `gain_forget` = `[gain, forget]`. `norm_weight` may broadcast too."""
    num_streams, num_tokens, d = message.shape
    assert dest.shape == message.shape and message.is_contiguous()
    assert dest.stride(1) == d and dest.stride(2) == 1
    assert gain_forget.shape == (num_streams, num_tokens, 2 * d)
    assert gain_forget.is_contiguous() and gain_forget.dtype == torch.float32
    assert norm_weight.shape == (num_streams, d) and norm_weight.stride(1) == 1
    out = message.new_empty((num_streams, num_tokens, d), dtype=torch.bfloat16)
    if num_tokens:
        _blend_rms_norm_kernel[(num_streams, num_tokens)](
            dest,
            message,
            gain_forget,
            norm_weight,
            out,
            dest.stride(0),
            norm_weight.stride(0),
            num_tokens,
            norm_eps,
            D=d,
            BLOCK=triton.next_power_of_2(d),
            num_warps=8,
            # The blend is unfused (no FMA), as the reference.
            enable_fp_fusion=False,
        )
    return out
