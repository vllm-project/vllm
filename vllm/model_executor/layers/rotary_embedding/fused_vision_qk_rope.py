# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse packed vision QK materialization with full NeoX rotary embedding."""

import torch

from vllm.triton_utils import tl, triton


def _supported_layout(x, cos, sin, num_heads: int, head_dim: int) -> bool:
    """Inspect tensor metadata only; device admission is checked separately."""
    return (
        num_heads == 16
        and head_dim == 72
        and x.ndim == 3
        and x.shape[0] > 0
        and x.shape[1] in (1, 2)
        and x.shape[2] == 3 * num_heads * head_dim
        and x.is_contiguous()
        and x.numel() < 2**63
        and x.shape[0] * num_heads * (head_dim // 2) < 2**31
        and x.dtype == cos.dtype == sin.dtype == torch.bfloat16
        and cos.ndim == sin.ndim == 2
        and cos.shape == sin.shape
        and cos.shape[0] >= x.shape[0]
        and cos.shape[1] == head_dim // 2
        and x.device == cos.device == sin.device
    )


@triton.jit(do_not_specialize=["SEQ"])
def _pack_rotary_kernel(
    X,
    C,
    S,
    OUT,
    SEQ,
    BATCH: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDE_INDEX: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if WIDE_INDEX:
        i = i.to(tl.int64)
    half = D // 2
    dim = i % half
    head = (i // half) % H
    position = i // (half * H)
    qb = tl.program_id(1)
    if WIDE_INDEX:
        qb = qb.to(tl.int64)
    batch = qb % BATCH
    qk = qb // BATCH
    mask = i < SEQ * H * half
    xoff = ((position * BATCH + batch) * 3 * H + qk * H + head) * D + dim
    ooff = (qb * SEQ * H + i // half) * D + dim
    x0 = tl.load(X + xoff, mask, other=0).to(tl.float32)
    x1 = tl.load(X + xoff + half, mask, other=0).to(tl.float32)
    c = tl.load(C + position * CS0 + dim * CS1, mask, other=1).to(tl.float32)
    s = tl.load(S + position * SS0 + dim * SS1, mask, other=0).to(tl.float32)
    o0 = x0 * c - x1 * s
    o1 = x0 * s + x1 * c
    tl.store(OUT + ooff, o0, mask)
    tl.store(OUT + ooff + half, o1, mask)


def try_fused_vision_qk_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    head_dim: int,
    *,
    is_neox_style: bool,
    enable_fp32_compute: bool,
) -> torch.Tensor | None:
    """Return packed rotated QK, or retain the caller's original fallback.

    V remains a view of the original QKV projection. Rotation arithmetic uses
    FP32 intermediates, matching the native FlashAttention rotary kernel.
    Sequence length is dynamic so different images do not compile new kernels.
    """
    if (
        not x.is_cuda
        or not is_neox_style
        or enable_fp32_compute
        or not _supported_layout(x, cos, sin, num_heads, head_dim)
    ):
        return None
    seq, batch, _ = x.shape
    output = torch.empty(
        (2 * batch, seq, num_heads, head_dim), dtype=x.dtype, device=x.device
    )
    _pack_rotary_kernel[
        (triton.cdiv(seq * num_heads * (head_dim // 2), 1024), 2 * batch)
    ](
        x,
        cos,
        sin,
        output,
        seq,
        batch,
        num_heads,
        head_dim,
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        BLOCK=1024,
        WIDE_INDEX=x.numel() >= 2**31,
        num_warps=4,
    )
    return output
