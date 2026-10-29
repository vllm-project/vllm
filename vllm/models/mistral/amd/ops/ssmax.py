# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SSMax query scaling ``q *= s * log(pos + 1) + b``, alone or fused with RoPE.

bf16 only. Every kernel reproduces the unfused rounding: the scale is computed
in bf16 (product and sum each rounded), and RoPE output is rounded to bf16
before it is scaled.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv


@triton.jit
def _round_bf16(x):
    """Round fp32 to the nearest-even bf16 value, kept in fp32.

    Done on the bits: Triton may narrow fp32 math followed by a bf16 cast into
    native bf16 instructions, which do not match torch on ties.
    """
    bits = x.to(tl.uint32, bitcast=True)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _ssmax_scale(log_pos_ptr, s_ptr, b_ptr, token, head, mask):
    """bf16 ``log_pos * s + b`` (each op rounded), returned in fp32.

    Inputs may be fp32 or bf16 (SSMax parameters load in the model dtype).
    """
    log_pos = _round_bf16(tl.load(log_pos_ptr + token, mask=mask).to(tl.float32))
    s = _round_bf16(tl.load(s_ptr + head, mask=mask).to(tl.float32))
    b = _round_bf16(tl.load(b_ptr + head, mask=mask).to(tl.float32))
    return _round_bf16(_round_bf16(log_pos * s) + b)


@triton.jit
def _ssmax_scaling_kernel(
    q_ptr,
    log_pos_ptr,
    s_ptr,
    b_ptr,
    out_ptr,
    q_token_stride,
    q_head_stride,
    numel,
    num_heads: tl.constexpr,
    in_dim: tl.constexpr,
    out_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < numel
    token = offsets // (num_heads * out_dim)
    head = (offsets // out_dim) % num_heads
    dim = offsets % out_dim
    q = tl.load(
        q_ptr + token * q_token_stride + head * q_head_stride + dim,
        mask=mask & (dim < in_dim),
        other=0.0,
    ).to(tl.float32)
    scale = _ssmax_scale(log_pos_ptr, s_ptr, b_ptr, token, head, mask)
    tl.store(out_ptr + offsets, _round_bf16(q * scale).to(tl.bfloat16), mask=mask)


def fused_ssmax_scaling(
    q: torch.Tensor,
    log_positions: torch.Tensor,
    s: torch.Tensor,
    b: torch.Tensor,
    out_head_dim: int | None = None,
) -> torch.Tensor:
    """Return ``F.pad(q, (0, out_head_dim - D)) * (log_positions * s + b)``.

    ``q`` is ``[T, H, D]`` with contiguous head dims; the padding (NoPE layers'
    zero rope dims) is fused into the same launch.
    """
    num_tokens, num_heads, in_dim = q.shape
    out_dim = out_head_dim or in_dim
    assert out_dim >= in_dim and q.stride(-1) == 1 and q.dtype == torch.bfloat16
    assert log_positions.shape == (num_tokens,)
    assert s.shape == b.shape == (num_heads,)
    out = torch.empty((num_tokens, num_heads, out_dim), dtype=q.dtype, device=q.device)
    if out.numel() == 0:
        return out
    block_size = 1024
    _ssmax_scaling_kernel[(cdiv(out.numel(), block_size),)](
        q,
        log_positions,
        s,
        b,
        out,
        q.stride(0),
        q.stride(1),
        out.numel(),
        num_heads,
        in_dim,
        out_dim,
        block_size,
    )
    return out


@triton.jit
def _rope_ssmax_scaling_kernel(
    q_ptr,
    k_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    log_pos_ptr,
    s_ptr,
    b_ptr,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    cache_stride,
    num_heads: tl.constexpr,
    head_block: tl.constexpr,
    nope_dim: tl.constexpr,
    nope_block: tl.constexpr,
    half_rope_dim: tl.constexpr,
    rope_block: tl.constexpr,
    is_neox_style: tl.constexpr,
):
    # One program per token, covering all heads.
    token = tl.program_id(0).to(tl.int64)
    heads = tl.arange(0, head_block)
    head_mask = heads < num_heads
    scale = _ssmax_scale(
        log_pos_ptr, s_ptr, b_ptr, token + heads * 0, heads, head_mask
    )[:, None]
    q_base = q_ptr + token * q_token_stride + heads[:, None] * q_head_stride

    nope = tl.arange(0, nope_block)[None, :]
    nope_mask = head_mask[:, None] & (nope < nope_dim)
    q_nope = tl.load(q_base + nope, mask=nope_mask).to(tl.float32)
    tl.store(q_base + nope, _round_bf16(q_nope * scale).to(tl.bfloat16), mask=nope_mask)

    offs = tl.arange(0, rope_block)
    rope_mask = offs < half_rope_dim
    if is_neox_style:
        first = offs
        second = half_rope_dim + offs
    else:
        first = offs * 2
        second = offs * 2 + 1
    pos = tl.load(positions_ptr + token)
    cache_row = cos_sin_cache_ptr + pos * cache_stride
    cos = tl.load(cache_row + offs, mask=rope_mask).to(tl.float32)
    sin = tl.load(cache_row + half_rope_dim + offs, mask=rope_mask).to(tl.float32)

    q_mask = head_mask[:, None] & rope_mask[None, :]
    q_first = q_base + nope_dim + first[None, :]
    q_second = q_base + nope_dim + second[None, :]
    q0 = tl.load(q_first, mask=q_mask).to(tl.float32)
    q1 = tl.load(q_second, mask=q_mask).to(tl.float32)
    # Explicit fma order: matches the FlashInfer RoPE of the unfused path.
    r0 = _round_bf16(tl.fma(q0, cos[None, :], -(q1 * sin[None, :])))
    r1 = _round_bf16(tl.fma(q1, cos[None, :], q0 * sin[None, :]))
    tl.store(q_first, _round_bf16(r0 * scale).to(tl.bfloat16), mask=q_mask)
    tl.store(q_second, _round_bf16(r1 * scale).to(tl.bfloat16), mask=q_mask)

    k_base = k_ptr + token * k_token_stride
    k0 = tl.load(k_base + first, mask=rope_mask).to(tl.float32)
    k1 = tl.load(k_base + second, mask=rope_mask).to(tl.float32)
    k0_rot = _round_bf16(tl.fma(k0, cos, -(k1 * sin)))
    k1_rot = _round_bf16(tl.fma(k1, cos, k0 * sin))
    tl.store(k_base + first, k0_rot.to(tl.bfloat16), mask=rope_mask)
    tl.store(k_base + second, k1_rot.to(tl.bfloat16), mask=rope_mask)


def fused_rope_ssmax_scaling_(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    log_positions: torch.Tensor,
    s: torch.Tensor,
    b: torch.Tensor,
    nope_dim: int,
    is_neox_style: bool,
) -> None:
    """In place: RoPE on ``q[..., nope_dim:]`` and ``k``, then SSMax on ``q``.

    ``q`` is ``[T, H, nope_dim + rope_dim]``, ``k`` is ``[T, 1, rope_dim]``
    (the shared MLA rope key); token strides are free.
    """
    num_tokens, num_heads, head_dim = q.shape
    rope_dim = head_dim - nope_dim
    assert nope_dim > 0 and rope_dim > 0 and rope_dim % 2 == 0
    assert cos_sin_cache.shape[1] == rope_dim
    assert k.shape == (num_tokens, 1, rope_dim)
    assert q.dtype == k.dtype == torch.bfloat16
    assert q.stride(-1) == 1 and k.stride(-1) == 1
    assert positions.shape == log_positions.shape == (num_tokens,)
    assert s.shape == b.shape == (num_heads,)
    if num_tokens == 0:
        return
    _rope_ssmax_scaling_kernel[(num_tokens,)](
        q,
        k,
        positions,
        cos_sin_cache,
        log_positions,
        s,
        b,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        cos_sin_cache.stride(0),
        num_heads=num_heads,
        head_block=triton.next_power_of_2(num_heads),
        nope_dim=nope_dim,
        nope_block=triton.next_power_of_2(nope_dim),
        half_rope_dim=rope_dim // 2,
        rope_block=triton.next_power_of_2(rope_dim // 2),
        is_neox_style=is_neox_style,
        num_warps=4,
    )
