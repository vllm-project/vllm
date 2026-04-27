# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused inverse RoPE + block-scaled FP8 quantization kernel for DeepseekV4 attention.

Output scale format is pre-transformed (MN-major TMA-aligned; FP32 on SM90,
INT32-packed UE8M0 on SM100) so fp8_einsum skips transform_sf_into_required_layout.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import use_dsv4_reference_kernels


@triton.jit
def _fused_inv_rope_fp8_quant_per_head(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    fp8_ptr,
    scale_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    fp8_stride_group,
    fp8_stride_token,
    scale_stride_group,
    scale_stride_k,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    TMA_ALIGNED_SCALES: tl.constexpr,
):
    # int64: stride multiply overflows int32 past num_tokens=32768 (IMA).
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh = tl.program_id(1).to(tl.int64)

    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    global_head = pid_gh
    qb_start = head_in_group * CHUNKS_PER_HEAD

    # Padding rows in the TMA-aligned scale buffer: fill with zero and skip quant.
    if pid_token >= num_tokens:
        if TMA_ALIGNED_SCALES:
            scale_addr = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + head_in_group * scale_stride_k
            )
            tl.store(scale_addr, tl.zeros((), dtype=tl.int32))
        else:
            block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
            qb_indices = qb_start + block_offsets
            scale_addrs = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + qb_indices * scale_stride_k
            )
            tl.store(scale_addrs, tl.zeros((CHUNKS_PER_HEAD,), dtype=tl.float32))
        return

    input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(input_base + offsets).to(tl.float32)

    rope_abs_start: tl.constexpr = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START
    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = offsets >= rope_abs_start
    rope_local = offsets - rope_abs_start

    x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(
        tl.float32
    )
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    x_add = x * cos_v + x_partner * sin_v
    x_sub = x * cos_v - x_partner * sin_v
    is_even = (rope_local & 1) == 0
    rotated = tl.where(is_even, x_add, x_sub)
    x = tl.where(is_rope, rotated, x)

    x_2d = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
    block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
    scale_raw = block_absmax * (1.0 / fp8_max)
    scales = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))

    scales_exp = tl.reshape(
        tl.broadcast_to(
            tl.reshape(scales, (CHUNKS_PER_HEAD, 1)),
            (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
        ),
        (HEAD_DIM,),
    )
    x_quant = tl.clamp(x / scales_exp, -fp8_max, fp8_max).to(tl.float8e4nv)

    fp8_base = (
        fp8_ptr
        + g * fp8_stride_group
        + pid_token * fp8_stride_token
        + qb_start * QUANT_GROUP_SIZE
    )
    tl.store(fp8_base + offsets, x_quant)

    block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
    qb_indices = qb_start + block_offsets
    if TMA_ALIGNED_SCALES:
        scale_bits = scales.to(tl.int32, bitcast=True)
        ue8m0_bytes = (scale_bits >> 23) & 0xFF
        packed_val = tl.sum(ue8m0_bytes << (block_offsets * 8))
        scale_addr = (
            scale_ptr
            + g * scale_stride_group
            + pid_token
            + head_in_group * scale_stride_k
        )
        tl.store(scale_addr, packed_val)
    else:
        scale_addrs = (
            scale_ptr + g * scale_stride_group + pid_token + qb_indices * scale_stride_k
        )
        tl.store(scale_addrs, scales)


@triton.jit
def _fused_inv_rope_fp32_quant_per_head(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    out_fp32_ptr,
    scale_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    out_stride_group,
    out_stride_token,
    scale_stride_group,
    scale_stride_k,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    """SM80 variant of `_fused_inv_rope_fp8_quant_per_head`. Writes
    fp32 (x_clamped, divided by per-block UE8M0 scale) instead of fp8.
    Triton on SM80 cannot compile `tl.float8e4nv`; the fp8 cast is done
    in PyTorch in the wrapper. Scales are written as fp32 (no TMA path)."""
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh = tl.program_id(1).to(tl.int64)

    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    global_head = pid_gh
    qb_start = head_in_group * CHUNKS_PER_HEAD

    if pid_token >= num_tokens:
        block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
        qb_indices = qb_start + block_offsets
        scale_addrs = (
            scale_ptr + g * scale_stride_group + pid_token + qb_indices * scale_stride_k
        )
        tl.store(scale_addrs, tl.zeros((CHUNKS_PER_HEAD,), dtype=tl.float32))
        return

    input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(input_base + offsets).to(tl.float32)

    rope_abs_start: tl.constexpr = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START
    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = offsets >= rope_abs_start
    rope_local = offsets - rope_abs_start

    x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(
        tl.float32
    )
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    x_add = x * cos_v + x_partner * sin_v
    x_sub = x * cos_v - x_partner * sin_v
    is_even = (rope_local & 1) == 0
    rotated = tl.where(is_even, x_add, x_sub)
    x = tl.where(is_rope, rotated, x)

    x_2d = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
    block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
    scale_raw = block_absmax * (1.0 / fp8_max)
    scales = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))

    scales_exp = tl.reshape(
        tl.broadcast_to(
            tl.reshape(scales, (CHUNKS_PER_HEAD, 1)),
            (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
        ),
        (HEAD_DIM,),
    )
    x_clamped = tl.clamp(x / scales_exp, -fp8_max, fp8_max)

    out_base = (
        out_fp32_ptr
        + g * out_stride_group
        + pid_token * out_stride_token
        + qb_start * QUANT_GROUP_SIZE
    )
    tl.store(out_base + offsets, x_clamped)

    block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
    qb_indices = qb_start + block_offsets
    scale_addrs = (
        scale_ptr + g * scale_stride_group + pid_token + qb_indices * scale_stride_k
    )
    tl.store(scale_addrs, scales)


def _fused_inv_rope_fp32_quant_triton(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    quant_group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SM80 hybrid path: Triton compute (inverse RoPE + per-block UE8M0
    scale + scaled-and-clamped fp32 output), then the FP8 cast happens in
    PyTorch in the caller. Avoids the per-token PyTorch op chain."""
    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size
    chunks_per_head = head_dim // quant_group_size

    out_fp32 = torch.empty(
        (n_groups, num_tokens, d), dtype=torch.float32, device=o.device
    )
    scales = torch.empty(
        (n_groups, num_tokens, num_scale_blocks),
        dtype=torch.float32,
        device=o.device,
    )

    _fused_inv_rope_fp32_quant_per_head[(num_tokens, n_groups * heads_per_group)](
        o,
        positions,
        cos_sin_cache,
        out_fp32,
        scales,
        num_tokens,
        heads_per_group=heads_per_group,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        out_stride_group=out_fp32.stride(0),
        out_stride_token=out_fp32.stride(1),
        scale_stride_group=scales.stride(0),
        scale_stride_k=scales.stride(2),
        fp8_max=torch.finfo(torch.float8_e4m3fn).max,
        eps=1e-10,
        QUANT_GROUP_SIZE=quant_group_size,
        CHUNKS_PER_HEAD=chunks_per_head,
        ROPE_START=nope_dim % quant_group_size,
        HALF_ROPE=rope_dim // 2,
        num_warps=1,
    )

    return out_fp32, scales


def _fused_inv_rope_fp8_quant_torch(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    quant_group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch fallback for SM80, where Triton can't compile
    `tl.float8e4nv`. Computes inverse GPT-J RoPE on the last `rope_dim`
    dims of each head, then per-block UE8M0 FP8 quantization.

    Returns (o_fp8, o_scale) with the same shapes/strides as the Triton
    kernel's transposed return: o_fp8 = (T, G, H_g*D), o_scale = (T, G, K)
    where K = H_g*D / quant_group_size."""
    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size
    half_rope = rope_dim // 2

    o_view = o.view(num_tokens, n_groups, heads_per_group, head_dim).to(torch.float32)
    rope_part = o_view[..., -rope_dim:].clone()
    even = rope_part[..., 0::2]
    odd = rope_part[..., 1::2]

    cos_sin = cos_sin_cache.index_select(0, positions.to(torch.long))
    cos = cos_sin[..., :half_rope].view(num_tokens, 1, 1, half_rope)
    sin = cos_sin[..., half_rope : 2 * half_rope].view(num_tokens, 1, 1, half_rope)

    new_even = even * cos + odd * sin
    new_odd = odd * cos - even * sin

    rotated = torch.empty_like(rope_part)
    rotated[..., 0::2] = new_even
    rotated[..., 1::2] = new_odd

    o_view[..., -rope_dim:] = rotated
    o_view = o_view.to(torch.bfloat16).to(torch.float32)
    flat = o_view.view(num_tokens, n_groups, num_scale_blocks, quant_group_size)

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    eps = 1e-10
    block_absmax = flat.abs().amax(dim=-1).clamp_min(eps)
    scale_raw = block_absmax / fp8_max
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale_raw)))
    scale_exp = scale.unsqueeze(-1)

    quantized = (flat / scale_exp).clamp(-fp8_max, fp8_max)
    fp8_flat = quantized.to(torch.float8_e4m3fn).view(num_tokens, n_groups, d)
    return fp8_flat, scale.to(torch.float32)


def fused_inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 128,
    tma_aligned_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused inverse RoPE + block-scaled FP8 quantization.

    Args:
        o: Attention output [num_tokens, num_heads, head_dim] bf16.
        positions: Token positions [num_tokens] int64.
        cos_sin_cache: Precomputed [max_pos, rope_dim] with cos||sin.
        n_groups: Number of output groups.
        heads_per_group: Heads per group.
        nope_dim: Non-RoPE dimensions per head (default 448).
        rope_dim: RoPE dimensions per head (default 64).
        quant_group_size: FP8 quantization block size (default 128).
        tma_aligned_scales: Output INT32 packed UE8M0 for SM100 (True)
                            or FP32 for SM90 (False).

    Returns:
        o_fp8: [T, G, D] float8_e4m3fn, strides (D, T*D, 1).
        o_scale: Pre-transformed scale tensor for fp8_einsum.
    """
    if use_dsv4_reference_kernels():
        # SM80 hybrid path: Triton kernel that produces fp32 (inverse RoPE
        # + per-block UE8M0 scaling), then PyTorch performs the fp8 cast
        # (Triton on SM80 cannot compile `tl.float8e4nv`).
        # tma_aligned_scales is ignored on SM80 since TMA is SM90+.
        out_fp32, scales = _fused_inv_rope_fp32_quant_triton(
            o,
            positions,
            cos_sin_cache,
            n_groups,
            heads_per_group,
            nope_dim,
            rope_dim,
            quant_group_size,
        )
        # PyTorch fp8 cast (works on SM80 via software path).
        fp8_buf = out_fp32.to(torch.float8_e4m3fn)
        return fp8_buf.transpose(0, 1), scales.transpose(0, 1)

    from vllm.utils.deep_gemm import get_tma_aligned_size

    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    assert head_dim % quant_group_size == 0
    assert nope_dim % quant_group_size == (quant_group_size - rope_dim)
    assert rope_dim % 2 == 0
    assert cos_sin_cache.shape[-1] == rope_dim
    assert cos_sin_cache.dtype == torch.float32

    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size
    chunks_per_head = head_dim // quant_group_size

    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max

    fp8_buf = torch.empty(
        (n_groups, num_tokens, d),
        dtype=fp8_dtype,
        device=o.device,
    )

    tma_aligned_T = get_tma_aligned_size(num_tokens, 4)
    if tma_aligned_scales:
        packed_sf_k = (num_scale_blocks + 3) // 4
        scale_buf = torch.empty(
            n_groups * packed_sf_k * tma_aligned_T,
            dtype=torch.int32,
            device=o.device,
        ).as_strided(
            (n_groups, num_tokens, packed_sf_k),
            (packed_sf_k * tma_aligned_T, 1, tma_aligned_T),
        )
    else:
        scale_buf = torch.empty(
            n_groups * num_scale_blocks * tma_aligned_T,
            dtype=torch.float32,
            device=o.device,
        ).as_strided(
            (n_groups, num_tokens, num_scale_blocks),
            (num_scale_blocks * tma_aligned_T, 1, tma_aligned_T),
        )

    common_args = dict(
        heads_per_group=heads_per_group,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        fp8_stride_group=fp8_buf.stride(0),
        fp8_stride_token=fp8_buf.stride(1),
        scale_stride_group=scale_buf.stride(0),
        scale_stride_k=scale_buf.stride(2),
        fp8_max=fp8_max,
        eps=1e-10,
        QUANT_GROUP_SIZE=quant_group_size,
        CHUNKS_PER_HEAD=chunks_per_head,
        ROPE_START=nope_dim % quant_group_size,
        HALF_ROPE=rope_dim // 2,
        TMA_ALIGNED_SCALES=tma_aligned_scales,
        num_stages=1,
        launch_pdl=False,
    )

    grid = (tma_aligned_T, n_groups * heads_per_group)
    _fused_inv_rope_fp8_quant_per_head[grid](
        o,
        positions,
        cos_sin_cache,
        fp8_buf,
        scale_buf,
        num_tokens,
        **common_args,
        num_warps=1,
    )

    return fp8_buf.transpose(0, 1), scale_buf.transpose(0, 1)
