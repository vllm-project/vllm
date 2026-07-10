# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Oscar-style per-row quantile clip + int2 KV cache pack kernels.

This module hosts the Triton kernels used by the unified HP+int2 KV cache
pool's Oscar rotation path. Rotation itself is done as a bf16 GEMM
(``rows @ R[layer]``) outside these kernels; what lives here is:

* ``_kv_oscar_rotate_k_clip_single_kernel`` -- single fused kernel that loads
  each rotated ``(token, head)`` row for K and V, derives the per-row
  ``clip_ratio`` percentile of ``|row|`` inline via ``tl.sort``, clamps, and
  packs into the int2 cache.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


def _is_power_of_two(n: int) -> bool:
    return (n != 0) and (n & (n - 1)) == 0







def _clip_index(clip_ratio: float, head_dim: int) -> int:
    """Resolve ``clip_ratio`` -> integer sort index, or ``-1`` to disable.

    The kernels skip the in-kernel ``tl.sort`` when ``CLIP_INDEX < 0``.
    """
    if clip_ratio <= 0.0:
        return -1
    idx = int(clip_ratio * head_dim)
    if idx >= head_dim:
        idx = head_dim - 1
    if idx < 0:
        idx = 0
    return idx


def _vectorized_elems_per_thread(dtype: torch.dtype) -> int:
    """Elements per thread for a 128-bit vectorized load.

    bf16/fp16 -> 8 (16 bits each), fp8 -> 16 (8 bits each).
    """
    if dtype in (torch.bfloat16, torch.float16):
        return 8
    if dtype.is_floating_point and dtype.itemsize == 1:
        return 16
    raise AssertionError(
        f"clip int2 kernel requires fp16, bf16 or fp8 input dtype, got {dtype}"
    )


def _pick_block_tok_and_num_warps(
    head_dim: int, elements_per_thread: int
) -> tuple[int, int]:
    """Pick ``(BLOCK_TOK, num_warps)`` so the ``[BLOCK_TOK, head_dim]`` row
    tile distributes exactly ``elements_per_thread`` elements (= 128 bits)
    to every thread along the contiguous dim axis.

    Default target is 4 rows / CTA; bumped up if ``head_dim`` is small so
    the tile fills at least one warp's worth of vectorized loads.
    """
    block_tok = 4
    while block_tok * head_dim < 32 * elements_per_thread:
        block_tok *= 2
    total_elems = block_tok * head_dim
    assert total_elems % (32 * elements_per_thread) == 0, (
        f"BLOCK_TOK={block_tok} head_dim={head_dim} epp={elements_per_thread}: "
        "tile size doesn't divide cleanly into 128-bit/thread loads"
    )
    num_warps = total_elems // (32 * elements_per_thread)
    return block_tok, num_warps



# ---------------------------------------------------------------------------
# Fused rotate (K) + clip + quantize + int2 pack kernel for K and V
# ---------------------------------------------------------------------------
#
# Used when:
#   * rotation mode is "oscar"
#   * V rotation is absorbed into the model weights (caller passes V already in
#     R_v space, so V skips rotation here)
#   * scales/zeros use the per-row single-scale layout
#   * K and V share head_dim (true for MHA configs targeted by oscar mixed-KV)
#
# A single program covers one head and BLOCK_TOK tokens, processing K and V
# back-to-back. K is rotated in-kernel via ``tl.dot(K, R_k)`` (R_k loaded once
# per program); V piggybacks on the same ``tok_offs`` / ``cache_loc`` /
# ``active`` precomputation and goes straight to clip + scale/zero + pack
# (skipping rotation since it's already absorbed). Halves the launch count
# relative to two separate kernels and lets the compiler overlap K/V memory
# traffic.


@triton.jit
def _kv_oscar_rotate_k_clip_single_kernel(
    k_input_ptr,
    v_input_ptr,
    R_ptr,
    loc_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_sz_ptr,
    v_sz_ptr,
    num_tokens,
    num_heads,
    k_input_stride_token,
    k_input_stride_head,
    k_input_stride_dim,
    v_input_stride_token,
    v_input_stride_head,
    v_input_stride_dim,
    R_stride_in,
    R_stride_out,
    k_cache_stride_loc,
    k_cache_stride_head,
    k_cache_stride_dim,
    v_cache_stride_loc,
    v_cache_stride_head,
    v_cache_stride_dim,
    k_sz_stride_loc,
    k_sz_stride_head,
    k_sz_stride_dim,
    v_sz_stride_loc,
    v_sz_stride_head,
    v_sz_stride_dim,
    HEAD_DIM: tl.constexpr,
    BLOCK_QUARTER: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    K_CLIP_INDEX: tl.constexpr,
    V_CLIP_INDEX: tl.constexpr,
    BSEARCH_ITERS: tl.constexpr,
):
    """Single fused kernel: rotate(K) + clip(K) + quant(K) + pack(K), then
    clip(V) + quant(V) + pack(V) sharing the same token tile. K and V share
    HEAD_DIM (asserted by the launcher); V skips rotation because R_v is
    absorbed into the model weights upstream.

    BLOCK_TOK is required to be ``>= 16`` so ``tl.dot`` has a valid M tile.
    The launcher enforces this; the kernel itself just trusts the constexpr.
    """
    pid_tok = tl.program_id(0)
    head_idx = tl.program_id(1)
    if head_idx >= num_heads:
        return

    tok_offs = pid_tok * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    tok_mask = tok_offs < num_tokens

    cache_loc = tl.load(loc_ptr + tok_offs, mask=tok_mask, other=0)
    valid_loc = cache_loc >= 0
    active = tok_mask & valid_loc

    full_offs = tl.arange(0, HEAD_DIM)
    dim_offs_q = tl.arange(0, BLOCK_QUARTER)

    # ---------- K: load + rotate + clip + scale/zero + pack + write ----------
    k_base = (
        tok_offs[:, None] * k_input_stride_token
        + head_idx * k_input_stride_head
        + full_offs[None, :] * k_input_stride_dim
    )
    k_tile = tl.load(
        k_input_ptr + k_base,
        mask=tok_mask[:, None],
        other=0.0,
    )

    r_in = tl.arange(0, HEAD_DIM)
    r_out = tl.arange(0, HEAD_DIM)
    R_offs = r_in[:, None] * R_stride_in + r_out[None, :] * R_stride_out
    R_tile = tl.load(R_ptr + R_offs)

    k_rows = tl.dot(k_tile, R_tile, out_dtype=tl.float32)

    if K_CLIP_INDEX >= 0:
        abs_rows = tl.abs(k_rows)
        if BSEARCH_ITERS > 0:
            target_above = HEAD_DIM - K_CLIP_INDEX
            thr_lo = tl.zeros([BLOCK_TOK], dtype=tl.float32)
            thr_hi = tl.max(abs_rows, axis=1)
            for _ in tl.static_range(BSEARCH_ITERS):
                thr_mid = (thr_lo + thr_hi) * 0.5
                cnt_above = tl.sum((abs_rows > thr_mid[:, None]).to(tl.int32), axis=1)
                too_many = cnt_above > target_above
                thr_lo = tl.where(too_many, thr_mid, thr_lo)
                thr_hi = tl.where(too_many, thr_hi, thr_mid)
            thr = thr_hi
        else:
            sorted_rows = tl.sort(abs_rows)
            pick = (full_offs == K_CLIP_INDEX)[None, :]
            thr = tl.sum(tl.where(pick, sorted_rows, 0.0), axis=1)
        k_rows = tl.minimum(
            tl.maximum(k_rows, -thr[:, None]),
            thr[:, None],
        )

    k_min = tl.min(k_rows, axis=1)
    k_max = tl.max(k_rows, axis=1)
    k_range = tl.maximum(k_max - k_min, 1e-8)
    k_scale = k_range / 3.0
    k_zero = -k_min / k_scale

    k_r = tl.reshape(k_rows, (BLOCK_TOK, 4, BLOCK_QUARTER))
    k_p = tl.permute(k_r, (0, 2, 1))
    k_s = tl.reshape(k_p, (BLOCK_TOK, BLOCK_QUARTER, 2, 2))
    k_even, k_odd = tl.split(k_s)
    k_v0, k_v2 = tl.split(k_even)
    k_v1, k_v3 = tl.split(k_odd)
    k_q0 = (k_v0 / k_scale[:, None] + k_zero[:, None] + 0.5).to(tl.uint8)
    k_q1 = (k_v1 / k_scale[:, None] + k_zero[:, None] + 0.5).to(tl.uint8)
    k_q2 = (k_v2 / k_scale[:, None] + k_zero[:, None] + 0.5).to(tl.uint8)
    k_q3 = (k_v3 / k_scale[:, None] + k_zero[:, None] + 0.5).to(tl.uint8)
    k_packed = k_q0 | (k_q1 << 2) | (k_q2 << 4) | (k_q3 << 6)

    k_cache_offset = (
        cache_loc[:, None] * k_cache_stride_loc
        + head_idx * k_cache_stride_head
        + dim_offs_q[None, :] * k_cache_stride_dim
    )
    tl.store(k_cache_ptr + k_cache_offset, k_packed, mask=active[:, None])

    k_sz_base = cache_loc * k_sz_stride_loc + head_idx * k_sz_stride_head
    tl.store(k_sz_ptr + k_sz_base + 0 * k_sz_stride_dim, k_scale, mask=active)
    tl.store(k_sz_ptr + k_sz_base + 1 * k_sz_stride_dim, k_zero, mask=active)

    # ---------- V: load + clip + scale/zero + pack + write (no rotate) ------
    v_base = (
        tok_offs[:, None] * v_input_stride_token
        + head_idx * v_input_stride_head
        + full_offs[None, :] * v_input_stride_dim
    )
    v_rows = tl.load(
        v_input_ptr + v_base,
        mask=tok_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    if V_CLIP_INDEX >= 0:
        abs_rows = tl.abs(v_rows)
        if BSEARCH_ITERS > 0:
            target_above = HEAD_DIM - V_CLIP_INDEX
            thr_lo = tl.zeros([BLOCK_TOK], dtype=tl.float32)
            thr_hi = tl.max(abs_rows, axis=1)
            for _ in tl.static_range(BSEARCH_ITERS):
                thr_mid = (thr_lo + thr_hi) * 0.5
                cnt_above = tl.sum((abs_rows > thr_mid[:, None]).to(tl.int32), axis=1)
                too_many = cnt_above > target_above
                thr_lo = tl.where(too_many, thr_mid, thr_lo)
                thr_hi = tl.where(too_many, thr_hi, thr_mid)
            thr = thr_hi
        else:
            sorted_rows = tl.sort(abs_rows)
            pick = (full_offs == V_CLIP_INDEX)[None, :]
            thr = tl.sum(tl.where(pick, sorted_rows, 0.0), axis=1)
        v_rows = tl.minimum(
            tl.maximum(v_rows, -thr[:, None]),
            thr[:, None],
        )

    v_min = tl.min(v_rows, axis=1)
    v_max = tl.max(v_rows, axis=1)
    v_range = tl.maximum(v_max - v_min, 1e-8)
    v_scale = v_range / 3.0
    v_zero = -v_min / v_scale

    v_r = tl.reshape(v_rows, (BLOCK_TOK, 4, BLOCK_QUARTER))
    v_p = tl.permute(v_r, (0, 2, 1))
    v_s = tl.reshape(v_p, (BLOCK_TOK, BLOCK_QUARTER, 2, 2))
    v_even, v_odd = tl.split(v_s)
    v_v0, v_v2 = tl.split(v_even)
    v_v1, v_v3 = tl.split(v_odd)
    v_q0 = (v_v0 / v_scale[:, None] + v_zero[:, None] + 0.5).to(tl.uint8)
    v_q1 = (v_v1 / v_scale[:, None] + v_zero[:, None] + 0.5).to(tl.uint8)
    v_q2 = (v_v2 / v_scale[:, None] + v_zero[:, None] + 0.5).to(tl.uint8)
    v_q3 = (v_v3 / v_scale[:, None] + v_zero[:, None] + 0.5).to(tl.uint8)
    v_packed = v_q0 | (v_q1 << 2) | (v_q2 << 4) | (v_q3 << 6)

    v_cache_offset = (
        cache_loc[:, None] * v_cache_stride_loc
        + head_idx * v_cache_stride_head
        + dim_offs_q[None, :] * v_cache_stride_dim
    )
    tl.store(v_cache_ptr + v_cache_offset, v_packed, mask=active[:, None])

    v_sz_base = cache_loc * v_sz_stride_loc + head_idx * v_sz_stride_head
    tl.store(v_sz_ptr + v_sz_base + 0 * v_sz_stride_dim, v_scale, mask=active)
    tl.store(v_sz_ptr + v_sz_base + 1 * v_sz_stride_dim, v_zero, mask=active)


def _pick_block_tok_and_num_warps_for_dot(
    head_dim: int, elements_per_thread: int
) -> tuple[int, int]:
    """Same target as :func:`_pick_block_tok_and_num_warps` but enforces
    ``BLOCK_TOK >= 16`` so ``tl.dot`` has a valid M dimension.
    """
    block_tok, num_warps = _pick_block_tok_and_num_warps(head_dim, elements_per_thread)
    if block_tok < 16:
        block_tok = 16
        total_elems = block_tok * head_dim
        assert total_elems % (32 * elements_per_thread) == 0, (
            f"BLOCK_TOK={block_tok} head_dim={head_dim} epp={elements_per_thread}: "
            "tile size doesn't divide cleanly into 128-bit/thread loads"
        )
        num_warps = total_elems // (32 * elements_per_thread)
    return block_tok, num_warps


def quantized_set_kv_int2_oscar_rotate_k_clip_triton(
    cache_k_unrotated: torch.Tensor,
    cache_v_rotated: torch.Tensor,
    R_k: torch.Tensor,
    loc: torch.Tensor,
    k_cache_buffer: torch.Tensor,
    v_cache_buffer: torch.Tensor,
    k_scales_zeros_buffer: torch.Tensor,
    v_scales_zeros_buffer: torch.Tensor,
    clip_ratio_k: float,
    clip_ratio_v: float,
) -> None:
    """Single-launch fused oscar K-rotation + clip + quantize + int2 pack for
    both K and V. V skips the rotation (caller must have absorbed R_v into the
    model weights).

    Per-program flow (one head, BLOCK_TOK tokens):
        K: load tile → ``K @ R_k`` (``tl.dot``) → per-row sort+clamp clip
           (skipped when ``clip_ratio_k <= 0``) → per-row scale/zero
           → quartered split → int2 pack → write quant slot + scale/zero.
        V: load tile (already in R_v space) → per-row sort+clamp clip
           → per-row scale/zero → quartered split → int2 pack → write.

    Restrictions enforced here:
        * ``cache_k_unrotated`` / ``cache_v_rotated`` share ``(num_tokens,
          num_heads, head_dim)``. Same head_dim is required because the kernel
          uses one ``HEAD_DIM`` constexpr for both halves.
        * Per-row (single-scale) scales/zeros layout for both K and V.
        * Power-of-two ``head_dim``.
        * ``R_k.dtype == cache_k_unrotated.dtype``.

    Pass ``clip_ratio_k = 0.0`` (or ``clip_ratio_v = 0.0``) to disable
    clipping for K (or V); the in-kernel sort is then skipped.
    """
    assert cache_k_unrotated.shape == cache_v_rotated.shape, (
        "K/V shape mismatch in oscar_rotate_k_clip (kernel requires identical "
        f"shapes incl. head_dim): {cache_k_unrotated.shape} vs {cache_v_rotated.shape}"
    )
    num_tokens, num_heads, head_dim = cache_k_unrotated.shape
    if num_tokens == 0:
        return

    assert head_dim % 4 == 0, (
        f"head_dim must be divisible by 4 for INT2, got {head_dim}"
    )
    assert _is_power_of_two(head_dim), (
        f"oscar rotate+clip int2 kernel requires power-of-two head_dim, got {head_dim}"
    )
    assert R_k.shape == (head_dim, head_dim), (
        f"R_k must be [head_dim, head_dim]={head_dim}x{head_dim}, "
        f"got {tuple(R_k.shape)}"
    )
    assert R_k.dtype == cache_k_unrotated.dtype, (
        f"R_k dtype ({R_k.dtype}) must match input dtype ({cache_k_unrotated.dtype})"
    )



    elements_per_thread = _vectorized_elems_per_thread(cache_k_unrotated.dtype)
    block_tok, num_warps = _pick_block_tok_and_num_warps_for_dot(
        head_dim, elements_per_thread
    )
    grid = (triton.cdiv(num_tokens, block_tok), num_heads)
    _kv_oscar_rotate_k_clip_single_kernel[grid](
        cache_k_unrotated,
        cache_v_rotated,
        R_k,
        loc,
        k_cache_buffer,
        v_cache_buffer,
        k_scales_zeros_buffer,
        v_scales_zeros_buffer,
        num_tokens,
        num_heads,
        cache_k_unrotated.stride(0),
        cache_k_unrotated.stride(1),
        cache_k_unrotated.stride(2),
        cache_v_rotated.stride(0),
        cache_v_rotated.stride(1),
        cache_v_rotated.stride(2),
        R_k.stride(0),
        R_k.stride(1),
        k_cache_buffer.stride(0),
        k_cache_buffer.stride(1),
        k_cache_buffer.stride(2),
        v_cache_buffer.stride(0),
        v_cache_buffer.stride(1),
        v_cache_buffer.stride(2),
        k_scales_zeros_buffer.stride(0),
        k_scales_zeros_buffer.stride(1),
        k_scales_zeros_buffer.stride(2),
        v_scales_zeros_buffer.stride(0),
        v_scales_zeros_buffer.stride(1),
        v_scales_zeros_buffer.stride(2),

        HEAD_DIM=head_dim,
        BLOCK_QUARTER=head_dim // 4,
        BLOCK_TOK=block_tok,
        K_CLIP_INDEX=_clip_index(clip_ratio_k, head_dim),
        V_CLIP_INDEX=_clip_index(clip_ratio_v, head_dim),
        BSEARCH_ITERS=(head_dim.bit_length() - 1) if head_dim >= 64 else 0,
        num_warps=num_warps,
        num_stages=1,
    )
