# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx950 split-K sparse decode that reads mxfp4_mla rows on the scaled MFMA.

QK takes the packed E2M1 codes and their E8M0 scales as they are, against an
MXFP8 query. PxV folds each token's scale into its V codes while widening them
to e4m3, so no row is ever expanded to bf16. Writes the same per-split partials
as the bf16 split-K kernel, so its reduce is shared.

The tile follows the DeepSeek V4.1 MXFP4 decode tile, whose 272-byte record is
this row at a 512-wide latent.
"""

from vllm.triton_utils import tl, triton

NATIVE_LATENT = 512
# tl.constexpr instances, so the kernels can read them as globals.
LATENT = tl.constexpr(NATIVE_LATENT)
# The scaled MFMA reduces 128 K-elements, and PxV reduces tokens in 32-token
# scale blocks, so a tile is 128 tokens.
TILE_K = tl.constexpr(128)
NUM_WARPS = 4
# Pipelining the tile loop doubles its live state and spills it.
NUM_STAGES = 1


@triton.jit
def _mxfp8_scale_bits(amax):
    """Biased E8M0 exponent that puts ``amax`` at the top of the e4m3 range."""
    amax = tl.maximum(amax, 1.1754943508222875e-38)
    bits = tl.ceil(tl.log2(amax / 448.0)) + 127.0
    return tl.minimum(tl.maximum(bits, 0.0), 254.0)


@triton.jit
def _mxfp8_quantize_rows(x, ROWS: tl.constexpr, COLS: tl.constexpr):
    """MXFP8-quantize ``x`` [ROWS, COLS] in registers, one scale per 32 lanes.

    Returns the rescaled fp32 values (to be cast to e4m3) and the
    [ROWS, COLS // 32] biased E8M0 exponents.
    """
    blocks = tl.reshape(x, (ROWS, COLS // 32, 32))
    bits = _mxfp8_scale_bits(tl.max(tl.abs(blocks), axis=2))
    # Multiply by the reciprocal: a divisor of 2**-127 would be subnormal and
    # flush to zero, turning an all-zero block into NaN.
    q = blocks * tl.exp2(127.0 - bits)[:, :, None]
    return tl.reshape(q, (ROWS, COLS)), bits


@triton.jit
def _bytes_of_words(w):
    """The four bytes of each int32 in w, little-endian, on a new last axis."""
    b0 = (w & 255).to(tl.uint8)
    b1 = ((w >> 8) & 255).to(tl.uint8)
    b2 = ((w >> 16) & 255).to(tl.uint8)
    b3 = (w >> 24).to(tl.uint8)
    return tl.join(tl.join(b0, b2), tl.join(b1, b3))


@triton.jit
def _fold_v_scales(k_scale, valid, BLOCK_K: tl.constexpr):
    """Per-token V multipliers relative to each 32-token block's largest scale.

    Returns ``(v_mult [BLOCK_K, 16] fp32, block_max [BLOCK_K // 32, 16] int32)``.
    The fold is exact while a token's scale is within 2**8 of its block's
    maximum, the headroom e4m3 has below 1.0; further below, its V flushes
    toward zero. Masked tokens get code 0 so they cannot raise the maximum.
    """
    codes = tl.where(valid[:, None], k_scale.to(tl.int32), 0)
    codes_blk = tl.reshape(codes, (BLOCK_K // 32, 32, LATENT // 32))
    block_max = tl.max(codes_blk, axis=1)
    shift = tl.reshape(codes_blk - block_max[:, None, :], (BLOCK_K, LATENT // 32))
    v_mult = ((tl.maximum(shift, -126) + 127) << 23).to(tl.float32, bitcast=True)
    return v_mult, block_max


@triton.jit
def _widen_v_128(
    row_ptr, valid, v_mult, block_max, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr
):
    """V for the 128 dims of scale groups 4 * CHUNK to 4 * CHUNK + 3, as e4m3.

    Returns ``(v_fp8 [BLOCK_K, 128], v_scale [128, BLOCK_K // 32])``: the e2m1
    codes widened to e4m3 times v_mult (a power of two per token and scale
    group) by the gfx950 scaled converts, eight values per word, and the
    per-(dim, 32 tokens) E8M0 block maxima the scaled MFMA applies on top.
    The op_sel halves and the byte order of ``_bytes_of_words`` follow from how
    ``tl.join`` interleaves; a wrong pairing still compiles, so the bitwise
    test in ``test_mxfp4_mla_native_decode.py`` guards them.
    """
    words = tl.reshape(
        tl.load(
            row_ptr.to(tl.pointer_type(tl.int32))[:, None, None]
            + CHUNK * 16
            + tl.arange(0, 2)[None, :, None] * 8
            + tl.arange(0, 8)[None, None, :],
            mask=valid[:, None, None],
            other=0,
        ),
        (BLOCK_K, 16),
    )
    # The 4 words of one scale group share its multiplier.
    chunk_sel = (tl.arange(0, 4) == CHUNK)[None, :, None]
    mult_c = tl.sum(
        tl.where(chunk_sel, tl.reshape(v_mult, (BLOCK_K, 4, 4)), 0.0), axis=1
    )
    mult_w = tl.reshape(
        tl.broadcast_to(mult_c[:, :, None], (BLOCK_K, 4, 4)), (BLOCK_K, 16)
    )
    max_c = tl.sum(
        tl.where(chunk_sel, tl.reshape(block_max, (BLOCK_K // 32, 4, 4)), 0), axis=1
    )
    v_scale = tl.reshape(
        tl.broadcast_to(tl.trans(max_c)[:, None, :], (4, 32, BLOCK_K // 32)),
        (128, BLOCK_K // 32),
    ).to(tl.uint8)
    lo, hi, _, _ = tl.inline_asm_elementwise(
        asm="""
        v_cvt_scalef32_pk_f16_fp4 $2, $4, $5
        v_cvt_scalef32_pk_fp8_f16 $0, $2, 1.0
        v_cvt_scalef32_pk_f16_fp4 $2, $4, $5 op_sel:[1,0,0]
        v_cvt_scalef32_pk_fp8_f16 $0, $2, 1.0 op_sel:[0,0,1]
        v_cvt_scalef32_pk_f16_fp4 $3, $4, $5 op_sel:[0,1,0]
        v_cvt_scalef32_pk_fp8_f16 $1, $3, 1.0
        v_cvt_scalef32_pk_f16_fp4 $3, $4, $5 op_sel:[1,1,0]
        v_cvt_scalef32_pk_fp8_f16 $1, $3, 1.0 op_sel:[0,0,1]
        """,
        constraints="=&v,=&v,=&v,=&v,v,v",
        args=[words, mult_w],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=True,
        pack=1,
    )
    v_fp8 = tl.reshape(_bytes_of_words(tl.join(lo, hi)), (BLOCK_K, 128)).to(
        tl.float8e4nv, bitcast=True
    )
    return v_fp8, v_scale


@triton.jit
def _pv_128(
    p_fp8,
    p_scale,
    row_ptr,
    valid,
    v_mult,
    block_max,
    acc,
    CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Add P @ V over the 128 dims of scale groups 4 * CHUNK to 4 * CHUNK + 3."""
    v_fp8, v_scale = _widen_v_128(row_ptr, valid, v_mult, block_max, CHUNK, BLOCK_K)
    return tl.dot_scaled(p_fp8, p_scale, "e4m3", v_fp8, v_scale, "e4m3", acc=acc)


@triton.jit
def _native_tile(
    q_fp8_t,
    q_scale,
    cache_ptr,
    slot,
    valid,
    row_pitch,
    scale_log2,
    head_mask,
    m_i,
    l_i,
    acc_0,
    acc_1,
    acc_2,
    acc_3,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One BLOCK_K-token tile; m_i is in log2 units.

    QK's scales lie along its reduction axis, so the record's scales are used
    as they are. PxV reduces over tokens, but the instruction scales V per
    (dim, 32 tokens), so each token's scale is folded into its V codes relative
    to the largest scale of its 32-token block (exact: both are powers of two),
    and that block maximum becomes V's E8M0 scale. P is at most 1, so 2**8 of
    headroom keeps P inside e4m3.
    """
    row_ptr = cache_ptr + tl.where(valid, slot, 0).to(tl.int64) * row_pitch
    packed = tl.load(
        row_ptr[:, None] + tl.arange(0, LATENT // 2)[None, :],
        mask=valid[:, None],
        other=0,
    )
    # E8M0 255 is NaN and e2m1 has no NaN, so capping the code scrubs it.
    k_scale = tl.minimum(
        tl.load(
            row_ptr[:, None] + LATENT // 2 + tl.arange(0, LATENT // 32)[None, :],
            mask=valid[:, None],
            other=127,
        ),
        254,
    ).to(tl.uint8)

    # Tokens on M, so the warps split the tile's tokens instead of each
    # recomputing all of it from the whole K.
    scores_t = tl.dot_scaled(packed, k_scale, "e2m1", q_fp8_t, q_scale, "e4m3")
    scores_t *= scale_log2
    tile_mask = valid[:, None] & head_mask[None, :]
    scores_t = tl.where(tile_mask, scores_t, -3.4028234663852886e38)
    m_new = tl.maximum(m_i, tl.max(scores_t, axis=0))
    alpha = tl.exp2(m_i - m_new)
    p_t = tl.where(tile_mask, tl.exp2(scores_t - m_new[None, :]), 0.0)
    l_new = l_i * alpha + tl.sum(p_t, axis=0)

    v_mult, block_max = _fold_v_scales(k_scale, valid, BLOCK_K)
    p_fp8 = tl.trans((p_t * 256.0).to(tl.float8e4nv))
    p_scale = tl.full((BLOCK_H, BLOCK_K // 32), 127 - 8, tl.uint8)
    alpha_2d = alpha[:, None]
    acc_0 = _pv_128(
        p_fp8, p_scale, row_ptr, valid, v_mult, block_max, acc_0 * alpha_2d, 0, BLOCK_K
    )
    acc_1 = _pv_128(
        p_fp8, p_scale, row_ptr, valid, v_mult, block_max, acc_1 * alpha_2d, 1, BLOCK_K
    )
    acc_2 = _pv_128(
        p_fp8, p_scale, row_ptr, valid, v_mult, block_max, acc_2 * alpha_2d, 2, BLOCK_K
    )
    acc_3 = _pv_128(
        p_fp8, p_scale, row_ptr, valid, v_mult, block_max, acc_3 * alpha_2d, 3, BLOCK_K
    )
    return m_new, l_new, acc_0, acc_1, acc_2, acc_3


@triton.jit
def mxfp4_mla_decode_partial_kernel(
    q_ptr,
    kv_ptr,
    kv_indices_ptr,
    kv_indptr_ptr,
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    q_stride_t,
    q_stride_h,
    row_pitch,
    pm_stride0,
    pm_stride_s,
    pa_stride0,
    pa_stride_s,
    pa_stride_h,
    num_heads,
    num_kv,
    scale,
    BLOCK_H: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """One (query, split, head block) of the split-K decode over packed rows.

    Same grid and partials as the bf16 split-K kernel: ``m`` in natural-log
    units, and an empty split stores the (-max, 0) sentinel.
    """
    query_idx = tl.program_id(0)
    split_id = tl.program_id(1)
    pid_h = tl.program_id(2)
    BLOCK_K: tl.constexpr = TILE_K

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    dims = tl.arange(0, LATENT)
    q = tl.load(
        q_ptr
        + query_idx * q_stride_t
        + head_offsets[:, None] * q_stride_h
        + dims[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )
    q_val, q_bits = _mxfp8_quantize_rows(q.to(tl.float32), BLOCK_H, LATENT)
    q_fp8_t = tl.trans(q_val.to(tl.float8e4nv))
    q_scale = q_bits.to(tl.uint8)
    scale_log2 = scale * 1.4426950408889634

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_0 = tl.zeros((BLOCK_H, 128), dtype=tl.float32)
    acc_1 = tl.zeros((BLOCK_H, 128), dtype=tl.float32)
    acc_2 = tl.zeros((BLOCK_H, 128), dtype=tl.float32)
    acc_3 = tl.zeros((BLOCK_H, 128), dtype=tl.float32)

    kv_start = tl.load(kv_indptr_ptr + query_idx)
    kv_len = tl.load(kv_indptr_ptr + query_idx + 1) - kv_start
    chunk = (kv_len + NUM_SPLITS - 1) // NUM_SPLITS
    kv_lo = split_id * chunk
    kv_hi = tl.minimum(kv_lo + chunk, kv_len)

    k_offsets = tl.arange(0, BLOCK_K)
    next_slot = tl.load(
        kv_indices_ptr + kv_start + kv_lo + k_offsets,
        mask=kv_lo + k_offsets < kv_hi,
        other=-1,
    )
    for k_start in tl.range(kv_lo, kv_hi, BLOCK_K):
        k_pos = k_start + k_offsets
        slot = next_slot
        next_pos = k_pos + BLOCK_K
        next_slot = tl.load(
            kv_indices_ptr + kv_start + next_pos, mask=next_pos < kv_hi, other=-1
        )
        valid = (k_pos < kv_hi) & (slot >= 0) & (slot < num_kv)
        m_i, l_i, acc_0, acc_1, acc_2, acc_3 = _native_tile(
            q_fp8_t,
            q_scale,
            kv_ptr,
            slot,
            valid,
            row_pitch,
            scale_log2,
            head_mask,
            m_i,
            l_i,
            acc_0,
            acc_1,
            acc_2,
            acc_3,
            BLOCK_H,
            BLOCK_K,
        )

    pm_base = query_idx * pm_stride0 + split_id * pm_stride_s + head_offsets
    m_store = tl.where(l_i > 0.0, m_i * 0.6931471805599453, neg_large)
    tl.store(part_m_ptr + pm_base, m_store, mask=head_mask)
    tl.store(part_l_ptr + pm_base, l_i, mask=head_mask)
    acc_base = (
        part_acc_ptr
        + query_idx * pa_stride0
        + split_id * pa_stride_s
        + head_offsets[:, None] * pa_stride_h
    )
    offsets = tl.arange(0, 128)[None, :]
    tl.store(acc_base + offsets, acc_0, mask=head_mask[:, None])
    tl.store(acc_base + 128 + offsets, acc_1, mask=head_mask[:, None])
    tl.store(acc_base + 256 + offsets, acc_2, mask=head_mask[:, None])
    tl.store(acc_base + 384 + offsets, acc_3, mask=head_mask[:, None])
