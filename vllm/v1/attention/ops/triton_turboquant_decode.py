# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fused TurboQuant decode attention.

Decode path: Triton stage1 (split-KV tiled attention scoring + value
accumulation) + stage2 (log-sum-exp reduction across splits). MSE-key presets
with GQA use a grouped stage 1 (one program per KV head) with tensor-core dots.

Supports FP8 (E4M3) keys, 3-bit and 4-bit uniform quantized values.
"""

import math
from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_FP8_E4B15: dict[int, int] = {}


def _use_fp8_e4b15(device: int = 0) -> int:
    """Return 1 if device needs fp8e4b15 (Ampere/Ada, SM < 8.9), else 0.
    On non-CUDA platforms (e.g. XPU), always returns 0 (use e4nv format).
    """
    if device not in _FP8_E4B15:
        if current_platform.is_cuda_alike():
            cap = torch.cuda.get_device_capability(device)
            _FP8_E4B15[device] = 1 if cap < (8, 9) else 0
        else:
            _FP8_E4B15[device] = 0
    return _FP8_E4B15[device]


# ---------------------------------------------------------------------------
# Stage 1: Fused TQ score + value accumulation (BLOCK_KV tiled)
# ---------------------------------------------------------------------------


@triton.jit
def _tq_decode_stage1(
    # Precomputed query projection
    Q_rot_ptr,  # [B, Hq, D] float32
    # Compressed KV cache (combined K+V)
    KV_cache_ptr,  # [num_blocks, block_size, Hk, padded_slot] uint8
    # Block table and sequence info
    Block_table_ptr,  # [B, max_num_blocks] int32
    Seq_lens_ptr,  # [B] int32
    # TQ parameters
    Centroids_ptr,  # [n_centroids] float32
    # Output (intermediate for stage2)
    Mid_o_ptr,  # [B, Hq, NUM_KV_SPLITS, D+1] float32
    # Strides
    stride_qb,
    stride_qh,  # Q strides: [B, Hq, D]
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,  # KV cache
    stride_bt_b,  # block_table stride per batch
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,  # mid_o strides
    # Constexpr dims
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # KV cache block_size (pages)
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,  # Hq // Hk
    # TQ layout constants
    MSE_BITS: tl.constexpr,  # 3 or 4
    MSE_BYTES: tl.constexpr,  # ceil(D * mse_bits / 8)
    KPS: tl.constexpr,  # key_packed_size
    VQB: tl.constexpr,  # value_quant_bits (4 or 8=FP8)
    VAL_DATA_BYTES: tl.constexpr,  # ceil(D * vqb / 8) or D for FP8
    # Score constants
    ATTN_SCALE: tl.constexpr,  # 1/sqrt(D)
    # Block tile sizes
    BLOCK_D: tl.constexpr,  # next_power_of_2(HEAD_DIM)
    BLOCK_KV: tl.constexpr,  # tokens per tile (16)
    KEY_FP8: tl.constexpr,  # 1 if K is stored as FP8
    NORM_CORRECTION: tl.constexpr = 0,  # 1 = re-normalize centroids
    FP8_E4B15: tl.constexpr = 0,  # 1 = use e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)
):
    bid = tl.program_id(0)  # batch index
    hid = tl.program_id(1)  # q_head index
    sid = tl.program_id(2)  # kv_split index

    kv_head = hid // KV_GROUP_SIZE

    # Sequence length for this batch
    seq_len = tl.load(Seq_lens_ptr + bid)

    # KV split range
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)

    if split_start >= split_end:
        return

    # Dimension offsets
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    # Load query vector: q_rot — [BLOCK_D] float32
    q_base = bid * stride_qb + hid * stride_qh
    q_rot = tl.load(Q_rot_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    # Precompute byte/bit index vectors for MSE gather loads
    if not KEY_FP8:
        mse_bit_off = d_offs * MSE_BITS
        mse_byte_idx = mse_bit_off // 8
        mse_bit_shift = mse_bit_off % 8
        mse_mask = (1 << MSE_BITS) - 1

    # Precompute value bit/byte index vectors (loop-invariant)
    if VQB == 3:
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8

    # Online softmax accumulators
    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    bt_base = bid * stride_bt_b

    # ================================================================
    # TILED LOOP: process BLOCK_KV tokens per iteration
    # ================================================================
    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)

        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        # ============================================================
        # COMPUTE ATTENTION SCORES: [BLOCK_KV]
        # ============================================================
        if KEY_FP8:
            k_addrs = slot_bases[:, None] + d_offs[None, :]
            k_raw = tl.load(
                KV_cache_ptr + k_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            )
            if FP8_E4B15:
                k_float = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
            else:
                k_float = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scores = (
                tl.sum(
                    tl.where(d_mask[None, :], q_rot[None, :] * k_float, 0.0),
                    axis=1,
                )
                * ATTN_SCALE
            )
            scores = tl.where(kv_mask, scores, -float("inf"))
        else:
            # MSE unpack + norms
            mse_addrs0 = slot_bases[:, None] + mse_byte_idx[None, :]
            mse_raw0 = tl.load(
                KV_cache_ptr + mse_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            mse_raw1 = tl.load(
                KV_cache_ptr + mse_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = mse_raw0 | (mse_raw1 << 8)
            mse_idx = (raw16 >> mse_bit_shift[None, :]) & mse_mask

            # Centroid gather + dot product
            c_vals = tl.load(
                Centroids_ptr + mse_idx,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0.0,
            )

            # Norm correction: re-normalize centroid vector to unit norm
            if NORM_CORRECTION:
                c_norm_sq = tl.sum(
                    tl.where(d_mask[None, :], c_vals * c_vals, 0.0),
                    axis=1,
                )
                c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
                c_vals = c_vals * c_inv_norm[:, None]

            term1 = tl.sum(
                tl.where(d_mask[None, :], q_rot[None, :] * c_vals, 0.0),
                axis=1,
            )

            # Load norms (fp16 -> fp32): norms are at MSE_BYTES offset
            norm_bases = slot_bases + MSE_BYTES
            n_lo = tl.load(KV_cache_ptr + norm_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            n_hi = tl.load(KV_cache_ptr + norm_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            vec_norms = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

            scores = vec_norms * term1 * ATTN_SCALE
            scores = tl.where(kv_mask, scores, -float("inf"))

        # ============================================================
        # ONLINE SOFTMAX UPDATE (block-level)
        # ============================================================
        n_e_max = tl.maximum(tl.max(scores, 0), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max)

        # ============================================================
        # VALUE LOAD + DEQUANTIZE: [BLOCK_KV, BLOCK_D]
        # ============================================================
        val_bases = slot_bases + KPS

        if VQB == 3:
            val_addrs0 = val_bases[:, None] + val_byte_idx[None, :]
            val_raw0 = tl.load(
                KV_cache_ptr + val_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            val_raw1 = tl.load(
                KV_cache_ptr + val_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = val_raw0 | (val_raw1 << 8)
            v_idx = ((raw16 >> val_bit_shift[None, :]) & 0x7).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_scales = (
                (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(
                tl.uint16
            )
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = v_idx * v_scales[:, None] + v_zeros[:, None]
        else:  # VQB == 4
            vb_idx = d_offs // 2
            vb_shift = (d_offs % 2) * 4
            val_addrs = val_bases[:, None] + vb_idx[None, :]
            val_raw = tl.load(
                KV_cache_ptr + val_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_scales = (
                (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(
                tl.uint16
            )
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = v_idx * v_scales[:, None] + v_zeros[:, None]

        # ============================================================
        # WEIGHTED VALUE ACCUMULATION
        # ============================================================
        acc = acc * re_scale + tl.sum(p[:, None] * values, 0)
        l_prev = l_prev * re_scale + tl.sum(p, 0)
        m_prev = n_e_max

    # Store partial result
    out_base = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(Mid_o_ptr + out_base + d_offs, acc / safe_l, mask=d_mask)
    lse = m_prev + tl.log(safe_l)
    tl.store(Mid_o_ptr + out_base + HEAD_DIM, lse)


# ---------------------------------------------------------------------------
# Stage 1 for MSE keys, GQA-grouped: one program per (batch, KV head, split).
# Each K/V tile is unpacked once and shared by the KV_GROUP_SIZE query heads
# of the group, and the scores and P @ V run as fp16 tensor-core dots with
# fp32 accumulation. Writes the same mid_o layout as _tq_decode_stage1.
# ---------------------------------------------------------------------------


@triton.jit
def _tq_unpack_tile(
    KV_cache_ptr,
    bases,  # [BLOCK_KV] int64 byte offset of the packed field
    kv_mask,  # [BLOCK_KV]
    BITS: tl.constexpr,  # 3 or 4
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Unpack a [BLOCK_KV, BLOCK_D] tile of LSB-first packed BITS-bit codes
    (element d at bit d * BITS) with contiguous byte loads. Returns int32
    codes in natural d order. Requires BLOCK_D == HEAD_DIM (power of 2, >= 8)."""
    m = kv_mask[:, None]
    if BITS == 4:
        b_offs = tl.arange(0, BLOCK_D // 2)
        raw = tl.load(KV_cache_ptr + bases[:, None] + b_offs[None, :], mask=m, other=0)
        raw = raw.to(tl.int32)
        codes = tl.join(raw & 0xF, raw >> 4)  # [BLOCK_KV, D/2, 2]
    else:  # BITS == 3: 8 codes per 3 bytes
        g_offs = tl.arange(0, BLOCK_D // 8) * 3
        ptr = KV_cache_ptr + bases[:, None] + g_offs[None, :]
        w = (
            tl.load(ptr, mask=m, other=0).to(tl.int32)
            | (tl.load(ptr + 1, mask=m, other=0).to(tl.int32) << 8)
            | (tl.load(ptr + 2, mask=m, other=0).to(tl.int32) << 16)
        )
        # final [.., a, b, c] holds code j = 4a + 2b + c (bits 3j of w)
        f0 = tl.join(
            tl.join(w & 7, (w >> 12) & 7), tl.join((w >> 6) & 7, (w >> 18) & 7)
        )
        f1 = tl.join(
            tl.join((w >> 3) & 7, (w >> 15) & 7), tl.join((w >> 9) & 7, (w >> 21) & 7)
        )
        codes = tl.join(f0, f1)  # [BLOCK_KV, D/8, 2, 2, 2]
    return tl.reshape(codes, [BLOCK_KV, BLOCK_D])


@triton.jit
def _tq_load_f16(KV_cache_ptr, bases, kv_mask):
    """fp16 stored as 2 little-endian bytes at bases -> fp32 [BLOCK_KV]."""
    lo = tl.load(KV_cache_ptr + bases, mask=kv_mask, other=0).to(tl.uint16)
    hi = tl.load(KV_cache_ptr + bases + 1, mask=kv_mask, other=0).to(tl.uint16)
    return (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _tq_fwht(x, ROWS: tl.constexpr, D: tl.constexpr, LOG2_D: tl.constexpr):
    """Unnormalized Walsh-Hadamard transform of each row of x [ROWS, D], i.e.
    x @ H_D with the +-1 Sylvester matrix. Constant-geometry form: every stage
    maps (x[2k], x[2k+1]) to (y[k], y[k + D/2]) = (x[2k] + x[2k+1], x[2k] -
    x[2k+1])."""
    for _ in tl.static_range(LOG2_D):
        a, b = tl.split(tl.reshape(x, [ROWS, D // 2, 2]))
        x = tl.reshape(tl.permute(tl.join(a + b, a - b), [0, 2, 1]), [ROWS, D])
    return x


@triton.jit
def _tq_load_q_group(
    Q_ptr,
    bid,
    kv_head,
    stride_qb,
    stride_qh,
    KV_GROUP_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROT_Q: tl.constexpr,
    LOG2_D: tl.constexpr,
    ROT_SCALE: tl.constexpr,
):
    """Rotated query block of one KV group as fp16 `[BLOCK_H, BLOCK_D]` (padded
    rows are 0) and an fp32 factor to multiply its dot products by. ROT_Q=1:
    Q_ptr is the raw query, rotated here by the Hadamard transform in fp32
    (q @ H / sqrt(D)); else Q_ptr holds q_rot.

    Rows whose |q_rot| can reach 2^15 (|q_rot| <= max|q| * sqrt(D) when rotating
    here) are divided by a power of 2 before the transform so the fp16 cast
    stays finite; the returned per-row factor `[BLOCK_H]` undoes it. Both steps
    are exact, and rows below the bound (all real queries) are unchanged.
    Scaling the raw query (not q_rot) keeps the fp16 operand in registers."""
    h_offs = tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    q_heads = kv_head * KV_GROUP_SIZE + h_offs
    q = tl.load(
        Q_ptr + bid * stride_qb + q_heads[:, None] * stride_qh + d_offs[None, :],
        mask=(h_offs < KV_GROUP_SIZE)[:, None],
        other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(q), 1)
    exponent = ((amax.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 127
    if ROT_Q:
        exponent += (LOG2_D + 1) // 2  # ceil(log2(sqrt(D)))
    shift = tl.maximum(exponent - 14, 0)
    q = q * ((127 - shift) << 23).to(tl.float32, bitcast=True)[:, None]
    if ROT_Q:
        q = _tq_fwht(q, BLOCK_H, BLOCK_D, LOG2_D) * ROT_SCALE
    return q.to(tl.float16), ((127 + shift) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _tq_decode_stage1_mse_gqa(
    Q_ptr,  # [B, Hq, D]: raw query (ROT_Q=1) or float32 q_rot
    KV_cache_ptr,  # [num_blocks, block_size, Hk, padded_slot] uint8
    Block_table_ptr,  # [B, max_num_blocks] int32
    Seq_lens_ptr,  # [B] int32
    Centroids_ptr,  # [n_centroids] float32
    Mid_o_ptr,  # [B, Hq, NUM_KV_SPLITS, D+1] float32
    stride_qb,
    stride_qh,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_bt_b,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    HEAD_DIM: tl.constexpr,  # power of 2
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    MSE_BITS: tl.constexpr,  # 3 or 4
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,  # 3 or 4
    VAL_DATA_BYTES: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    BLOCK_KV: tl.constexpr,  # tokens per tile, >= 16
    BLOCK_H: tl.constexpr,  # padded group size (>= 16, or >= 8 if HEADS_ON_N)
    NORM_CORRECTION: tl.constexpr,
    ROT_Q: tl.constexpr,
    LOG2_D: tl.constexpr,
    ROT_SCALE: tl.constexpr,
    HEADS_ON_N: tl.constexpr,
):
    """HEADS_ON_N=0: scores = q @ K^T ([BLOCK_H, D] x [D, BLOCK_KV]) and
    acc += p @ V, heads on the MMA M dimension (padded to 16).
    HEADS_ON_N=1: the transposed products scores^T = K @ q^T and
    acc^T += V^T @ p^T, heads on N (padded to 8): smaller accumulator and
    query registers, faster for some formats (see _MSE_GQA_LAUNCH).

    All operands are fp16 with fp32 accumulation: q_rot, the centroids and
    p * v_scale are rounded once to fp16; the value codes are exact in fp16;
    value = code * scale + zero is applied as (p * scale) @ code + p @ zero.
    """
    bid = tl.program_id(0)  # batch index
    kv_head = tl.program_id(1)  # kv head index
    sid = tl.program_id(2)  # kv split index

    seq_len = tl.load(Seq_lens_ptr + bid)
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    d_offs = tl.arange(0, HEAD_DIM)
    kv_range = tl.arange(0, BLOCK_KV)
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < KV_GROUP_SIZE
    q_heads = kv_head * KV_GROUP_SIZE + h_offs

    q, q_scale = _tq_load_q_group(
        Q_ptr,
        bid,
        kv_head,
        stride_qb,
        stride_qh,
        KV_GROUP_SIZE,
        BLOCK_H,
        HEAD_DIM,
        ROT_Q,
        LOG2_D,
        ROT_SCALE,
    )
    if HEADS_ON_N:
        qT = tl.trans(q)  # [D, BLOCK_H]
        onesT = tl.full([HEAD_DIM, BLOCK_H], 1.0, dtype=tl.float16)
        acc = tl.zeros([HEAD_DIM, BLOCK_H], dtype=tl.float32)
    else:
        ones = tl.full([BLOCK_H, HEAD_DIM], 1.0, dtype=tl.float16)
        acc = tl.zeros([BLOCK_H, HEAD_DIM], dtype=tl.float32)
    qk_scale = q_scale * ATTN_SCALE  # [BLOCK_H]
    m_prev = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
    l_prev = tl.zeros([BLOCK_H], dtype=tl.float32)

    bt_base = bid * stride_bt_b
    kv_head_off = tl.cast(kv_head, tl.int64) * stride_cache_head

    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end
        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx, mask=kv_mask, other=0
        ).to(tl.int64)
        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + kv_head_off
        )

        # ---- scores (q . c on the raw centroids; per-token norms after) ----
        mse_idx = _tq_unpack_tile(
            KV_cache_ptr, slot_bases, kv_mask, MSE_BITS, BLOCK_KV, HEAD_DIM
        )
        c_vals = tl.load(Centroids_ptr + mse_idx, mask=kv_mask[:, None], other=0.0)
        c16 = c_vals.to(tl.float16)
        vec_norms = _tq_load_f16(KV_cache_ptr, slot_bases + MSE_BYTES, kv_mask)
        if HEADS_ON_N:
            s = vec_norms[:, None] * tl.dot(c16, qT) * qk_scale[None, :]
            if NORM_CORRECTION:
                # |c|^2 per token as a ones-column MMA (lands in score layout)
                nsq = tl.dot((c_vals * c_vals).to(tl.float16), onesT)
                s = s / tl.sqrt(nsq + 1e-16)
            s = tl.where(kv_mask[:, None], s, -float("inf"))
            n_e_max = tl.maximum(tl.max(s, 0), m_prev)
            re_scale = tl.exp(m_prev - n_e_max)
            p = tl.exp(s - n_e_max[None, :])
        else:
            s = vec_norms[None, :] * tl.dot(q, tl.trans(c16)) * qk_scale[:, None]
            if NORM_CORRECTION:
                nsq = tl.dot(ones, tl.trans((c_vals * c_vals).to(tl.float16)))
                s = s / tl.sqrt(nsq + 1e-16)
            s = tl.where(kv_mask[None, :], s, -float("inf"))
            n_e_max = tl.maximum(tl.max(s, 1), m_prev)
            re_scale = tl.exp(m_prev - n_e_max)
            p = tl.exp(s - n_e_max[:, None])

        # ---- values: codes [BLOCK_KV, D], value = code * scale + zero ----
        val_bases = slot_bases + KPS
        v_idx = _tq_unpack_tile(
            KV_cache_ptr, val_bases, kv_mask, VQB, BLOCK_KV, HEAD_DIM
        )
        v_scales = _tq_load_f16(KV_cache_ptr, val_bases + VAL_DATA_BYTES, kv_mask)
        v_zeros = _tq_load_f16(KV_cache_ptr, val_bases + VAL_DATA_BYTES + 2, kv_mask)
        if HEADS_ON_N:
            ps = (p * v_scales[:, None]).to(tl.float16)
            pv = tl.dot(tl.trans(v_idx.to(tl.float16)), ps)  # [D, BLOCK_H]
            pv += tl.sum(p * v_zeros[:, None], 0)[None, :]
            acc = acc * re_scale[None, :] + pv
            l_prev = l_prev * re_scale + tl.sum(p, 0)
        else:
            ps = (p * v_scales[None, :]).to(tl.float16)
            pv = tl.dot(ps, v_idx.to(tl.float16))  # [BLOCK_H, D]
            pv += tl.sum(p * v_zeros[None, :], 1)[:, None]
            acc = acc * re_scale[:, None] + pv
            l_prev = l_prev * re_scale + tl.sum(p, 1)
        m_prev = n_e_max

    out_base = bid * stride_mid_b + sid * stride_mid_s
    head_off = q_heads * stride_mid_h
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    if HEADS_ON_N:
        out = acc / safe_l[None, :]
        out_ptrs = Mid_o_ptr + out_base + head_off[None, :] + d_offs[:, None]
        tl.store(out_ptrs, out, mask=h_mask[None, :])
    else:
        out = acc / safe_l[:, None]
        out_ptrs = Mid_o_ptr + out_base + head_off[:, None] + d_offs[None, :]
        tl.store(out_ptrs, out, mask=h_mask[:, None])
    lse = m_prev + tl.log(safe_l)
    tl.store(Mid_o_ptr + out_base + head_off + HEAD_DIM, lse, mask=h_mask)


# ---------------------------------------------------------------------------
# Stage 2: log-sum-exp merge of the split partials. One program per (batch,
# query head, head_dim chunk). The splits are reduced as [BLOCK_S, BLOCK_DC]
# tiles whose loads are independent, instead of one dependent load chain per
# split, so small batches with many splits are not latency bound.
# ---------------------------------------------------------------------------


@triton.jit
def _tq_decode_stage2(
    Mid_o_ptr,  # [B, Hq, NUM_KV_SPLITS, D+1] float32
    O_ptr,  # [B, Hq, D] query dtype
    Lse_ptr,  # [B, Hq] float32
    Seq_lens_ptr,  # [B] int32
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    stride_ob,
    stride_oh,
    stride_lse_b,
    NUM_KV_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_DC: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    dc = tl.program_id(2)

    seq_len = tl.load(Seq_lens_ptr + bid)
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    base = Mid_o_ptr + bid.to(tl.int64) * stride_mid_b + hid * stride_mid_h
    d_offs = dc * BLOCK_DC + tl.arange(0, BLOCK_DC)
    d_mask = d_offs < HEAD_DIM
    s_range = tl.arange(0, BLOCK_S)

    m = tl.full([], -float("inf"), dtype=tl.float32)
    l_sum = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([BLOCK_DC], dtype=tl.float32)
    for s0 in tl.static_range(0, NUM_KV_SPLITS, BLOCK_S):
        s = s0 + s_range
        # stage 1 wrote split s iff its start (s * split_len) < seq_len
        valid = (s < NUM_KV_SPLITS) & (s * split_len < seq_len)
        lse_s = tl.load(
            base + s * stride_mid_s + HEAD_DIM, mask=valid, other=-float("inf")
        )
        tile = tl.load(
            base + s[:, None] * stride_mid_s + d_offs[None, :],
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        m_new = tl.maximum(m, tl.max(lse_s, 0))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        w = tl.exp(lse_s - m_safe)  # 0 for invalid splits
        alpha = tl.exp(m - m_safe)
        acc = acc * alpha + tl.sum(w[:, None] * tile, 0)
        l_sum = l_sum * alpha + tl.sum(w, 0)
        m = m_new

    safe_l = tl.where(l_sum > 0.0, l_sum, 1.0)
    tl.store(
        O_ptr + bid * stride_ob + hid * stride_oh + d_offs,
        (acc / safe_l).to(O_ptr.dtype.element_ty),
        mask=d_mask,
    )
    if dc == 0:
        tl.store(Lse_ptr + bid * stride_lse_b + hid, m + tl.log(safe_l))


# ---------------------------------------------------------------------------
# Pre-dequant kernel: Bulk dequant K (MSE+norms) and V to fp16
# ---------------------------------------------------------------------------


@triton.jit
def _tq_full_dequant_kv(
    KV_cache_ptr,
    Block_table_ptr,
    Centroids_ptr,
    K_out_ptr,  # [B, Hk, max_seq, D] float16
    V_out_ptr,  # [B, Hk, max_seq, D] float16
    stride_ko_b,
    stride_ko_h,
    stride_ko_s,
    stride_vo_b,
    stride_vo_h,
    stride_vo_s,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_bt_b,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    MSE_BITS: tl.constexpr,
    KEY_FP8: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NORM_CORRECTION: tl.constexpr = 0,
    FP8_E4B15: tl.constexpr = 0,  # 1 = use e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)
):
    """Full dequant: reconstruct K (MSE centroids * norm or FP8) and V to fp16."""
    pos = tl.program_id(0)
    bh = tl.program_id(1)
    bid = bh // NUM_KV_HEADS
    hid = bh % NUM_KV_HEADS

    page_idx = pos // BLOCK_SIZE
    page_off = pos % BLOCK_SIZE
    block_num = tl.load(Block_table_ptr + bid * stride_bt_b + page_idx).to(tl.int64)
    slot_base = (
        block_num * stride_cache_block
        + tl.cast(page_off, tl.int64) * stride_cache_pos
        + tl.cast(hid, tl.int64) * stride_cache_head
    )

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    # === K dequant ===
    ko_base = bid * stride_ko_b + hid * stride_ko_h + pos * stride_ko_s
    if KEY_FP8:
        k_raw = tl.load(KV_cache_ptr + slot_base + d_offs, mask=d_mask, other=0)
        if FP8_E4B15:
            k_recon = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
        else:
            k_recon = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(tl.float16), mask=d_mask)
    else:
        # MSE unpack (3-bit or 4-bit) + norms
        mse_bit_off = d_offs * MSE_BITS
        mse_byte_idx = mse_bit_off // 8
        mse_bit_shift = mse_bit_off % 8
        mse_umask = (1 << MSE_BITS) - 1

        mse_raw0 = tl.load(
            KV_cache_ptr + slot_base + mse_byte_idx, mask=d_mask, other=0
        ).to(tl.int32)
        mse_raw1 = tl.load(
            KV_cache_ptr + slot_base + mse_byte_idx + 1, mask=d_mask, other=0
        ).to(tl.int32)
        raw16_key = mse_raw0 | (mse_raw1 << 8)
        mse_idx = (raw16_key >> mse_bit_shift) & mse_umask

        k_mse = tl.load(Centroids_ptr + mse_idx, mask=d_mask, other=0.0)

        # Norm correction: re-normalize centroid vector to unit norm
        if NORM_CORRECTION:
            c_norm_sq = tl.sum(tl.where(d_mask, k_mse * k_mse, 0.0), axis=0)
            c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
            k_mse = k_mse * c_inv_norm

        # Norms at MSE_BYTES offset (no QJL bytes)
        norm_base = slot_base + MSE_BYTES
        n_lo = tl.load(KV_cache_ptr + norm_base).to(tl.uint16)
        n_hi = tl.load(KV_cache_ptr + norm_base + 1).to(tl.uint16)
        vec_norm = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

        k_recon = vec_norm * k_mse
        tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(tl.float16), mask=d_mask)

    # === V dequant ===
    val_base = slot_base + KPS
    if VQB == 4:
        vb_idx = d_offs // 2
        vb_shift = (d_offs % 2) * 4
        val_raw = tl.load(KV_cache_ptr + val_base + vb_idx, mask=d_mask, other=0).to(
            tl.int32
        )
        v_idx = ((val_raw >> vb_shift) & 0xF).to(tl.float32)

        sc_base = val_base + VAL_DATA_BYTES
        sc_lo = tl.load(KV_cache_ptr + sc_base).to(tl.uint16)
        sc_hi = tl.load(KV_cache_ptr + sc_base + 1).to(tl.uint16)
        v_scale = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        zr_lo = tl.load(KV_cache_ptr + sc_base + 2).to(tl.uint16)
        zr_hi = tl.load(KV_cache_ptr + sc_base + 3).to(tl.uint16)
        v_zero = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        v_vals = v_idx * v_scale + v_zero
    elif VQB == 3:
        # 3-bit value unpack: 8 values per 3 bytes
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8
        val_raw0 = tl.load(
            KV_cache_ptr + val_base + val_byte_idx, mask=d_mask, other=0
        ).to(tl.int32)
        val_raw1 = tl.load(
            KV_cache_ptr + val_base + val_byte_idx + 1, mask=d_mask, other=0
        ).to(tl.int32)
        raw16_val = val_raw0 | (val_raw1 << 8)
        v_idx = ((raw16_val >> val_bit_shift) & 0x7).to(tl.float32)

        sc_base = val_base + VAL_DATA_BYTES
        sc_lo = tl.load(KV_cache_ptr + sc_base).to(tl.uint16)
        sc_hi = tl.load(KV_cache_ptr + sc_base + 1).to(tl.uint16)
        v_scale = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        zr_lo = tl.load(KV_cache_ptr + sc_base + 2).to(tl.uint16)
        zr_hi = tl.load(KV_cache_ptr + sc_base + 3).to(tl.uint16)
        v_zero = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        v_vals = v_idx * v_scale + v_zero
    else:
        v_vals = tl.zeros([BLOCK_D], dtype=tl.float32)

    vo_base = bid * stride_vo_b + hid * stride_vo_h + pos * stride_vo_s
    tl.store(V_out_ptr + vo_base + d_offs, v_vals.to(tl.float16), mask=d_mask)


# ---------------------------------------------------------------------------
# Launcher — cached constants + fused GEMM
# ---------------------------------------------------------------------------

_layout_cache: dict = {}


def _get_layout(D, mse_bits, value_quant_bits, key_packed_size):
    """Get cached layout constants."""
    key = (D, mse_bits, value_quant_bits, key_packed_size)
    cfg = _layout_cache.get(key)
    if cfg is None:
        val_data_bytes = math.ceil(D * value_quant_bits / 8)
        cfg = {
            "mse_bytes": math.ceil(D * mse_bits / 8),
            "val_data_bytes": val_data_bytes,
            "mse_bits": mse_bits,
            "n_centroids": 2**mse_bits,
            "BLOCK_D": triton.next_power_of_2(D),
        }
        _layout_cache[key] = cfg
    return cfg


# Grouped MSE-key stage 1 launch settings, from an H200 sweep (D in {128,
# 256}, batch 1-32, context 4K-128K): (heads on MMA N, BLOCK_KV, num_warps),
# keyed by (D <= 128, MSE key bits, value bits).
_MSE_GQA_LAUNCH: dict[tuple[bool, int, int], tuple[int, int, int]] = {
    (True, 4, 4): (0, 16, 1),  # turboquant_4bit_nc
    (True, 3, 4): (1, 32, 1),  # turboquant_k3v4_nc
    (True, 3, 3): (0, 16, 2),  # turboquant_3bit_nc
}
_MSE_GQA_LAUNCH_D256 = (0, 16, 2)


def _use_mse_gqa(key_fp8: bool, kv_group_size: int, head_dim: int) -> bool:
    """Grouped stage 1 is used on CUDA for MSE keys with GQA at the tuned and
    tested head sizes; everything else runs _tq_decode_stage1. (ROCm gfx950
    uses FlyDSL.)"""
    return (
        current_platform.is_cuda()
        and not key_fp8
        and kv_group_size > 1
        and head_dim in (128, 256)
    )


def triton_turboquant_decode_attention(
    query: torch.Tensor,  # [B, Hq, D] — original query
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, padded_slot] uint8
    block_table: torch.Tensor,  # [B, max_num_blocks] int32
    seq_lens: torch.Tensor,  # [B] int32
    Pi: torch.Tensor,  # [D, D] float32
    centroids: torch.Tensor,  # [n_centroids] float32
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_fp8: bool = False,
    norm_correction: bool = False,
    PiT: torch.Tensor | None = None,  # [D, D] pre-computed Pi.T contiguous
    # Pre-allocated buffers (optional, avoids per-call allocation)
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    buf_holder: Any = None,
    max_num_kv_splits: int = 32,  # fixed split count (must be constant for cudagraph)
    pi_is_hadamard: bool = False,  # Pi is the Sylvester Hadamard / sqrt(D)
) -> torch.Tensor:
    """Launch fused TQ decode attention (Triton stage1 + stage2).

    Returns: output tensor [B, Hq, D] in query's dtype.
    """
    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    device = query.device

    cfg = _get_layout(D, mse_bits, value_quant_bits, key_packed_size)
    grouped = _use_mse_gqa(key_fp8, kv_group_size, D)
    # The grouped kernel rotates the query itself (fast Walsh-Hadamard
    # transform in registers) when Pi is the Sylvester Hadamard.
    rotate_in_kernel = grouped and pi_is_hadamard

    # Compute q_rot = q @ Pi.T (rotated query for MSE key scoring)
    # FP8 path: pass query directly (float16); kernel casts inline.
    # MSE path: external GEMM (cuBLAS), float32, unless rotated in-kernel.
    if key_fp8:
        q_rot = query.contiguous()
    elif rotate_in_kernel:
        q_rot = query if query.stride(-1) == 1 else query.contiguous()
    else:
        q_float = query.float()
        if PiT is None:
            PiT = Pi.T.contiguous()
        q_rot = (q_float @ PiT).contiguous()

    NUM_KV_SPLITS = max_num_kv_splits

    if (
        mid_o_buf is not None
        and mid_o_buf.shape[0] >= B
        and mid_o_buf.shape[2] >= NUM_KV_SPLITS
    ):
        mid_o = mid_o_buf[:B, :Hq, :NUM_KV_SPLITS, :]
    else:
        mid_o = torch.empty(
            B,
            Hq,
            NUM_KV_SPLITS,
            D + 1,
            dtype=torch.float32,
            device=device,
        )
        if buf_holder is not None:
            buf_holder._tq_mid_o_buf = mid_o

    # Stage 1: split-KV tiled attention scoring + value accumulation
    if grouped:
        if D <= 128:
            heads_on_n, block_kv, num_warps = _MSE_GQA_LAUNCH.get(
                (True, mse_bits, value_quant_bits), (0, 16, 1)
            )
        else:
            heads_on_n, block_kv, num_warps = _MSE_GQA_LAUNCH_D256
        block_h = max(8 if heads_on_n else 16, triton.next_power_of_2(kv_group_size))
        _tq_decode_stage1_mse_gqa[(B, Hk, NUM_KV_SPLITS)](
            q_rot,
            kv_cache,
            block_table,
            seq_lens,
            centroids,
            mid_o,
            q_rot.stride(0),
            q_rot.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            KV_GROUP_SIZE=kv_group_size,
            MSE_BITS=mse_bits,
            MSE_BYTES=cfg["mse_bytes"],
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=cfg["val_data_bytes"],
            ATTN_SCALE=scale,
            BLOCK_KV=block_kv,
            BLOCK_H=block_h,
            NORM_CORRECTION=1 if norm_correction else 0,
            ROT_Q=1 if rotate_in_kernel else 0,
            LOG2_D=int(math.log2(D)),
            ROT_SCALE=1.0 / math.sqrt(D),
            HEADS_ON_N=heads_on_n,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        fp8_e4b15 = _use_fp8_e4b15(device.index or 0)
        BLOCK_KV = 4
        grid = (B, Hq, NUM_KV_SPLITS)
        _tq_decode_stage1[grid](
            q_rot,
            kv_cache,
            block_table,
            seq_lens,
            centroids,
            mid_o,
            q_rot.stride(0),
            q_rot.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            NUM_KV_HEADS=Hk,
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            KV_GROUP_SIZE=kv_group_size,
            MSE_BITS=mse_bits,
            MSE_BYTES=cfg["mse_bytes"],
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=cfg["val_data_bytes"],
            ATTN_SCALE=scale,
            BLOCK_D=cfg["BLOCK_D"],
            BLOCK_KV=BLOCK_KV,
            KEY_FP8=1 if key_fp8 else 0,
            NORM_CORRECTION=1 if norm_correction else 0,
            FP8_E4B15=fp8_e4b15,
            num_warps=1,
            num_stages=1,
        )

    # Stage 2: Reduce across KV splits
    # Output in query dtype — eliminates float16_copy kernel after stage2
    out_dtype = query.dtype
    if (
        output_buf is not None
        and output_buf.shape[0] >= B
        and output_buf.dtype == out_dtype
    ):
        output = output_buf[:B, :Hq, :D]
    else:
        output = torch.empty(B, Hq, D, dtype=out_dtype, device=device)
        if buf_holder is not None:
            buf_holder._tq_output_buf = output
    if lse_buf is not None and lse_buf.shape[0] >= B:
        lse = lse_buf[:B, :Hq]
    else:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=device)
        if buf_holder is not None:
            buf_holder._tq_lse_buf = lse

    block_dc = min(32 if B * Hq < 2048 else 64, cfg["BLOCK_D"])
    _tq_decode_stage2[(B, Hq, triton.cdiv(D, block_dc))](
        mid_o,
        output,
        lse,
        seq_lens,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        HEAD_DIM=D,
        BLOCK_S=min(triton.next_power_of_2(NUM_KV_SPLITS), 128),
        BLOCK_DC=block_dc,
        num_warps=2,
        num_stages=1,
    )

    return output  # already in query dtype
