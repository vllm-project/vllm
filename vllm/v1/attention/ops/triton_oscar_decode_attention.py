# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fused OSCAR INT2 decode attention.

Decode path: Triton stage1 (split-KV tiled attention scoring + value
accumulation with BLOCK_H head grouping for GQA) + stage2 (log-sum-exp
reduction across splits).

Supports 2-bit quantization for both K and V cache using OSCAR layout.
Uses tl.dot for Tensor Core acceleration on score and value accumulation.
"""

from typing import Any

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import (
    _fwd_kernel_stage2,
)

# ---------------------------------------------------------------------------
# Stage 1: Fused OSCAR score + value accumulation (BLOCK_H grouped, tl.dot)
# ---------------------------------------------------------------------------


@triton.jit
def _oscar_decode_stage1(
    # Precomputed query projection (already rotated!)
    Q_rot_ptr,  # [B, Hq, D] float32
    # Compressed KV cache (combined K+V)
    KV_cache_ptr,  # [num_blocks, block_size, Hk, padded_slot] uint8
    # Block table and sequence info
    Block_table_ptr,  # [B, max_num_blocks] int32
    Seq_lens_ptr,  # [B] int32
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
    BLOCK_QUARTER: tl.constexpr,  # HEAD_DIM // 4
    K_START: tl.constexpr,
    K_SZ_START: tl.constexpr,
    V_START: tl.constexpr,
    V_SZ_START: tl.constexpr,
    # Score constants
    ATTN_SCALE: tl.constexpr,  # 1/sqrt(D)
    # Block tile sizes
    BLOCK_KV: tl.constexpr,  # tokens per tile (64)
    BLOCK_H: tl.constexpr,  # heads per program (16)
    Q_HEAD_NUM: tl.constexpr,  # total query heads
):
    bid = tl.program_id(0)  # batch index
    hid_group = tl.program_id(1)  # head group index
    sid = tl.program_id(2)  # kv_split index

    # Head grouping: process VALID_BLOCK_H q-heads sharing 1 KV head.
    # BLOCK_H is always a power-of-2 for Tensor Core tile alignment;
    # mask_h disables padded lanes when KV_GROUP_SIZE < BLOCK_H.
    VALID_BLOCK_H: tl.constexpr = BLOCK_H if KV_GROUP_SIZE > BLOCK_H else KV_GROUP_SIZE
    cur_head = hid_group * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (hid_group + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < Q_HEAD_NUM)

    cur_kv_head = hid_group // tl.cdiv(KV_GROUP_SIZE, BLOCK_H)

    # Sequence length for this batch
    seq_len = tl.load(Seq_lens_ptr + bid)

    # KV split range
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)

    if split_start >= split_end:
        return

    dim_offs_q = tl.arange(0, BLOCK_QUARTER)
    kv_range = tl.arange(0, BLOCK_KV)

    # Load query vectors: [BLOCK_H, BLOCK_QUARTER] × 4 quarters
    q_base = bid * stride_qb + cur_head * stride_qh  # [BLOCK_H]

    q0 = tl.load(
        Q_rot_ptr + q_base[:, None] + dim_offs_q[None, :],
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.float32)
    q1 = tl.load(
        Q_rot_ptr + q_base[:, None] + (dim_offs_q[None, :] + BLOCK_QUARTER),
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.float32)
    q2 = tl.load(
        Q_rot_ptr + q_base[:, None] + (dim_offs_q[None, :] + 2 * BLOCK_QUARTER),
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.float32)
    q3 = tl.load(
        Q_rot_ptr + q_base[:, None] + (dim_offs_q[None, :] + 3 * BLOCK_QUARTER),
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.float32)

    # Online softmax accumulators: [BLOCK_H] scalars, [BLOCK_H, BQ] values
    m_prev = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
    l_prev = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc0 = tl.zeros([BLOCK_H, BLOCK_QUARTER], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_H, BLOCK_QUARTER], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_H, BLOCK_QUARTER], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_H, BLOCK_QUARTER], dtype=tl.float32)

    bt_base = bid * stride_bt_b

    # ================================================================
    # TILED LOOP: process BLOCK_KV tokens per iteration
    # KV data loaded once, shared across all BLOCK_H query heads.
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
            + tl.cast(cur_kv_head, tl.int64) * stride_cache_head
        )

        # ============================================================
        # LOAD + DEQUANTIZE K: [BLOCK_KV, BLOCK_QUARTER] × 4 quarters
        # ============================================================
        k_addrs = slot_bases[:, None] + K_START + dim_offs_q[None, :]
        k_packed = tl.load(KV_cache_ptr + k_addrs, mask=kv_mask[:, None], other=0).to(
            tl.int32
        )

        k_scale_addrs = slot_bases + K_SZ_START
        sc_lo = tl.load(KV_cache_ptr + k_scale_addrs, mask=kv_mask, other=0).to(
            tl.uint16
        )
        sc_hi = tl.load(KV_cache_ptr + k_scale_addrs + 1, mask=kv_mask, other=0).to(
            tl.uint16
        )
        k_scale = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

        zr_lo = tl.load(KV_cache_ptr + k_scale_addrs + 2, mask=kv_mask, other=0).to(
            tl.uint16
        )
        zr_hi = tl.load(KV_cache_ptr + k_scale_addrs + 3, mask=kv_mask, other=0).to(
            tl.uint16
        )
        k_zero = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

        k_q0 = (k_packed & 0x3).to(tl.float32)
        k_q1 = ((k_packed >> 2) & 0x3).to(tl.float32)
        k_q2 = ((k_packed >> 4) & 0x3).to(tl.float32)
        k_q3 = ((k_packed >> 6) & 0x3).to(tl.float32)

        # Dequant: [BLOCK_KV, BLOCK_QUARTER]
        k_v0 = (k_q0 - k_zero[:, None]) * k_scale[:, None]
        k_v1 = (k_q1 - k_zero[:, None]) * k_scale[:, None]
        k_v2 = (k_q2 - k_zero[:, None]) * k_scale[:, None]
        k_v3 = (k_q3 - k_zero[:, None]) * k_scale[:, None]

        # ============================================================
        # COMPUTE SCORES via tl.dot: [BLOCK_H, BLOCK_KV]
        # 4 quarter-width matmuls summed together.
        # q: [BLOCK_H, BQ], k^T: [BQ, BLOCK_KV] → [BLOCK_H, BLOCK_KV]
        # ============================================================
        k_v0_f16 = k_v0.to(tl.float16)
        k_v1_f16 = k_v1.to(tl.float16)
        k_v2_f16 = k_v2.to(tl.float16)
        k_v3_f16 = k_v3.to(tl.float16)

        scores = tl.dot(q0.to(tl.float16), tl.trans(k_v0_f16))
        scores += tl.dot(q1.to(tl.float16), tl.trans(k_v1_f16))
        scores += tl.dot(q2.to(tl.float16), tl.trans(k_v2_f16))
        scores += tl.dot(q3.to(tl.float16), tl.trans(k_v3_f16))
        scores = scores.to(tl.float32) * ATTN_SCALE

        scores = tl.where(mask_h[:, None] & kv_mask[None, :], scores, -float("inf"))

        # ============================================================
        # ONLINE SOFTMAX UPDATE (block-level): [BLOCK_H]
        # ============================================================
        n_e_max = tl.maximum(tl.max(scores, 1), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max[:, None])  # [BLOCK_H, BLOCK_KV]

        # ============================================================
        # VALUE LOAD + DEQUANT: [BLOCK_KV, BLOCK_QUARTER] × 4
        # ============================================================
        v_addrs = slot_bases[:, None] + V_START + dim_offs_q[None, :]
        v_packed = tl.load(KV_cache_ptr + v_addrs, mask=kv_mask[:, None], other=0).to(
            tl.int32
        )

        v_scale_addrs = slot_bases + V_SZ_START
        sc_lo = tl.load(KV_cache_ptr + v_scale_addrs, mask=kv_mask, other=0).to(
            tl.uint16
        )
        sc_hi = tl.load(KV_cache_ptr + v_scale_addrs + 1, mask=kv_mask, other=0).to(
            tl.uint16
        )
        v_scale = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

        zr_lo = tl.load(KV_cache_ptr + v_scale_addrs + 2, mask=kv_mask, other=0).to(
            tl.uint16
        )
        zr_hi = tl.load(KV_cache_ptr + v_scale_addrs + 3, mask=kv_mask, other=0).to(
            tl.uint16
        )
        v_zero = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

        v_q0 = (v_packed & 0x3).to(tl.float32)
        v_q1 = ((v_packed >> 2) & 0x3).to(tl.float32)
        v_q2 = ((v_packed >> 4) & 0x3).to(tl.float32)
        v_q3 = ((v_packed >> 6) & 0x3).to(tl.float32)

        v_v0 = (v_q0 - v_zero[:, None]) * v_scale[:, None]
        v_v1 = (v_q1 - v_zero[:, None]) * v_scale[:, None]
        v_v2 = (v_q2 - v_zero[:, None]) * v_scale[:, None]
        v_v3 = (v_q3 - v_zero[:, None]) * v_scale[:, None]

        # ============================================================
        # WEIGHTED VALUE ACCUMULATION via tl.dot: [BLOCK_H, BQ]
        # p: [BLOCK_H, BLOCK_KV], v: [BLOCK_KV, BQ]
        # ============================================================
        p_f16 = p.to(tl.float16)
        acc0 = acc0 * re_scale[:, None] + tl.dot(p_f16, v_v0.to(tl.float16))
        acc1 = acc1 * re_scale[:, None] + tl.dot(p_f16, v_v1.to(tl.float16))
        acc2 = acc2 * re_scale[:, None] + tl.dot(p_f16, v_v2.to(tl.float16))
        acc3 = acc3 * re_scale[:, None] + tl.dot(p_f16, v_v3.to(tl.float16))

        l_prev = l_prev * re_scale + tl.sum(p, 1)
        m_prev = n_e_max

    # Store partial results per head
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)  # [BLOCK_H]
    lse = m_prev + tl.log(safe_l)  # [BLOCK_H]

    # Write [BLOCK_H, BLOCK_QUARTER] × 4 + [BLOCK_H] lse
    out_base = (
        bid * stride_mid_b + cur_head * stride_mid_h + sid * stride_mid_s
    )  # [BLOCK_H]

    tl.store(
        Mid_o_ptr + out_base[:, None] + dim_offs_q[None, :],
        acc0 / safe_l[:, None],
        mask=mask_h[:, None],
    )
    tl.store(
        Mid_o_ptr + out_base[:, None] + dim_offs_q[None, :] + BLOCK_QUARTER,
        acc1 / safe_l[:, None],
        mask=mask_h[:, None],
    )
    tl.store(
        Mid_o_ptr + out_base[:, None] + dim_offs_q[None, :] + 2 * BLOCK_QUARTER,
        acc2 / safe_l[:, None],
        mask=mask_h[:, None],
    )
    tl.store(
        Mid_o_ptr + out_base[:, None] + dim_offs_q[None, :] + 3 * BLOCK_QUARTER,
        acc3 / safe_l[:, None],
        mask=mask_h[:, None],
    )
    tl.store(
        Mid_o_ptr + out_base + HEAD_DIM,
        lse,
        mask=mask_h,
    )


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


def triton_oscar_decode_attention(
    query: torch.Tensor,  # [B, Hq, D] — original query
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, padded_slot] uint8
    block_table: torch.Tensor,  # [B, max_num_blocks] int32
    seq_lens: torch.Tensor,  # [B] int32
    Pi: torch.Tensor,  # [D, D] float32
    scale: float,
    # Pre-allocated buffers (optional, avoids per-call allocation)
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    buf_holder: Any = None,
    max_num_kv_splits: int = 32,
) -> torch.Tensor:
    """Launch fused OSCAR INT2 decode attention (Triton stage1 + stage2).

    Returns: output tensor [B, Hq, D] in query's dtype.
    """
    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    device = query.device

    # Compute rotated query: q_rot = q @ Pi (cached on buf_holder)
    # Pi is symmetric Hadamard, so Pi = Pi^T.
    if buf_holder is not None and hasattr(buf_holder, "_oscar_q_rot_buf"):
        q_rot_buf = buf_holder._oscar_q_rot_buf
        if q_rot_buf.shape[0] >= B:
            q_rot = q_rot_buf[:B]
            torch.matmul(query.float(), Pi, out=q_rot)
        else:
            q_rot = (query.float() @ Pi).contiguous()
            buf_holder._oscar_q_rot_buf = q_rot
    else:
        q_rot = (query.float() @ Pi).contiguous()
        if buf_holder is not None:
            buf_holder._oscar_q_rot_buf = q_rot

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
            buf_holder._oscar_mid_o_buf = mid_o

    # Calculate layout offsets
    block_quarter = D // 4
    sz_bytes = 4
    k_aligned_size = (block_quarter + sz_bytes + 15) // 16 * 16

    # GQA head grouping (matches triton_decode_attention grouped kernel)
    BLOCK_H = 16
    VALID_BLOCK_H = min(BLOCK_H, kv_group_size)

    # Stage 1: split-KV tiled attention scoring + value accumulation
    BLOCK_KV = 64
    grid = (
        B,
        triton.cdiv(Hq, VALID_BLOCK_H),
        NUM_KV_SPLITS,
    )

    num_warps = 4

    _oscar_decode_stage1[grid](
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
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
        BLOCK_QUARTER=block_quarter,
        K_START=0,
        K_SZ_START=block_quarter,
        V_START=k_aligned_size,
        V_SZ_START=k_aligned_size + block_quarter,
        ATTN_SCALE=scale,
        BLOCK_KV=BLOCK_KV,
        BLOCK_H=BLOCK_H,
        Q_HEAD_NUM=Hq,
        num_warps=num_warps,
        num_stages=2,
    )

    # Stage 2: Reduce across KV splits
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
            buf_holder._oscar_output_buf = output

    if lse_buf is not None and lse_buf.shape[0] >= B:
        lse = lse_buf[:B, :Hq]
    else:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=device)
        if buf_holder is not None:
            buf_holder._oscar_lse_buf = lse

    block_d = triton.next_power_of_2(D)

    grid2 = (B, Hq)
    _fwd_kernel_stage2[grid2](
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
        BLOCK_DV=block_d,
        Lv=D,
        OUTPUT_FP16=1 if out_dtype == torch.float16 else 0,
        num_warps=4,
        num_stages=2,
    )

    return output
