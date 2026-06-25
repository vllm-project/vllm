# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fused OSCAR INT2 decode attention.

Decode path: split-KV tiled scoring + value accumulation (stage1) followed
by a log-sum-exp reduction across splits (stage2, reused from
``triton_decode_attention``).

Keys and values are stored as asymmetric INT2 (4 indices per byte) with one
fp16 ``(scale, zero)`` pair per vector. The query passed in is already
rotated by ``R_k`` so that scores against the rotated stored keys equal the
true ``Q K^T``; the value-side ``R_v^T`` inverse is applied by the caller to
the returned output, which lives in rotated-V space.
"""

import math

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2


@triton.jit
def _oscar_decode_stage1(
    Q_rot_ptr,  # [B, Hq, D] fp32 — query already rotated by R_k
    KV_cache_ptr,  # [num_blocks, block_size, Hk, slot_size] uint8
    Block_table_ptr,  # [B, max_num_blocks] int32
    Seq_lens_ptr,  # [B] int32
    Mid_o_ptr,  # [B, Hq, NUM_KV_SPLITS, D+1] fp32
    stride_qb,
    stride_qh,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_bt_b,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    KEY_DATA_BYTES: tl.constexpr,  # D // 4
    KEY_PACKED: tl.constexpr,  # key region size incl. meta
    VALUE_DATA_BYTES: tl.constexpr,  # D // 4
    KEY_LEVELS: tl.constexpr,
    VALUE_LEVELS: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    sid = tl.program_id(2)

    kv_head = hid // KV_GROUP_SIZE
    seq_len = tl.load(Seq_lens_ptr + bid)

    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    # INT2 unpack index vectors (loop-invariant): 4 indices per byte.
    byte_idx = d_offs // 4
    bit_shift = (d_offs % 4) * 2

    q_base = bid * stride_qb + hid * stride_qh
    q_rot = tl.load(Q_rot_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    bt_base = bid * stride_bt_b

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
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        # ---- dequant K (INT2) and score ----
        k_byte = tl.load(
            KV_cache_ptr + slot_bases[:, None] + byte_idx[None, :],
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        q_k = ((k_byte >> bit_shift[None, :]) & (KEY_LEVELS - 1)).to(tl.float32)

        k_meta = slot_bases + KEY_DATA_BYTES
        ksc_lo = tl.load(KV_cache_ptr + k_meta, mask=kv_mask, other=0).to(tl.uint16)
        ksc_hi = tl.load(KV_cache_ptr + k_meta + 1, mask=kv_mask, other=0).to(tl.uint16)
        k_scale = (ksc_lo | (ksc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        kzr_lo = tl.load(KV_cache_ptr + k_meta + 2, mask=kv_mask, other=0).to(tl.uint16)
        kzr_hi = tl.load(KV_cache_ptr + k_meta + 3, mask=kv_mask, other=0).to(tl.uint16)
        k_zero = (kzr_lo | (kzr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

        k_deq = q_k * k_scale[:, None] + k_zero[:, None]
        scores = (
            tl.sum(tl.where(d_mask[None, :], q_rot[None, :] * k_deq, 0.0), axis=1)
            * ATTN_SCALE
        )
        scores = tl.where(kv_mask, scores, -float("inf"))

        n_e_max = tl.maximum(tl.max(scores, 0), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max)

        # ---- dequant V (INT2) ----
        v_base = slot_bases + KEY_PACKED
        v_byte = tl.load(
            KV_cache_ptr + v_base[:, None] + byte_idx[None, :],
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        q_v = ((v_byte >> bit_shift[None, :]) & (VALUE_LEVELS - 1)).to(tl.float32)

        v_meta = v_base + VALUE_DATA_BYTES
        vsc_lo = tl.load(KV_cache_ptr + v_meta, mask=kv_mask, other=0).to(tl.uint16)
        vsc_hi = tl.load(KV_cache_ptr + v_meta + 1, mask=kv_mask, other=0).to(tl.uint16)
        v_scale = (vsc_lo | (vsc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        vzr_lo = tl.load(KV_cache_ptr + v_meta + 2, mask=kv_mask, other=0).to(tl.uint16)
        vzr_hi = tl.load(KV_cache_ptr + v_meta + 3, mask=kv_mask, other=0).to(tl.uint16)
        v_zero = (vzr_lo | (vzr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        values = q_v * v_scale[:, None] + v_zero[:, None]

        acc = acc * re_scale + tl.sum(p[:, None] * values, 0)
        l_prev = l_prev * re_scale + tl.sum(p, 0)
        m_prev = n_e_max

    out_base = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(Mid_o_ptr + out_base + d_offs, acc / safe_l, mask=d_mask)
    tl.store(Mid_o_ptr + out_base + HEAD_DIM, m_prev + tl.log(safe_l))


@triton.jit
def _oscar_full_dequant_kv(
    KV_cache_ptr,
    Block_table_ptr,
    K_out_ptr,  # [B, Hk, max_seq, D] fp16 — rotated-space K
    V_out_ptr,  # [B, Hk, max_seq, D] fp16 — rotated-space V
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
    KEY_DATA_BYTES: tl.constexpr,
    KEY_PACKED: tl.constexpr,
    VALUE_DATA_BYTES: tl.constexpr,
    KEY_LEVELS: tl.constexpr,
    VALUE_LEVELS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Dequant cached INT2 K/V to fp16 (still in rotated space)."""
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
    byte_idx = d_offs // 4
    bit_shift = (d_offs % 4) * 2

    # K
    k_byte = tl.load(KV_cache_ptr + slot_base + byte_idx, mask=d_mask, other=0).to(
        tl.int32
    )
    q_k = ((k_byte >> bit_shift) & (KEY_LEVELS - 1)).to(tl.float32)
    k_meta = slot_base + KEY_DATA_BYTES
    ksc = (
        (
            (tl.load(KV_cache_ptr + k_meta).to(tl.uint16))
            | (tl.load(KV_cache_ptr + k_meta + 1).to(tl.uint16) << 8)
        )
        .to(tl.float16, bitcast=True)
        .to(tl.float32)
    )
    kzr = (
        (
            (tl.load(KV_cache_ptr + k_meta + 2).to(tl.uint16))
            | (tl.load(KV_cache_ptr + k_meta + 3).to(tl.uint16) << 8)
        )
        .to(tl.float16, bitcast=True)
        .to(tl.float32)
    )
    k_recon = q_k * ksc + kzr
    ko_base = bid * stride_ko_b + hid * stride_ko_h + pos * stride_ko_s
    tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(tl.float16), mask=d_mask)

    # V
    v_base = slot_base + KEY_PACKED
    v_byte = tl.load(KV_cache_ptr + v_base + byte_idx, mask=d_mask, other=0).to(
        tl.int32
    )
    q_v = ((v_byte >> bit_shift) & (VALUE_LEVELS - 1)).to(tl.float32)
    v_meta = v_base + VALUE_DATA_BYTES
    vsc = (
        (
            (tl.load(KV_cache_ptr + v_meta).to(tl.uint16))
            | (tl.load(KV_cache_ptr + v_meta + 1).to(tl.uint16) << 8)
        )
        .to(tl.float16, bitcast=True)
        .to(tl.float32)
    )
    vzr = (
        (
            (tl.load(KV_cache_ptr + v_meta + 2).to(tl.uint16))
            | (tl.load(KV_cache_ptr + v_meta + 3).to(tl.uint16) << 8)
        )
        .to(tl.float16, bitcast=True)
        .to(tl.float32)
    )
    v_recon = q_v * vsc + vzr
    vo_base = bid * stride_vo_b + hid * stride_vo_h + pos * stride_vo_s
    tl.store(V_out_ptr + vo_base + d_offs, v_recon.to(tl.float16), mask=d_mask)


def oscar_decode_attention(
    q_rot: torch.Tensor,  # [B, Hq, D] — query already rotated by R_k
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, slot_size] uint8
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    key_levels: int,
    value_levels: int,
    key_data_bytes: int,
    key_packed_size: int,
    value_data_bytes: int,
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    max_num_kv_splits: int = 16,
) -> torch.Tensor:
    """Fused OSCAR INT2 decode attention. Returns ``[B, Hq, D]`` in
    rotated-V space (caller applies ``R_v^T``)."""
    B, Hq, D = q_rot.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    device = q_rot.device
    BLOCK_D = triton.next_power_of_2(D)
    NUM_KV_SPLITS = max_num_kv_splits

    q_rot = q_rot.contiguous().float()

    if (
        mid_o_buf is not None
        and mid_o_buf.shape[0] >= B
        and mid_o_buf.shape[2] >= NUM_KV_SPLITS
    ):
        mid_o = mid_o_buf[:B, :Hq, :NUM_KV_SPLITS, :]
    else:
        mid_o = torch.empty(
            B, Hq, NUM_KV_SPLITS, D + 1, dtype=torch.float32, device=device
        )

    grid = (B, Hq, NUM_KV_SPLITS)
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
        KEY_DATA_BYTES=key_data_bytes,
        KEY_PACKED=key_packed_size,
        VALUE_DATA_BYTES=value_data_bytes,
        KEY_LEVELS=key_levels,
        VALUE_LEVELS=value_levels,
        ATTN_SCALE=scale,
        BLOCK_D=BLOCK_D,
        BLOCK_KV=4,
        num_warps=1,
        num_stages=1,
    )

    out_dtype = torch.float32
    if output_buf is not None and output_buf.shape[0] >= B:
        output = output_buf[:B, :Hq, :D]
    else:
        output = torch.empty(B, Hq, D, dtype=out_dtype, device=device)
    if lse_buf is not None and lse_buf.shape[0] >= B:
        lse = lse_buf[:B, :Hq]
    else:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=device)

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
        BLOCK_DV=BLOCK_D,
        Lv=D,
        OUTPUT_FP16=0,
        num_warps=4,
        num_stages=2,
    )
    return output


def oscar_full_dequant_kv(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cached_len: int,
    num_kv_heads: int,
    head_dim: int,
    key_levels: int,
    value_levels: int,
    key_data_bytes: int,
    key_packed_size: int,
    value_data_bytes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequant the first ``cached_len`` cached tokens to fp16 (rotated space).

    Returns ``(k, v)`` each ``[cached_len, Hk, D]``.
    """
    device = kv_cache.device
    block_size = kv_cache.shape[1]
    alloc_len = math.ceil(cached_len / block_size) * block_size
    BLOCK_D = triton.next_power_of_2(head_dim)
    k_buf = torch.empty(
        1, num_kv_heads, alloc_len, head_dim, dtype=torch.float16, device=device
    )
    v_buf = torch.empty_like(k_buf)

    grid = (alloc_len, num_kv_heads)
    _oscar_full_dequant_kv[grid](
        kv_cache,
        block_table,
        k_buf,
        v_buf,
        k_buf.stride(0),
        k_buf.stride(1),
        k_buf.stride(2),
        v_buf.stride(0),
        v_buf.stride(1),
        v_buf.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        NUM_KV_HEADS=num_kv_heads,
        KEY_DATA_BYTES=key_data_bytes,
        KEY_PACKED=key_packed_size,
        VALUE_DATA_BYTES=value_data_bytes,
        KEY_LEVELS=key_levels,
        VALUE_LEVELS=value_levels,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    k = k_buf[0, :, :cached_len, :].transpose(0, 1)  # [cached_len, Hk, D]
    v = v_buf[0, :, :cached_len, :].transpose(0, 1)
    return k, v
