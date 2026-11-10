# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fused TurboQuant decode attention.

Decode path: Triton stage1 (split-KV tiled attention scoring + value
accumulation) + stage2 (log-sum-exp reduction across splits).

Supports FP8 (E4M3) keys, 3-bit and 4-bit uniform quantized values.
"""

import math
from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import (
    _fwd_kernel_stage2,
)

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
    KV_cache_u16_ptr,  # uint16 view of same storage — Opt#3 SoA loads
    # Block table and sequence info
    Block_table_ptr,  # [B, max_num_blocks] int32
    Seq_lens_ptr,  # [B] int32
    # TQ parameters
    Centroids_ptr,  # [n_centroids] float32
    # Output (intermediate for stage2)
    Mid_o_ptr,  # [B, Hq, NUM_KV_SPLITS, D+1] float32
    # Sink support
    Sink_ptr,  # [Hq] float32, pre-computed sink attention logits per head
    # Strides
    stride_qb,
    stride_qh,  # Q strides: [B, Hq, D]
    stride_cache_block: tl.int64,  # bytes per block (bs*H*slot_aligned)
    stride_bt_b: tl.int64,  # block_table stride per batch
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
    VQB: tl.constexpr,  # value_quant_bits (4 or 8=FP8)
    VAL_DATA_BYTES: tl.constexpr,  # ceil(D * vqb / 8) or D for FP8
    # Opt#3 SoA layout
    KEY_DATA_BYTES: tl.constexpr,  # MSE_BYTES (MSE) or HEAD_DIM (FP8)
    META_REGION_OFFSET: tl.constexpr,  # bytes from block start to SoA region
    NUM_SOA_FIELDS: tl.constexpr,  # 3 (MSE) or 2 (FP8)
    SOA_K_NORM: tl.constexpr,  # 0 (MSE); unused for FP8
    SOA_V_SCALE: tl.constexpr,  # 1 (MSE) / 0 (FP8)
    SOA_V_ZERO: tl.constexpr,  # 2 (MSE) / 1 (FP8)
    # Score constants
    ATTN_SCALE: tl.constexpr,  # 1/sqrt(D)
    # Block tile sizes
    BLOCK_D: tl.constexpr,  # next_power_of_2(HEAD_DIM)
    BLOCK_KV: tl.constexpr,  # tokens per tile (16)
    KEY_FP8: tl.constexpr,  # 1 if K is stored as FP8
    NORM_CORRECTION: tl.constexpr = 0,  # 1 = re-normalize centroids
    FP8_E4B15: tl.constexpr = 0,  # 1 = use e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)
    USE_SINKS: tl.constexpr = 0,  # 1 = use sink tokens for attention anchoring
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
    # Sink tokens provide a pre-computed attention bias that should be
    # included in the softmax normalization. For the first split (sid==0),
    # we initialize m_prev with the sink logit. L is always 1.0 for all
    # splits to match unified attention semantics.
    if USE_SINKS:
        if sid == 0:
            m_prev = tl.load(Sink_ptr + hid).to(tl.float32)
            l_prev = 1.0
        else:
            m_prev = -float("inf")
            l_prev = 0.0
    else:
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

        # Opt#3 SoA addressing: data region then metadata region per block.
        slot_within_block = page_off.to(tl.int64)
        block_base = block_nums * stride_cache_block
        DATA_BYTES_PER_SLOT: tl.constexpr = KEY_DATA_BYTES + VAL_DATA_BYTES
        data_bases = (
            block_base
            + slot_within_block * (NUM_KV_HEADS * DATA_BYTES_PER_SLOT)
            + tl.cast(kv_head, tl.int64) * DATA_BYTES_PER_SLOT
        )
        head_meta_u16_base = (block_base + META_REGION_OFFSET) // 2 + tl.cast(
            kv_head, tl.int64
        ) * (NUM_SOA_FIELDS * BLOCK_SIZE)
        knorm_u16_addrs = (
            head_meta_u16_base + SOA_K_NORM * BLOCK_SIZE + slot_within_block
        )
        vscale_u16_addrs = (
            head_meta_u16_base + SOA_V_SCALE * BLOCK_SIZE + slot_within_block
        )
        vzero_u16_addrs = (
            head_meta_u16_base + SOA_V_ZERO * BLOCK_SIZE + slot_within_block
        )

        # ============================================================
        # COMPUTE ATTENTION SCORES: [BLOCK_KV]
        # ============================================================
        if KEY_FP8:
            k_addrs = data_bases[:, None] + d_offs[None, :]
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
            mse_addrs0 = data_bases[:, None] + mse_byte_idx[None, :]
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

            # Opt#1: norm-correction (1/||c_vec||) is pre-folded into the
            # stored per-token scalar by the MSE store kernel when
            # NORM_CORRECTION=1, so we skip the per-tile sum+sqrt+divide.
            # NORM_CORRECTION is kept in the signature for API parity; it is
            # implicit in the stored norm value now.

            term1 = tl.sum(
                tl.where(d_mask[None, :], q_rot[None, :] * c_vals, 0.0),
                axis=1,
            )

            # Opt#3: K-norms live in the per-block SoA region. Single u16
            # load per token; contiguous across tokens within a block.
            norm_u16 = tl.load(
                KV_cache_u16_ptr + knorm_u16_addrs, mask=kv_mask, other=0
            )
            vec_norms = norm_u16.to(tl.float16, bitcast=True).to(tl.float32)

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
        # Opt#3: V data immediately follows K data in the slot (no
        # interleaved metadata). Scale/zero come from the SoA metadata
        # region as single u16 loads per field per token.
        # ============================================================
        val_bases = data_bases + KEY_DATA_BYTES

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

        scale_u16 = tl.load(KV_cache_u16_ptr + vscale_u16_addrs, mask=kv_mask, other=0)
        zero_u16 = tl.load(KV_cache_u16_ptr + vzero_u16_addrs, mask=kv_mask, other=0)
        v_scales = scale_u16.to(tl.float16, bitcast=True).to(tl.float32)
        v_zeros = zero_u16.to(tl.float16, bitcast=True).to(tl.float32)
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
# Pre-dequant kernel: Bulk dequant K (MSE+norms) and V to fp16
# ---------------------------------------------------------------------------


@triton.jit
def _tq_full_dequant_kv(
    KV_cache_ptr,
    KV_cache_u16_ptr,
    Block_table_ptr,
    Centroids_ptr,
    K_out_ptr,  # [B, Hk, max_seq, D] float16
    V_out_ptr,  # [B, Hk, max_seq, D] float16
    stride_ko_b: tl.int64,
    stride_ko_h: tl.int64,
    stride_ko_s: tl.int64,
    stride_vo_b: tl.int64,
    stride_vo_h: tl.int64,
    stride_vo_s: tl.int64,
    stride_cache_block: tl.int64,
    stride_bt_b: tl.int64,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    MSE_BITS: tl.constexpr,
    KEY_FP8: tl.constexpr,
    # Opt#3 SoA layout
    KEY_DATA_BYTES: tl.constexpr,
    META_REGION_OFFSET: tl.constexpr,
    NUM_SOA_FIELDS: tl.constexpr,
    SOA_K_NORM: tl.constexpr,
    SOA_V_SCALE: tl.constexpr,
    SOA_V_ZERO: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NORM_CORRECTION: tl.constexpr = 0,
    FP8_E4B15: tl.constexpr = 0,  # 1 = use e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)
):
    """Full dequant: reconstruct K (MSE centroids * norm or FP8) and V to fp16.

    Opt#3 SoA layout: data region [bs, H, KD+VD] + per-block SoA metadata.
    """
    pos = tl.program_id(0)
    bh = tl.program_id(1)
    bid = bh // NUM_KV_HEADS
    hid = bh % NUM_KV_HEADS

    page_idx = pos // BLOCK_SIZE
    page_off = pos % BLOCK_SIZE
    block_num = tl.load(Block_table_ptr + bid * stride_bt_b + page_idx).to(tl.int64)

    block_base = block_num * stride_cache_block
    slot_within_block = tl.cast(page_off, tl.int64)
    DATA_BYTES_PER_SLOT: tl.constexpr = KEY_DATA_BYTES + VAL_DATA_BYTES
    data_base = (
        block_base
        + slot_within_block * (NUM_KV_HEADS * DATA_BYTES_PER_SLOT)
        + tl.cast(hid, tl.int64) * DATA_BYTES_PER_SLOT
    )
    head_meta_u16_base = (block_base + META_REGION_OFFSET) // 2 + tl.cast(
        hid, tl.int64
    ) * (NUM_SOA_FIELDS * BLOCK_SIZE)
    knorm_u16_addr = head_meta_u16_base + SOA_K_NORM * BLOCK_SIZE + slot_within_block
    vscale_u16_addr = head_meta_u16_base + SOA_V_SCALE * BLOCK_SIZE + slot_within_block
    vzero_u16_addr = head_meta_u16_base + SOA_V_ZERO * BLOCK_SIZE + slot_within_block

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    # === K dequant ===
    ko_base = bid * stride_ko_b + hid * stride_ko_h + pos * stride_ko_s
    if KEY_FP8:
        k_raw = tl.load(KV_cache_ptr + data_base + d_offs, mask=d_mask, other=0)
        if FP8_E4B15:
            k_recon = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
        else:
            k_recon = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(tl.float16), mask=d_mask)
    else:
        mse_bit_off = d_offs * MSE_BITS
        mse_byte_idx = mse_bit_off // 8
        mse_bit_shift = mse_bit_off % 8
        mse_umask = (1 << MSE_BITS) - 1

        mse_raw0 = tl.load(
            KV_cache_ptr + data_base + mse_byte_idx, mask=d_mask, other=0
        ).to(tl.int32)
        mse_raw1 = tl.load(
            KV_cache_ptr + data_base + mse_byte_idx + 1, mask=d_mask, other=0
        ).to(tl.int32)
        raw16_key = mse_raw0 | (mse_raw1 << 8)
        mse_idx = (raw16_key >> mse_bit_shift) & mse_umask

        k_mse = tl.load(Centroids_ptr + mse_idx, mask=d_mask, other=0.0)

        # Opt#1: norm-correction is pre-folded into the stored scalar.
        # Opt#3: K-norm lives in the SoA region — single u16 load.
        norm_u16 = tl.load(KV_cache_u16_ptr + knorm_u16_addr)
        vec_norm = norm_u16.to(tl.float16, bitcast=True).to(tl.float32)

        k_recon = vec_norm * k_mse
        tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(tl.float16), mask=d_mask)

    # === V dequant ===
    val_base = data_base + KEY_DATA_BYTES
    if VQB == 4:
        vb_idx = d_offs // 2
        vb_shift = (d_offs % 2) * 4
        val_raw = tl.load(KV_cache_ptr + val_base + vb_idx, mask=d_mask, other=0).to(
            tl.int32
        )
        v_idx = ((val_raw >> vb_shift) & 0xF).to(tl.float32)
    elif VQB == 3:
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
    else:
        v_idx = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Opt#3: V scale/zero single-u16 loads from SoA region.
    sc_u16 = tl.load(KV_cache_u16_ptr + vscale_u16_addr)
    zr_u16 = tl.load(KV_cache_u16_ptr + vzero_u16_addr)
    v_scale = sc_u16.to(tl.float16, bitcast=True).to(tl.float32)
    v_zero = zr_u16.to(tl.float16, bitcast=True).to(tl.float32)
    v_vals = v_idx * v_scale + v_zero

    vo_base = bid * stride_vo_b + hid * stride_vo_h + pos * stride_vo_s
    tl.store(V_out_ptr + vo_base + d_offs, v_vals.to(tl.float16), mask=d_mask)


# ---------------------------------------------------------------------------
# Stage 2: Reuse from triton_decode_attention.py
# ---------------------------------------------------------------------------

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
    sinks: torch.Tensor | None = None,  # [Hq] float32, pre-computed sink logits
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

    # Opt#3 SoA layout constants (derived locally; match store-side values).
    key_data_bytes = D if key_fp8 else cfg["mse_bytes"]
    data_bytes_per_slot = key_data_bytes + cfg["val_data_bytes"]
    meta_region_offset = block_size * Hk * data_bytes_per_slot
    num_soa_fields = 2 if key_fp8 else 3
    soa_k_norm = 0
    soa_v_scale = 0 if key_fp8 else 1
    soa_v_zero = 1 if key_fp8 else 2
    kv_cache_u16 = kv_cache.view(torch.uint16)

    # Compute q_rot = q @ Pi.T (rotated query for MSE key scoring)
    # FP8 path: pass query directly (float16); kernel casts inline.
    # MSE path: still needs external GEMM (cuBLAS), so q_rot is float32.
    if key_fp8:
        q_rot = query.contiguous()
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
    fp8_e4b15 = _use_fp8_e4b15(device.index or 0)
    BLOCK_KV = 4
    grid = (B, Hq, NUM_KV_SPLITS)

    # Prepare sink pointer (convert to float32 on device if provided)
    use_sinks = sinks is not None

    _tq_decode_stage1[grid](
        q_rot,
        kv_cache,
        kv_cache_u16,
        block_table,
        seq_lens,
        centroids,
        mid_o,
        sinks,
        q_rot.stride(0),
        q_rot.stride(1),
        kv_cache.stride(0),
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
        VQB=value_quant_bits,
        VAL_DATA_BYTES=cfg["val_data_bytes"],
        KEY_DATA_BYTES=key_data_bytes,
        META_REGION_OFFSET=meta_region_offset,
        NUM_SOA_FIELDS=num_soa_fields,
        SOA_K_NORM=soa_k_norm,
        SOA_V_SCALE=soa_v_scale,
        SOA_V_ZERO=soa_v_zero,
        ATTN_SCALE=scale,
        BLOCK_D=cfg["BLOCK_D"],
        BLOCK_KV=BLOCK_KV,
        KEY_FP8=1 if key_fp8 else 0,
        NORM_CORRECTION=1 if norm_correction else 0,
        FP8_E4B15=fp8_e4b15,
        USE_SINKS=1 if use_sinks else 0,
        num_warps=1,
        num_stages=1,
    )

    # Stage 2: Reduce across KV splits
    if output_buf is not None and output_buf.shape[0] >= B:
        output = output_buf[:B, :Hq, :D]
    else:
        output = torch.empty(B, Hq, D, dtype=torch.float32, device=device)
        if buf_holder is not None:
            buf_holder._tq_output_buf = output
    if lse_buf is not None and lse_buf.shape[0] >= B:
        lse = lse_buf[:B, :Hq]
    else:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=device)
        if buf_holder is not None:
            buf_holder._tq_lse_buf = lse

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
        BLOCK_DV=cfg["BLOCK_D"],
        Lv=D,
        num_warps=4,
        num_stages=2,
    )

    return output.to(query.dtype)
