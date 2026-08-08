# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fused TurboQuant decode attention.

Decode path: Triton stage1 (split-KV tiled attention scoring + value
accumulation) + stage2 (log-sum-exp reduction across splits).

Supports FP8 (E4M3) keys, 3-bit and 4-bit uniform quantized values.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonPointerInputVariant,
    TritonWarmupTensor,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import (
    _DECODE_STAGE2_KERNEL,
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


@triton.jit(
    do_not_specialize=[
        "stride_qb",
        "stride_qh",
        "stride_cache_block",
        "stride_cache_pos",
        "stride_cache_head",
        "stride_bt_b",
        "stride_mid_b",
        "stride_mid_h",
        "stride_mid_s",
    ],
    do_not_specialize_on_alignment=["Seq_lens_ptr"],
)
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
# Pre-dequant kernel: Bulk dequant K (MSE+norms) and V to fp16
# ---------------------------------------------------------------------------


@triton.jit(
    do_not_specialize=[
        "stride_ko_b",
        "stride_ko_h",
        "stride_ko_s",
        "stride_vo_b",
        "stride_vo_h",
        "stride_vo_s",
        "stride_cache_block",
        "stride_cache_pos",
        "stride_cache_head",
        "stride_bt_b",
    ]
)
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


def _is_triton_aligned(tensor: torch.Tensor) -> bool:
    return tensor.data_ptr() % 16 == 0


class TurboQuantDecodeStage1Kernel(
    VllmJitKernel["TurboQuantDecodeStage1Kernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        query_dtype: torch.dtype
        input_variant: TritonPointerInputVariant
        NUM_KV_HEADS: int
        HEAD_DIM: int
        BLOCK_SIZE: int
        NUM_KV_SPLITS: int
        KV_GROUP_SIZE: int
        MSE_BITS: int
        MSE_BYTES: int
        KPS: int
        VQB: int
        VAL_DATA_BYTES: int
        ATTN_SCALE: float
        BLOCK_D: int
        KEY_FP8: int
        NORM_CORRECTION: int
        FP8_E4B15: int

    kernel = _tq_decode_stage1

    def dispatch(  # type: ignore[override]
        self,
        *,
        query_dtype: torch.dtype,
        q_aligned: bool,
        kv_cache_aligned: bool,
        block_table_aligned: bool,
        centroids_aligned: bool,
        mid_o_aligned: bool,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        num_kv_splits: int,
        mse_bits: int,
        mse_bytes: int,
        key_packed_size: int,
        value_quant_bits: int,
        val_data_bytes: int,
        scale: float,
        key_fp8: bool,
        norm_correction: bool,
        fp8_e4b15: int,
    ) -> CompileKey:
        input_variant = TritonPointerInputVariant.from_alignment(
            q=q_aligned,
            kv_cache=kv_cache_aligned,
            block_table=block_table_aligned,
            centroids=centroids_aligned,
            mid_o=mid_o_aligned,
        )
        return self.CompileKey(
            query_dtype=query_dtype,
            input_variant=input_variant,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size,
            NUM_KV_SPLITS=num_kv_splits,
            KV_GROUP_SIZE=num_query_heads // num_kv_heads,
            MSE_BITS=mse_bits,
            MSE_BYTES=mse_bytes,
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            ATTN_SCALE=scale,
            BLOCK_D=triton.next_power_of_2(head_dim),
            KEY_FP8=1 if key_fp8 else 0,
            NORM_CORRECTION=1 if norm_correction else 0,
            FP8_E4B15=fp8_e4b15,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        cache_dtype = vllm_config.cache_config.cache_dtype
        if not str(cache_dtype).startswith("turboquant_"):
            return []

        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        head_dim = model_config.get_head_size()
        num_query_heads = model_config.get_num_attention_heads(parallel_config)
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        tq_config = TurboQuantConfig.from_cache_dtype(str(cache_dtype), head_dim)
        mse_bits = tq_config.key_mse_bits
        value_quant_bits = tq_config.effective_value_quant_bits
        return self._trace_dispatch(self.dispatch)(
            query_dtype=(model_config.dtype if tq_config.key_fp8 else torch.float32),
            q_aligned=True,
            kv_cache_aligned=True,
            block_table_aligned=True,
            centroids_aligned=True,
            mid_o_aligned=True,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=vllm_config.cache_config.block_size,
            num_kv_splits=(
                vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
            ),
            mse_bits=mse_bits,
            mse_bytes=math.ceil(head_dim * mse_bits / 8),
            key_packed_size=tq_config.key_packed_size,
            value_quant_bits=value_quant_bits,
            val_data_bytes=math.ceil(head_dim * value_quant_bits / 8),
            scale=head_dim**-0.5,
            key_fp8=tq_config.key_fp8,
            norm_correction=tq_config.norm_correction,
            fp8_e4b15=_use_fp8_e4b15(),
        )

    def compile(self, compile_key: CompileKey) -> None:
        variant = compile_key.input_variant
        self.kernel.warmup(
            variant.pointer("q", compile_key.query_dtype),
            variant.pointer("kv_cache", torch.uint8),
            variant.pointer("block_table", torch.int32),
            TritonWarmupTensor(torch.int32),
            variant.pointer("centroids", torch.float32),
            variant.pointer("mid_o", torch.float32),
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            NUM_KV_HEADS=compile_key.NUM_KV_HEADS,
            HEAD_DIM=compile_key.HEAD_DIM,
            BLOCK_SIZE=compile_key.BLOCK_SIZE,
            NUM_KV_SPLITS=compile_key.NUM_KV_SPLITS,
            KV_GROUP_SIZE=compile_key.KV_GROUP_SIZE,
            MSE_BITS=compile_key.MSE_BITS,
            MSE_BYTES=compile_key.MSE_BYTES,
            KPS=compile_key.KPS,
            VQB=compile_key.VQB,
            VAL_DATA_BYTES=compile_key.VAL_DATA_BYTES,
            ATTN_SCALE=compile_key.ATTN_SCALE,
            BLOCK_D=compile_key.BLOCK_D,
            BLOCK_KV=4,
            KEY_FP8=compile_key.KEY_FP8,
            NORM_CORRECTION=compile_key.NORM_CORRECTION,
            FP8_E4B15=compile_key.FP8_E4B15,
            num_warps=1,
            num_stages=1,
            grid=(1, 1, compile_key.NUM_KV_SPLITS),
        )

    def __call__(
        self,
        q_rot: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        centroids: torch.Tensor,
        mid_o: torch.Tensor,
        *,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        num_kv_splits: int,
        mse_bits: int,
        mse_bytes: int,
        key_packed_size: int,
        value_quant_bits: int,
        val_data_bytes: int,
        scale: float,
        key_fp8: bool,
        norm_correction: bool,
        fp8_e4b15: int,
    ) -> None:
        key = self.dispatch(
            query_dtype=q_rot.dtype,
            q_aligned=_is_triton_aligned(q_rot),
            kv_cache_aligned=_is_triton_aligned(kv_cache),
            block_table_aligned=_is_triton_aligned(block_table),
            centroids_aligned=_is_triton_aligned(centroids),
            mid_o_aligned=_is_triton_aligned(mid_o),
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            num_kv_splits=num_kv_splits,
            mse_bits=mse_bits,
            mse_bytes=mse_bytes,
            key_packed_size=key_packed_size,
            value_quant_bits=value_quant_bits,
            val_data_bytes=val_data_bytes,
            scale=scale,
            key_fp8=key_fp8,
            norm_correction=norm_correction,
            fp8_e4b15=fp8_e4b15,
        )
        grid = (q_rot.size(0), num_query_heads, num_kv_splits)
        self.kernel[grid](
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
            NUM_KV_HEADS=key.NUM_KV_HEADS,
            HEAD_DIM=key.HEAD_DIM,
            BLOCK_SIZE=key.BLOCK_SIZE,
            NUM_KV_SPLITS=key.NUM_KV_SPLITS,
            KV_GROUP_SIZE=key.KV_GROUP_SIZE,
            MSE_BITS=key.MSE_BITS,
            MSE_BYTES=key.MSE_BYTES,
            KPS=key.KPS,
            VQB=key.VQB,
            VAL_DATA_BYTES=key.VAL_DATA_BYTES,
            ATTN_SCALE=key.ATTN_SCALE,
            BLOCK_D=key.BLOCK_D,
            BLOCK_KV=4,
            KEY_FP8=key.KEY_FP8,
            NORM_CORRECTION=key.NORM_CORRECTION,
            FP8_E4B15=key.FP8_E4B15,
            num_warps=1,
            num_stages=1,
        )


_TQ_DECODE_STAGE1_KERNEL = TurboQuantDecodeStage1Kernel()


class TurboQuantFullDequantKernel(
    VllmJitKernel["TurboQuantFullDequantKernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        input_variant: TritonPointerInputVariant
        HEAD_DIM: int
        BLOCK_SIZE: int
        NUM_KV_HEADS: int
        MSE_BYTES: int
        KPS: int
        VQB: int
        VAL_DATA_BYTES: int
        MSE_BITS: int
        KEY_FP8: int
        BLOCK_D: int
        NORM_CORRECTION: int
        FP8_E4B15: int

    kernel = _tq_full_dequant_kv

    def dispatch(  # type: ignore[override]
        self,
        *,
        kv_cache_aligned: bool,
        block_table_aligned: bool,
        centroids_aligned: bool,
        k_out_aligned: bool,
        v_out_aligned: bool,
        head_dim: int,
        block_size: int,
        num_kv_heads: int,
        mse_bytes: int,
        key_packed_size: int,
        value_quant_bits: int,
        val_data_bytes: int,
        mse_bits: int,
        key_fp8: bool,
        norm_correction: bool,
        fp8_e4b15: int,
    ) -> CompileKey:
        input_variant = TritonPointerInputVariant.from_alignment(
            kv_cache=kv_cache_aligned,
            block_table=block_table_aligned,
            centroids=centroids_aligned,
            k_out=k_out_aligned,
            v_out=v_out_aligned,
        )
        return self.CompileKey(
            input_variant=input_variant,
            HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size,
            NUM_KV_HEADS=num_kv_heads,
            MSE_BYTES=mse_bytes,
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            MSE_BITS=mse_bits,
            KEY_FP8=1 if key_fp8 else 0,
            BLOCK_D=triton.next_power_of_2(head_dim),
            NORM_CORRECTION=1 if norm_correction else 0,
            FP8_E4B15=fp8_e4b15,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        cache_dtype = vllm_config.cache_config.cache_dtype
        if not str(cache_dtype).startswith("turboquant_"):
            return []

        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        head_dim = model_config.get_head_size()
        tq_config = TurboQuantConfig.from_cache_dtype(str(cache_dtype), head_dim)
        mse_bits = tq_config.key_mse_bits
        value_quant_bits = tq_config.effective_value_quant_bits
        return self._trace_dispatch(self.dispatch)(
            kv_cache_aligned=True,
            block_table_aligned=True,
            centroids_aligned=True,
            k_out_aligned=True,
            v_out_aligned=True,
            head_dim=head_dim,
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=model_config.get_num_kv_heads(parallel_config),
            mse_bytes=math.ceil(head_dim * mse_bits / 8),
            key_packed_size=tq_config.key_packed_size,
            value_quant_bits=value_quant_bits,
            val_data_bytes=math.ceil(head_dim * value_quant_bits / 8),
            mse_bits=mse_bits,
            key_fp8=tq_config.key_fp8,
            norm_correction=tq_config.norm_correction,
            fp8_e4b15=_use_fp8_e4b15(),
        )

    def compile(self, compile_key: CompileKey) -> None:
        variant = compile_key.input_variant
        self.kernel.warmup(
            variant.pointer("kv_cache", torch.uint8),
            variant.pointer("block_table", torch.int32),
            variant.pointer("centroids", torch.float32),
            variant.pointer("k_out", torch.float16),
            variant.pointer("v_out", torch.float16),
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            HEAD_DIM=compile_key.HEAD_DIM,
            BLOCK_SIZE=compile_key.BLOCK_SIZE,
            NUM_KV_HEADS=compile_key.NUM_KV_HEADS,
            MSE_BYTES=compile_key.MSE_BYTES,
            KPS=compile_key.KPS,
            VQB=compile_key.VQB,
            VAL_DATA_BYTES=compile_key.VAL_DATA_BYTES,
            MSE_BITS=compile_key.MSE_BITS,
            KEY_FP8=compile_key.KEY_FP8,
            BLOCK_D=compile_key.BLOCK_D,
            NORM_CORRECTION=compile_key.NORM_CORRECTION,
            FP8_E4B15=compile_key.FP8_E4B15,
            num_warps=4,
            grid=(1, compile_key.NUM_KV_HEADS),
        )

    def __call__(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        centroids: torch.Tensor,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
        *,
        head_dim: int,
        block_size: int,
        num_kv_heads: int,
        mse_bytes: int,
        key_packed_size: int,
        value_quant_bits: int,
        val_data_bytes: int,
        mse_bits: int,
        key_fp8: bool,
        norm_correction: bool,
        fp8_e4b15: int,
    ) -> None:
        key = self.dispatch(
            kv_cache_aligned=_is_triton_aligned(kv_cache),
            block_table_aligned=_is_triton_aligned(block_table),
            centroids_aligned=_is_triton_aligned(centroids),
            k_out_aligned=_is_triton_aligned(k_out),
            v_out_aligned=_is_triton_aligned(v_out),
            head_dim=head_dim,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            mse_bytes=mse_bytes,
            key_packed_size=key_packed_size,
            value_quant_bits=value_quant_bits,
            val_data_bytes=val_data_bytes,
            mse_bits=mse_bits,
            key_fp8=key_fp8,
            norm_correction=norm_correction,
            fp8_e4b15=fp8_e4b15,
        )
        grid = (k_out.size(2), k_out.size(0) * num_kv_heads)
        self.kernel[grid](
            kv_cache,
            block_table,
            centroids,
            k_out,
            v_out,
            k_out.stride(0),
            k_out.stride(1),
            k_out.stride(2),
            v_out.stride(0),
            v_out.stride(1),
            v_out.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            HEAD_DIM=key.HEAD_DIM,
            BLOCK_SIZE=key.BLOCK_SIZE,
            NUM_KV_HEADS=key.NUM_KV_HEADS,
            MSE_BYTES=key.MSE_BYTES,
            KPS=key.KPS,
            VQB=key.VQB,
            VAL_DATA_BYTES=key.VAL_DATA_BYTES,
            MSE_BITS=key.MSE_BITS,
            KEY_FP8=key.KEY_FP8,
            BLOCK_D=key.BLOCK_D,
            NORM_CORRECTION=key.NORM_CORRECTION,
            FP8_E4B15=key.FP8_E4B15,
            num_warps=4,
        )


_TQ_FULL_DEQUANT_KERNEL = TurboQuantFullDequantKernel()


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
) -> torch.Tensor:
    """Launch fused TQ decode attention (Triton stage1 + stage2).

    Returns: output tensor [B, Hq, D] in query's dtype.
    """
    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    device = query.device

    cfg = _get_layout(D, mse_bits, value_quant_bits, key_packed_size)

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
    _TQ_DECODE_STAGE1_KERNEL(
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        mid_o,
        num_query_heads=Hq,
        num_kv_heads=Hk,
        head_dim=D,
        block_size=block_size,
        num_kv_splits=NUM_KV_SPLITS,
        mse_bits=mse_bits,
        mse_bytes=cfg["mse_bytes"],
        key_packed_size=key_packed_size,
        value_quant_bits=value_quant_bits,
        val_data_bytes=cfg["val_data_bytes"],
        scale=scale,
        key_fp8=key_fp8,
        norm_correction=norm_correction,
        fp8_e4b15=fp8_e4b15,
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

    _DECODE_STAGE2_KERNEL(
        mid_o,
        output,
        lse,
        seq_lens,
        num_kv_splits=NUM_KV_SPLITS,
        block_dv=cfg["BLOCK_D"],
        lv=D,
    )

    return output  # already in query dtype
