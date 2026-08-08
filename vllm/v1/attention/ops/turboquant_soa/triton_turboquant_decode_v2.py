# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optimized Triton TurboQuant decode attention (v2).

FLUTE-paper optimizations applied:
  1. Grouped Q heads: grid over (B, Hk, splits) instead of (B, Hq, splits).
     Each program loads BLOCK_M Q heads sharing a KV head into a 2D tile,
     enabling tl.dot on tensor cores (MFMA/WMMA) for both Q·K and P·V.
  2. Vectorized pair LUT: precompute pair_table[i][j] = (T[i], T[j])
     offline. At runtime, extract adjacent index pairs and fetch two
     dequantized centroids with a single gather, halving LUT lookups.
  3. exp2 instead of exp: scores pre-scaled by log2(e) so the hardware-
     native exp2 instruction replaces the more expensive exp.
  4. Wider index extraction: for 4-bit MSE, two adjacent 4-bit indices share
     a byte. One byte load yields both, eliminating redundant loads.
  5. Centroids pre-warmed in L1 at kernel start.
  6. BLOCK_KV = TILE_SIZE raised to 16-32 (from 4), reducing loop iterations
     and softmax rescaling overhead.

Stage 2 is reused unchanged from triton_decode_attention.py.
"""

import math
from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2

from .triton_turboquant_decode import _use_fp8_e4b15

# ROCm prefers num_stages=1 in attention-like kernels to reduce shared-memory
# pressure (mirrors the pattern used in triton_decode_attention.py and
# triton_turboquant_decode.py).
_is_hip = current_platform.is_rocm()

# On ROCm, bf16 has the same MFMA throughput as fp16 but wider dynamic range
# (8-bit exponent vs 5-bit), which is safer for attention scores.  On CUDA,
# fp16 tensor cores may be faster than bf16 for some shapes.
_DOT_DTYPE = tl.bfloat16 if _is_hip else tl.float16


# ---------------------------------------------------------------------------
# Pair LUT construction (FLUTE) — called once at launcher time
# ---------------------------------------------------------------------------


def build_pair_lut(centroids: torch.Tensor) -> torch.Tensor:
    """Build vectorized pair lookup table.

    For N centroids, returns a [N, N, 2] float32 tensor where
    pair_lut[i, j] = (centroids[i], centroids[j]).
    Flattened to [N*N, 2] for kernel indexing: pair_lut[i*N + j].

    For 4-bit MSE (N=16): 16*16*2*4 = 2048 bytes — fits in L1/smem.
    For 3-bit MSE (N=8):   8*8*2*4  = 512 bytes.
    """
    N = centroids.shape[0]
    # pair_lut[i,j,0] = centroids[i], pair_lut[i,j,1] = centroids[j]
    c = centroids.float()
    lut = torch.empty(N, N, 2, dtype=torch.float32, device=centroids.device)
    lut[:, :, 0] = c[:, None]
    lut[:, :, 1] = c[None, :]
    return lut.reshape(N * N, 2).contiguous()


# ---------------------------------------------------------------------------
# Stage 1 v2: Grouped-Q + pair LUT + tl.dot + exp2
# ---------------------------------------------------------------------------


@triton.jit
def _tq_decode_stage1_v2(
    Q_rot_ptr,
    KV_cache_ptr,
    KV_cache_u16_ptr,  # uint16 view — Opt#3 SoA loads
    Block_table_ptr,
    Seq_lens_ptr,
    Centroids_ptr,
    Pair_lut_ptr,
    Mid_o_ptr,
    stride_qb,
    stride_qh,
    stride_cache_block,  # bytes per block (bs*H*slot_aligned)
    stride_bt_b,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PADDED_SLOT: tl.constexpr,
    MAX_NUM_BLOCKS: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    # Opt#3 SoA layout
    KEY_DATA_BYTES: tl.constexpr,
    META_REGION_OFFSET: tl.constexpr,
    NUM_SOA_FIELDS: tl.constexpr,
    SOA_K_NORM: tl.constexpr,
    SOA_V_SCALE: tl.constexpr,
    SOA_V_ZERO: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    KEY_FP8: tl.constexpr,
    NORM_CORRECTION: tl.constexpr = 0,
    FP8_E4B15: tl.constexpr = 0,
    USE_PAIR_LUT: tl.constexpr = 0,
    USE_BF16_DOT: tl.constexpr = 0,
):
    # Grid: (B, Hk, NUM_KV_SPLITS) — one program per KV head group
    bid = tl.program_id(0)
    kv_hid = tl.program_id(1)
    sid = tl.program_id(2)

    seq_len = tl.load(Seq_lens_ptr + bid)
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)

    if split_start >= split_end:
        return

    # Dimension and tile offsets
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    m_offs = tl.arange(0, BLOCK_M)
    t_offs = tl.arange(0, TILE_SIZE)

    # Q head indices for this KV group: [kv_hid * GS, kv_hid * GS + GS)
    q_head_offs = kv_hid * KV_GROUP_SIZE + m_offs
    q_mask = m_offs < KV_GROUP_SIZE

    # Load Q tile: [BLOCK_M, BLOCK_D] float32
    q_addrs = bid * stride_qb + q_head_offs[:, None] * stride_qh + d_offs[None, :]
    Q = tl.load(
        Q_rot_ptr + q_addrs,
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    # Pre-warm centroids in L1
    if not KEY_FP8:
        _c_warm = tl.load(Centroids_ptr + tl.arange(0, N_CENTROIDS))  # noqa: F841

    # Loop-invariant bit-extraction constants for MSE keys
    if not KEY_FP8:
        if MSE_BITS == 4:
            pass  # 4-bit byte extraction is computed inline in the tile loop
        else:
            # Generic bit extraction
            mse_bit_off = d_offs * MSE_BITS
            mse_byte_idx = mse_bit_off // 8
            mse_bit_shift = mse_bit_off % 8
            mse_mask_val = (1 << MSE_BITS) - 1

    # Loop-invariant bit-extraction for 3-bit values
    if VQB == 3:
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8

    # exp2 pre-scaling
    RCP_LN2: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471805599453
    QK_SCALE = ATTN_SCALE * RCP_LN2

    # Online softmax accumulators: [BLOCK_M] for max, [BLOCK_M] for L
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.zeros([BLOCK_M], dtype=tl.float32)

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    bt_base = bid * stride_bt_b

    # ================================================================
    # TILED LOOP: TILE_SIZE KV tokens per iteration
    # ================================================================
    for start_n in range(split_start, split_end, TILE_SIZE):
        kv_offs = start_n + t_offs
        kv_mask_1d = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx,
            mask=kv_mask_1d,
            other=0,
        ).to(tl.int64)

        # Opt#3 SoA addressing: data region then per-block SoA metadata.
        slot_within_block = page_off.to(tl.int64)
        block_base = block_nums * stride_cache_block
        DATA_BYTES_PER_SLOT: tl.constexpr = KEY_DATA_BYTES + VAL_DATA_BYTES
        data_bases = (
            block_base
            + slot_within_block * (NUM_KV_HEADS * DATA_BYTES_PER_SLOT)
            + tl.cast(kv_hid, tl.int64) * DATA_BYTES_PER_SLOT
        )
        head_meta_u16_base = (block_base + META_REGION_OFFSET) // 2 + tl.cast(
            kv_hid, tl.int64
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
        # KEY DEQUANT → K_T: [BLOCK_D, TILE_SIZE] for tl.dot(Q, K_T)
        # ============================================================
        if KEY_FP8:
            k_addrs = data_bases[:, None] + d_offs[None, :]
            k_raw = tl.load(
                KV_cache_ptr + k_addrs,
                mask=kv_mask_1d[:, None] & d_mask[None, :],
                other=0,
            )
            if FP8_E4B15:
                K_f = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
            else:
                K_f = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            # K_f: [TILE_SIZE, BLOCK_D] → transpose for tl.dot
            K_T = tl.trans(K_f)
            # S: [BLOCK_M, TILE_SIZE]  (bf16 dot on ROCm for better dynamic range)
            if USE_BF16_DOT:
                S = QK_SCALE * tl.dot(Q.to(tl.bfloat16), K_T.to(tl.bfloat16))
            else:
                S = QK_SCALE * tl.dot(Q.to(tl.float16), K_T.to(tl.float16))
        else:
            # --- MSE dequantization ---
            if MSE_BITS == 4 and USE_PAIR_LUT:
                # OPT#1 (load-halving): FLUTE §3.2 pair LUT, but with each
                # packed byte loaded EXACTLY ONCE instead of twice (the
                # previous code built byte_addrs from d_offs // 2 which has
                # the pattern [0,0,1,1,2,2,...], duplicating every byte
                # load). We load the packed bytes as a [TILE_SIZE, HALF_D]
                # tile, decode both nibbles, then issue a single 3-D gather
                # against pair_lut which returns (T[lo], T[hi]) in slots
                # 0/1. Reshape [TILE_SIZE, HALF_D, 2] -> [TILE_SIZE, BLOCK_D]
                # interleaves the pairs into the expected dim order
                # (d=2k -> T[lo_k], d=2k+1 -> T[hi_k]).
                HALF_D: tl.constexpr = BLOCK_D // 2
                half_offs = tl.arange(0, HALF_D)
                # Each byte k covers dims (2k, 2k+1); it is in-range iff
                # 2k < HEAD_DIM. HEAD_DIM is always even in our layouts,
                # so this is equivalent to k < HEAD_DIM // 2.
                byte_mask = (half_offs * 2) < HEAD_DIM

                byte_addrs = data_bases[:, None] + half_offs[None, :]
                byte_raw = tl.load(
                    KV_cache_ptr + byte_addrs,
                    mask=kv_mask_1d[:, None] & byte_mask[None, :],
                    other=0,
                ).to(tl.int32)

                lo_idx = byte_raw & 0xF  # [TILE_SIZE, HALF_D]
                hi_idx = (byte_raw >> 4) & 0xF  # [TILE_SIZE, HALF_D]
                pair_key = lo_idx * N_CENTROIDS + hi_idx  # [TILE_SIZE, HALF_D]

                # Single 3-D gather: pair_lut[pair_key*2 + {0,1}]
                # -> [TILE_SIZE, HALF_D, 2] = (T[lo_k], T[hi_k]) per element.
                pair_slot = tl.arange(0, 2)
                c_pair = tl.load(
                    Pair_lut_ptr + pair_key[:, :, None] * 2 + pair_slot[None, None, :],
                    mask=(kv_mask_1d[:, None, None] & byte_mask[None, :, None]),
                    other=0.0,
                )
                # Row-major reshape interleaves pairs:
                #   out[t, 2k]   = c_pair[t, k, 0] = T[lo_k]
                #   out[t, 2k+1] = c_pair[t, k, 1] = T[hi_k]
                c_vals = tl.reshape(c_pair, [TILE_SIZE, BLOCK_D])
            elif MSE_BITS == 4:
                # 4-bit without pair LUT: exploit byte alignment
                half_idx = d_offs // 2
                nibble_shift = (d_offs % 2) * 4
                mse_addrs = data_bases[:, None] + half_idx[None, :]
                mse_raw = tl.load(
                    KV_cache_ptr + mse_addrs,
                    mask=kv_mask_1d[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)
                mse_idx = (mse_raw >> nibble_shift[None, :]) & 0xF
                c_vals = tl.load(
                    Centroids_ptr + mse_idx,
                    mask=kv_mask_1d[:, None] & d_mask[None, :],
                    other=0.0,
                )
            else:
                # Generic 3-bit path: two byte loads → 16-bit → shift/mask
                mse_addrs0 = data_bases[:, None] + mse_byte_idx[None, :]
                mse_raw0 = tl.load(
                    KV_cache_ptr + mse_addrs0,
                    mask=kv_mask_1d[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)
                mse_raw1 = tl.load(
                    KV_cache_ptr + mse_addrs0 + 1,
                    mask=kv_mask_1d[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)
                raw16 = mse_raw0 | (mse_raw1 << 8)
                mse_idx = (raw16 >> mse_bit_shift[None, :]) & mse_mask_val
                c_vals = tl.load(
                    Centroids_ptr + mse_idx,
                    mask=kv_mask_1d[:, None] & d_mask[None, :],
                    other=0.0,
                )

            # Opt#1: norm-correction is pre-folded into the stored per-token
            # scalar at store time, so no per-tile sum+sqrt+divide here. The
            # stored vec_norm carries ||k||/||c_vec||; multiply by c_vals
            # below gives the original c_vec/||c_vec|| * ||k||.

            # Opt#3: K-norms from per-block SoA region. Single u16 load per
            # token; contiguous across tokens within a block.
            norm_u16 = tl.load(
                KV_cache_u16_ptr + knorm_u16_addrs, mask=kv_mask_1d, other=0
            )
            vec_norms = norm_u16.to(tl.float16, bitcast=True).to(tl.float32)

            # Reconstruct K: K_recon[t, d] = norm[t] * centroid[t, d]
            K_recon = c_vals * vec_norms[:, None]
            # K_T: [BLOCK_D, TILE_SIZE]
            K_T = tl.trans(K_recon)

            # S: [BLOCK_M, TILE_SIZE] via tensor core MMA
            if USE_BF16_DOT:
                S = QK_SCALE * tl.dot(Q.to(tl.bfloat16), K_T.to(tl.bfloat16))
            else:
                S = QK_SCALE * tl.dot(Q.to(tl.float16), K_T.to(tl.float16))

        # Mask out-of-range positions
        S = tl.where(kv_mask_1d[None, :], S, float("-inf"))

        # ============================================================
        # ONLINE SOFTMAX with exp2
        # ============================================================
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.math.exp2(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.math.exp2(M - m_j)

        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        # ============================================================
        # VALUE DEQUANT → V: [TILE_SIZE, BLOCK_D] for tl.dot(P, V)
        # Opt#3: V data lives at data_base + KEY_DATA_BYTES; scale/zero
        # move to the SoA metadata region (one u16 load each per token).
        # ============================================================
        val_bases = data_bases + KEY_DATA_BYTES

        if VQB == 3:
            val_addrs0 = val_bases[:, None] + val_byte_idx[None, :]
            val_raw0 = tl.load(
                KV_cache_ptr + val_addrs0,
                mask=kv_mask_1d[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            val_raw1 = tl.load(
                KV_cache_ptr + val_addrs0 + 1,
                mask=kv_mask_1d[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = val_raw0 | (val_raw1 << 8)
            v_idx = ((raw16 >> val_bit_shift[None, :]) & 0x7).to(tl.float32)
        else:  # VQB == 4
            # OPT#3 (load-halving, value path): same pattern as OPT#1 for
            # keys -- load each byte exactly once as [TILE_SIZE, HALF_D],
            # decode both nibbles, then tl.interleave to [TILE_SIZE, BLOCK_D]
            # with pattern [v_lo0, v_hi0, v_lo1, v_hi1, ...].
            V_HALF_D: tl.constexpr = BLOCK_D // 2
            v_half_offs = tl.arange(0, V_HALF_D)
            v_byte_mask = (v_half_offs * 2) < HEAD_DIM

            val_addrs = val_bases[:, None] + v_half_offs[None, :]
            val_byte = tl.load(
                KV_cache_ptr + val_addrs,
                mask=kv_mask_1d[:, None] & v_byte_mask[None, :],
                other=0,
            ).to(tl.int32)
            v_lo = (val_byte & 0xF).to(tl.float32)  # [T, HALF_D]
            v_hi = ((val_byte >> 4) & 0xF).to(tl.float32)  # [T, HALF_D]
            v_idx = tl.interleave(v_lo, v_hi)  # [T, BLOCK_D]

        scale_u16 = tl.load(
            KV_cache_u16_ptr + vscale_u16_addrs, mask=kv_mask_1d, other=0
        )
        zero_u16 = tl.load(KV_cache_u16_ptr + vzero_u16_addrs, mask=kv_mask_1d, other=0)
        v_scales = scale_u16.to(tl.float16, bitcast=True).to(tl.float32)
        v_zeros = zero_u16.to(tl.float16, bitcast=True).to(tl.float32)
        V = v_idx * v_scales[:, None] + v_zeros[:, None]

        # P·V accumulation via tensor core MMA
        # P: [BLOCK_M, TILE_SIZE], V: [TILE_SIZE, BLOCK_D]
        if USE_BF16_DOT:
            acc += tl.dot(P.to(tl.bfloat16), V.to(tl.bfloat16))
        else:
            acc += tl.dot(P.to(tl.float16), V.to(tl.float16))

    # ================================================================
    # EPILOGUE: Store per-Q-head partial results for stage2
    # ================================================================
    # acc: [BLOCK_M, BLOCK_D], L: [BLOCK_M], M: [BLOCK_M]
    # Stage2 expects per-Q-head: mid_o[B, Hq, split, D+1]
    safe_L = tl.where(L > 0.0, L, 1.0)
    one_over_L = 1.0 / safe_L
    acc = acc * one_over_L[:, None]

    # Convert M from log2 space back to natural log for stage2 compatibility
    lse = M * LN2 + tl.log(safe_L)

    # Store each Q head's result using 2D scatter
    # q_head_offs: [BLOCK_M] = kv_hid * KV_GROUP_SIZE + m_offs
    out_addrs = (
        bid * stride_mid_b
        + q_head_offs[:, None] * stride_mid_h
        + sid * stride_mid_s
        + d_offs[None, :]
    )
    tl.store(
        Mid_o_ptr + out_addrs,
        acc,
        mask=q_mask[:, None] & d_mask[None, :],
    )
    # Store LSE per Q head
    lse_addrs = (
        bid * stride_mid_b + q_head_offs * stride_mid_h + sid * stride_mid_s + HEAD_DIM
    )
    tl.store(Mid_o_ptr + lse_addrs, lse, mask=q_mask)


# ---------------------------------------------------------------------------
# Launcher v2
# ---------------------------------------------------------------------------

_layout_cache: dict = {}


def _get_layout(D, mse_bits, value_quant_bits, key_packed_size):
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


def _get_pair_lut(centroids: torch.Tensor) -> torch.Tensor:
    """Return a fresh pair-LUT for ``centroids`` on each call.

    The LUT is tiny (N*N*2 fp32, e.g. 2KB for 4-bit MSE) so the build cost
    is negligible compared to attention. We avoid caching by data_ptr()
    because CUDA allocator memory reuse across different centroid tensors
    can silently return a stale LUT (subtle correctness bug). If this ever
    shows up on a profile, cache by a hash-of-values fingerprint instead.
    """
    return build_pair_lut(centroids)


def triton_turboquant_decode_attention_v2(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    Pi: torch.Tensor,
    centroids: torch.Tensor,
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    value_packed_size: int,
    max_seq_len: int = 0,  # unused; kept for backward compatibility
    key_fp8: bool = False,
    norm_correction: bool = False,
    PiT: torch.Tensor | None = None,
    # Pre-allocated buffers (optional, required for CUDA-graph stability).
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    buf_holder: Any = None,
    # Fixed split count — MUST be a compile-time constant across iterations
    # for CUDA-graph capture/replay to work. Mirrors the v1 launcher contract.
    max_num_kv_splits: int = 32,
) -> torch.Tensor:
    """Launch optimized TQ decode attention (v2 stage1 + shared stage2).

    Follows the same buffer-reuse + fixed-grid contract as the v1 launcher
    (triton_turboquant_decode_attention) so the backend can capture a CUDA
    graph across both versions.
    """
    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    padded_slot = kv_cache.shape[3]
    max_num_blocks = block_table.shape[1]
    n_centroids = centroids.shape[0]
    kv_group_size = Hq // Hk
    device = query.device
    del max_seq_len  # no longer used: splits is fixed via max_num_kv_splits

    cfg = _get_layout(D, mse_bits, value_quant_bits, key_packed_size)

    # Opt#3 SoA layout constants (match store-side computation).
    key_data_bytes = D if key_fp8 else cfg["mse_bytes"]
    data_bytes_per_slot = key_data_bytes + cfg["val_data_bytes"]
    meta_region_offset = block_size * Hk * data_bytes_per_slot
    num_soa_fields = 2 if key_fp8 else 3
    soa_k_norm = 0
    soa_v_scale = 0 if key_fp8 else 1
    soa_v_zero = 1 if key_fp8 else 2
    kv_cache_u16 = kv_cache.view(torch.uint16)

    # Compute q_rot = q @ Pi.T
    if key_fp8:
        q_rot = query.float().contiguous()
    else:
        q_float = query.float()
        if PiT is None:
            PiT = Pi.T.contiguous()
        q_rot = (q_float @ PiT).contiguous()

    # BLOCK_M: pad KV_GROUP_SIZE to power of 2, minimum 16 for tensor cores
    BLOCK_M = max(16, triton.next_power_of_2(kv_group_size))

    # TILE_SIZE (BLOCK_KV): tokens per inner-loop iteration
    TILE_SIZE = 16

    # Fixed split count — must be constant across calls for cudagraph replay.
    NUM_KV_SPLITS = max_num_kv_splits

    # --- mid_o buffer reuse (same pattern as v1 launcher) ---
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

    fp8_e4b15 = _use_fp8_e4b15(device.index or 0)

    # Build pair LUT for 4-bit MSE (FLUTE §3.2)
    use_pair_lut = mse_bits == 4 and not key_fp8
    pair_lut = _get_pair_lut(centroids) if use_pair_lut else centroids

    # Platform-dependent pipelining depth: ROCm prefers num_stages=1.
    stage1_num_stages = 1 if _is_hip else 2

    # --- Stage 1: v2 kernel ---
    # Grid over KV heads (not Q heads) — each program handles all Q heads
    grid = (B, Hk, NUM_KV_SPLITS)
    _tq_decode_stage1_v2[grid](
        q_rot,
        kv_cache,
        kv_cache_u16,
        block_table,
        seq_lens,
        centroids,
        pair_lut,
        mid_o,
        q_rot.stride(0),
        q_rot.stride(1),
        kv_cache.stride(0),
        block_table.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        NUM_Q_HEADS=Hq,
        NUM_KV_HEADS=Hk,
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        PADDED_SLOT=padded_slot,
        MAX_NUM_BLOCKS=max_num_blocks,
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
        N_CENTROIDS=n_centroids,
        ATTN_SCALE=scale,
        BLOCK_D=cfg["BLOCK_D"],
        TILE_SIZE=TILE_SIZE,
        BLOCK_M=BLOCK_M,
        KEY_FP8=1 if key_fp8 else 0,
        NORM_CORRECTION=1 if norm_correction else 0,
        FP8_E4B15=fp8_e4b15,
        USE_PAIR_LUT=1 if use_pair_lut else 0,
        USE_BF16_DOT=1 if _is_hip else 0,
        num_warps=4,
        num_stages=stage1_num_stages,
    )

    # --- output / lse buffer reuse (same pattern as v1 launcher) ---
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

    # --- Stage 2: reduce across KV splits (unchanged) ---
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
