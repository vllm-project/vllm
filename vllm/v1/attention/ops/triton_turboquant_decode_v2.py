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

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2

_FP8_E4B15: dict[int, int] = {}


def _use_fp8_e4b15(device: int = 0) -> int:
    """Return 1 if device needs fp8e4b15 (Ampere/Ada, SM < 8.9), else 0."""
    if device not in _FP8_E4B15:
        cap = torch.cuda.get_device_capability(device)
        _FP8_E4B15[device] = 1 if cap < (8, 9) else 0
    return _FP8_E4B15[device]


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
    Block_table_ptr,
    Seq_lens_ptr,
    Centroids_ptr,
    Pair_lut_ptr,
    Mid_o_ptr,
    stride_qb,
    stride_qh,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
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
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    KEY_FP8: tl.constexpr,
    NORM_CORRECTION: tl.constexpr = 0,
    FP8_E4B15: tl.constexpr = 0,
    USE_PAIR_LUT: tl.constexpr = 0,
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

        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_hid, tl.int64) * stride_cache_head
        )

        # ============================================================
        # KEY DEQUANT → K_T: [BLOCK_D, TILE_SIZE] for tl.dot(Q, K_T)
        # ============================================================
        if KEY_FP8:
            k_addrs = slot_bases[:, None] + d_offs[None, :]
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
            # S: [BLOCK_M, TILE_SIZE]
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

                byte_addrs = slot_bases[:, None] + half_offs[None, :]
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
                mse_addrs = slot_bases[:, None] + half_idx[None, :]
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
                mse_addrs0 = slot_bases[:, None] + mse_byte_idx[None, :]
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

            if NORM_CORRECTION:
                c_norm_sq = tl.sum(
                    tl.where(d_mask[None, :], c_vals * c_vals, 0.0),
                    axis=1,
                )
                c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
                c_vals = c_vals * c_inv_norm[:, None]

            # Load norms: [TILE_SIZE] fp16→fp32
            norm_bases = slot_bases + MSE_BYTES
            n_lo = tl.load(
                KV_cache_ptr + norm_bases,
                mask=kv_mask_1d,
                other=0,
            ).to(tl.uint16)
            n_hi = tl.load(
                KV_cache_ptr + norm_bases + 1,
                mask=kv_mask_1d,
                other=0,
            ).to(tl.uint16)
            vec_norms = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

            # Reconstruct K: K_recon[t, d] = norm[t] * centroid[t, d]
            K_recon = c_vals * vec_norms[:, None]
            # K_T: [BLOCK_D, TILE_SIZE]
            K_T = tl.trans(K_recon)

            # S: [BLOCK_M, TILE_SIZE] via tensor core MMA
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
        # ============================================================
        val_bases = slot_bases + KPS

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

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(
                KV_cache_ptr + sc_bases,
                mask=kv_mask_1d,
                other=0,
            ).to(tl.uint16)
            sc_hi = tl.load(
                KV_cache_ptr + sc_bases + 1,
                mask=kv_mask_1d,
                other=0,
            ).to(tl.uint16)
            v_scales = (
                (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            zr_lo = tl.load(
                KV_cache_ptr + sc_bases + 2,
                mask=kv_mask_1d,
                other=0,
            ).to(tl.uint16)
            zr_hi = tl.load(
                KV_cache_ptr + sc_bases + 3,
                mask=kv_mask_1d,
                other=0,
            ).to(tl.uint16)
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            V = v_idx * v_scales[:, None] + v_zeros[:, None]
        else:  # VQB == 4
            # OPT#3 (load-halving, value path): same pattern as OPT#1 for
            # keys -- the previous code used `vb_idx = d_offs // 2`, which
            # has the duplicate sequence [0,0,1,1,2,2,...], issuing every
            # packed-value byte load TWICE per tile. Load each byte exactly
            # once as [TILE_SIZE, HALF_D], decode both nibbles, then
            # tl.interleave to [TILE_SIZE, BLOCK_D] with pattern
            # [v_lo0, v_hi0, v_lo1, v_hi1, ...] matching the original dim
            # layout expected by V = v_idx * v_scales + v_zeros.
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

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(
                KV_cache_ptr + sc_bases,
                mask=kv_mask_1d,
                other=0,
            ).to(tl.uint16)
            sc_hi = tl.load(
                KV_cache_ptr + sc_bases + 1,
                mask=kv_mask_1d,
                other=0,
            ).to(tl.uint16)
            v_scales = (
                (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            zr_lo = tl.load(
                KV_cache_ptr + sc_bases + 2,
                mask=kv_mask_1d,
                other=0,
            ).to(tl.uint16)
            zr_hi = tl.load(
                KV_cache_ptr + sc_bases + 3,
                mask=kv_mask_1d,
                other=0,
            ).to(tl.uint16)
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            V = v_idx * v_scales[:, None] + v_zeros[:, None]

        # P·V accumulation via tensor core MMA
        # P: [BLOCK_M, TILE_SIZE], V: [TILE_SIZE, BLOCK_D]
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


_pair_lut_cache: dict = {}


def _get_pair_lut(centroids: torch.Tensor) -> torch.Tensor:
    key = centroids.data_ptr()
    lut = _pair_lut_cache.get(key)
    if lut is None:
        lut = build_pair_lut(centroids)
        _pair_lut_cache[key] = lut
    return lut


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
    num_kv_splits: int = 128,
    max_seq_len: int = 0,
    key_fp8: bool = False,
    norm_correction: bool = False,
    PiT: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch optimized TQ decode attention (v2 stage1 + shared stage2)."""
    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    padded_slot = kv_cache.shape[3]
    max_num_blocks = block_table.shape[1]
    n_centroids = centroids.shape[0]
    kv_group_size = Hq // Hk
    device = query.device

    cfg = _get_layout(D, mse_bits, value_quant_bits, key_packed_size)

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

    # Occupancy-aware NUM_KV_SPLITS
    MIN_TOKENS_PER_SPLIT = 128
    max_seq = max_seq_len if max_seq_len > 0 else num_kv_splits * MIN_TOKENS_PER_SPLIT
    effective = max(1, max_seq // MIN_TOKENS_PER_SPLIT)
    NUM_KV_SPLITS = 1
    while min(effective, num_kv_splits) >= NUM_KV_SPLITS * 2:
        NUM_KV_SPLITS *= 2

    SM_COUNT = torch.cuda.get_device_properties(device).multi_processor_count
    TARGET_GRID = SM_COUNT * 2
    grid_blocks = B * Hk * NUM_KV_SPLITS
    if grid_blocks < TARGET_GRID:
        needed = math.ceil(TARGET_GRID / (B * Hk))
        ns = NUM_KV_SPLITS
        while ns < needed and ns < 128:
            ns *= 2
        max_allowed = max(1, max_seq // 16)
        ns = min(ns, max_allowed, 128)
        final = NUM_KV_SPLITS
        while final * 2 <= ns:
            final *= 2
        NUM_KV_SPLITS = final

    mid_o = torch.empty(
        B,
        Hq,
        NUM_KV_SPLITS,
        D + 1,
        dtype=torch.float32,
        device=device,
    )

    fp8_e4b15 = _use_fp8_e4b15(device.index or 0)

    # Build pair LUT for 4-bit MSE (FLUTE §3.2)
    use_pair_lut = mse_bits == 4 and not key_fp8
    pair_lut = _get_pair_lut(centroids) if use_pair_lut else centroids

    # --- Stage 1: v2 kernel ---
    # Grid over KV heads (not Q heads) — each program handles all Q heads
    grid = (B, Hk, NUM_KV_SPLITS)
    _tq_decode_stage1_v2[grid](
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        pair_lut,
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
        KPS=key_packed_size,
        VQB=value_quant_bits,
        VAL_DATA_BYTES=cfg["val_data_bytes"],
        N_CENTROIDS=n_centroids,
        ATTN_SCALE=scale,
        BLOCK_D=cfg["BLOCK_D"],
        TILE_SIZE=TILE_SIZE,
        BLOCK_M=BLOCK_M,
        KEY_FP8=1 if key_fp8 else 0,
        NORM_CORRECTION=1 if norm_correction else 0,
        FP8_E4B15=fp8_e4b15,
        USE_PAIR_LUT=1 if use_pair_lut else 0,
        num_warps=4,
        num_stages=2,
    )

    # --- Stage 2: reduce across KV splits (unchanged) ---
    output = torch.empty(B, Hq, D, dtype=torch.float32, device=device)
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
        BLOCK_DV=cfg["BLOCK_D"],
        Lv=D,
        num_warps=4,
        num_stages=2,
    )

    return output.to(query.dtype)
