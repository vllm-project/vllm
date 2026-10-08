# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels for TurboQuant KV store.

Two kernels:
1. _tq_fused_store_fp8: FP8 key scatter + value uniform quantization.
2. _tq_fused_store_mse: Fused binary-search bucketize + MSE index
   packing + value quantization.

The launcher `triton_turboquant_store` selects the appropriate kernel.
"""

import math

import torch

from vllm.triton_utils import tl, triton

from .triton_turboquant_decode import _use_fp8_e4b15

# ═══════════════════════════════════════════════════════════════════════
# Shared: value uniform quantization + pack + SoA scale/zero store
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _store_quantized_value(
    Value_ptr,
    KV_cache_ptr,
    KV_cache_u16_ptr,  # Opt#3: u16 view for SoA scale/zero stores
    base,  # pid * D offset into Value_ptr
    data_base,  # byte offset into KV_cache_ptr for this slot+head's DATA region
    vscale_u16_addr,  # u16 element index for V-scale in the SoA region
    vzero_u16_addr,  # u16 element index for V-zero in the SoA region
    d_offs,  # tl.arange(0, BLOCK_D)
    d_mask,  # d_offs < D
    D: tl.constexpr,
    # Offset (bytes) from data_base to V-data region within the slot's data.
    # For MSE keys this equals MSE_BYTES; for FP8 keys it equals D.
    V_DATA_OFFSET: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_VAL: tl.constexpr,
    BLOCK_GRP: tl.constexpr,
):
    """Uniform quantization of values to VQB bits, pack, and store with
    scale/zero in the per-block SoA metadata region (Opt#3 layout)."""
    if VQB == 3:
        val_vec = tl.load(Value_ptr + base + d_offs, mask=d_mask, other=0.0).to(
            tl.float32
        )
        val_min = tl.min(tl.where(d_mask, val_vec, float("inf")), axis=0)
        val_max = tl.max(tl.where(d_mask, val_vec, -float("inf")), axis=0)
        v_scale = (val_max - val_min) / 7.0
        v_scale = tl.where(v_scale > 1e-8, v_scale, 1e-8)

        q_vals = tl.minimum(
            tl.maximum(((val_vec - val_min) / v_scale + 0.5).to(tl.int32), 0), 7
        )

        grp_offs = tl.arange(0, BLOCK_GRP)
        grp_mask = grp_offs < (D // 8)
        q_grp = tl.reshape(q_vals, [BLOCK_GRP, 8])
        shifts_3bit = tl.arange(0, 8) * 3
        packed_24 = tl.sum(q_grp << shifts_3bit[None, :], axis=1)
        b0 = (packed_24 & 0xFF).to(tl.uint8)
        b1 = ((packed_24 >> 8) & 0xFF).to(tl.uint8)
        b2 = ((packed_24 >> 16) & 0xFF).to(tl.uint8)
        tl.store(
            KV_cache_ptr + data_base + V_DATA_OFFSET + grp_offs * 3,
            b0,
            mask=grp_mask,
        )
        tl.store(
            KV_cache_ptr + data_base + V_DATA_OFFSET + grp_offs * 3 + 1,
            b1,
            mask=grp_mask,
        )
        tl.store(
            KV_cache_ptr + data_base + V_DATA_OFFSET + grp_offs * 3 + 2,
            b2,
            mask=grp_mask,
        )
    else:  # VQB == 4
        val_vec = tl.load(Value_ptr + base + d_offs, mask=d_mask, other=0.0).to(
            tl.float32
        )
        val_min = tl.min(tl.where(d_mask, val_vec, float("inf")), axis=0)
        val_max = tl.max(tl.where(d_mask, val_vec, -float("inf")), axis=0)
        v_scale = (val_max - val_min) / 15.0
        v_scale = tl.where(v_scale > 1e-8, v_scale, 1e-8)

        # Quantize all D elements from register (no re-load)
        q_all = tl.minimum(
            tl.maximum(((val_vec - val_min) / v_scale + 0.5).to(tl.int32), 0), 15
        )
        q_pairs = tl.reshape(q_all, [BLOCK_D // 2, 2])
        shifts_4 = tl.arange(0, 2) * 4
        packed_val = tl.sum((q_pairs & 0xF) << shifts_4[None, :], axis=1).to(tl.uint8)
        val_offs = tl.arange(0, BLOCK_D // 2)
        val_mask = val_offs < VAL_DATA_BYTES
        tl.store(
            KV_cache_ptr + data_base + V_DATA_OFFSET + val_offs,
            packed_val,
            mask=val_mask,
        )

    # ── SoA scale/zero store (Opt#3) ──────────────────────────────────
    # Scale and zero each get one fp16 stored at their token's slot index
    # within the per-head SoA region. Single u16 store per field vs the
    # previous 2×u8 writes per field.
    sc_u16 = v_scale.to(tl.float16).to(tl.uint16, bitcast=True)
    zr_u16 = val_min.to(tl.float16).to(tl.uint16, bitcast=True)
    tl.store(KV_cache_u16_ptr + vscale_u16_addr, sc_u16)
    tl.store(KV_cache_u16_ptr + vzero_u16_addr, zr_u16)


# ═══════════════════════════════════════════════════════════════════════
# FP8 key store + value uniform quantization
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _tq_fused_store_fp8(
    Key_ptr,  # [NH, D] float16/bfloat16 — raw keys
    Value_ptr,  # [NH, D] float16/bfloat16 — raw values
    KV_cache_ptr,  # [total_bytes] uint8 (flattened view)
    KV_cache_u16_ptr,  # uint16-aliased view of the same storage (Opt#3)
    Slot_mapping_ptr,  # [N] int32 — per-token slot indices
    # Cache strides (stride_cache_block = bytes per block = bs*H*slot_aligned)
    stride_cache_block: tl.constexpr,
    # Dimensions
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    # SoA layout (Opt#3): data region is [bs, H, KEY_DATA_BYTES+VAL_DATA_BYTES],
    # followed by metadata region [H, NUM_SOA_FIELDS, bs] of fp16.
    KEY_DATA_BYTES: tl.constexpr,  # FP8: = D
    VAL_DATA_BYTES: tl.constexpr,
    META_REGION_OFFSET: tl.constexpr,  # bytes from block start to SoA region
    NUM_SOA_FIELDS: tl.constexpr,  # FP8: 2 (V-scale, V-zero)
    SOA_V_SCALE: tl.constexpr,  # FP8: 0
    SOA_V_ZERO: tl.constexpr,  # FP8: 1
    # Value quantization
    VQB: tl.constexpr,
    # Packing block sizes
    BLOCK_VAL: tl.constexpr,
    BLOCK_GRP: tl.constexpr = 16,
    FP8_E4B15: tl.constexpr = 0,  # 1 = e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)
):
    """FP8 key cast+scatter + value uniform quantization (SoA layout)."""
    pid = tl.program_id(0)
    token_idx = pid // H
    head_idx = pid % H

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    blk = (slot // BLOCK_SIZE).to(tl.int64)
    off = (slot % BLOCK_SIZE).to(tl.int64)
    head_idx_i64 = tl.cast(head_idx, tl.int64)

    # Block start byte address — stride_cache_block is unchanged (same total
    # bytes per block as the AoS layout).
    block_base = blk * stride_cache_block

    # Data region: per-slot stride = H * DATA_BYTES_PER_SLOT; per-head stride
    # within a slot = DATA_BYTES_PER_SLOT.
    DATA_BYTES_PER_SLOT: tl.constexpr = KEY_DATA_BYTES + VAL_DATA_BYTES
    data_base = (
        block_base
        + off * (H * DATA_BYTES_PER_SLOT)
        + head_idx_i64 * DATA_BYTES_PER_SLOT
    )

    # SoA metadata region (u16 element indexing): head strip = NUM_SOA_FIELDS
    # consecutive bs-long fp16 arrays.
    head_meta_u16_base = (block_base + META_REGION_OFFSET) // 2 + head_idx_i64 * (
        NUM_SOA_FIELDS * BLOCK_SIZE
    )
    vscale_u16_addr = head_meta_u16_base + SOA_V_SCALE * BLOCK_SIZE + off
    vzero_u16_addr = head_meta_u16_base + SOA_V_ZERO * BLOCK_SIZE + off

    base = pid * D

    # ── FP8 KEY: cast to FP8 in-kernel and store at data_base ────────
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    k_vals = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0)
    k_fp8 = k_vals.to(tl.float8e4b15) if FP8_E4B15 else k_vals.to(tl.float8e4nv)
    k_bytes = k_fp8.to(tl.uint8, bitcast=True)
    tl.store(KV_cache_ptr + data_base + d_offs, k_bytes, mask=d_mask)

    # ── VALUE QUANTIZE + PACK + SoA scale/zero store ─────────────────
    _store_quantized_value(
        Value_ptr,
        KV_cache_ptr,
        KV_cache_u16_ptr,
        base,
        data_base,
        vscale_u16_addr,
        vzero_u16_addr,
        d_offs,
        d_mask,
        D=D,
        V_DATA_OFFSET=KEY_DATA_BYTES,  # V follows K in the data region
        VQB=VQB,
        VAL_DATA_BYTES=VAL_DATA_BYTES,
        BLOCK_D=BLOCK_D,
        BLOCK_VAL=BLOCK_VAL,
        BLOCK_GRP=BLOCK_GRP,
    )


# ═══════════════════════════════════════════════════════════════════════
# Fused MSE store: bucketize + MSE index pack + norm store + value pack
# (eliminates 4 PyTorch kernel launches per layer vs pack-only kernel)
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _tq_fused_store_mse(
    # Post-rotation inputs
    Y_ptr,  # [NH, D] float32 — rotated normalized keys (x_hat @ PiT)
    Norms_ptr,  # [NH] float32 — key vector norms (||k||)
    Value_ptr,  # [NH, D] float32 — raw values
    # Quantization tables
    Midpoints_ptr,  # [n_centroids-1] float32
    Centroids_ptr,  # [n_centroids] float32 — used only when NORM_CORRECTION=1
    # Cache and indexing
    KV_cache_ptr,  # [total_bytes] uint8 (flattened view)
    KV_cache_u16_ptr,  # uint16-aliased view of the same storage (Opt#3)
    Slot_mapping_ptr,  # [N] int32 — per-token slot indices
    # Cache strides (stride_cache_block = bs*H*slot_aligned, unchanged)
    stride_cache_block: tl.constexpr,
    # Dimensions
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    # SoA layout (Opt#3): data region [bs, H, MSE_BYTES+VAL_DATA_BYTES],
    # then metadata region [H, NUM_SOA_FIELDS, bs] of fp16.
    MSE_BYTES: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    META_REGION_OFFSET: tl.constexpr,  # bytes
    NUM_SOA_FIELDS: tl.constexpr,  # MSE: 3 (k_norm, v_scale, v_zero)
    SOA_K_NORM: tl.constexpr,  # MSE: 0
    SOA_V_SCALE: tl.constexpr,  # MSE: 1
    SOA_V_ZERO: tl.constexpr,  # MSE: 2
    # Value quantization
    VQB: tl.constexpr,
    # Packing block sizes
    BLOCK_VAL: tl.constexpr,
    # MSE params
    MSE_BITS: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    # When 1, fold the per-token centroid-vector norm ||c_t|| into the
    # stored scalar: store ||k_t|| / ||c_t|| instead of ||k_t||. This
    # removes the per-tile sum+sqrt+divide cost from the attention kernel's
    # K-load path. Load-side math becomes a plain multiply K = c * stored.
    NORM_CORRECTION: tl.constexpr = 0,
    BLOCK_GRP: tl.constexpr = 16,
):
    """Fused MSE quantize + pack + SoA-metadata store (Opt#3 layout).

    Binary-search bucketize → MSE-index pack → K-norm store (SoA) → value
    uniform quantize + pack → V scale/zero store (SoA), in one kernel.
    """
    pid = tl.program_id(0)
    token_idx = pid // H
    head_idx = pid % H

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    blk = (slot // BLOCK_SIZE).to(tl.int64)
    off = (slot % BLOCK_SIZE).to(tl.int64)
    head_idx_i64 = tl.cast(head_idx, tl.int64)

    # Block start byte address — stride unchanged from AoS (same total
    # bytes per block); layout inside the block differs.
    block_base = blk * stride_cache_block

    # Data region offset (packed K indices + packed V data).
    DATA_BYTES_PER_SLOT: tl.constexpr = MSE_BYTES + VAL_DATA_BYTES
    data_base = (
        block_base
        + off * (H * DATA_BYTES_PER_SLOT)
        + head_idx_i64 * DATA_BYTES_PER_SLOT
    )

    # SoA metadata addresses (u16 element indexing).
    head_meta_u16_base = (block_base + META_REGION_OFFSET) // 2 + head_idx_i64 * (
        NUM_SOA_FIELDS * BLOCK_SIZE
    )
    knorm_u16_addr = head_meta_u16_base + SOA_K_NORM * BLOCK_SIZE + off
    vscale_u16_addr = head_meta_u16_base + SOA_V_SCALE * BLOCK_SIZE + off
    vzero_u16_addr = head_meta_u16_base + SOA_V_ZERO * BLOCK_SIZE + off

    base = pid * D
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # ── 1. BINARY SEARCH BUCKETIZE ────────────────────────────────────
    y_vec = tl.load(Y_ptr + base + d_offs, mask=d_mask, other=0.0)
    lo = tl.zeros([BLOCK_D], dtype=tl.int32)
    hi = tl.full([BLOCK_D], N_CENTROIDS - 1, dtype=tl.int32)
    for _ in range(MSE_BITS):
        mid = (lo + hi) >> 1
        safe_mid = tl.minimum(mid, N_CENTROIDS - 2)
        mid_val = tl.load(Midpoints_ptr + safe_mid, mask=d_mask, other=0.0)
        lo = tl.where(y_vec >= mid_val, mid + 1, lo)
        hi = tl.where(y_vec >= mid_val, hi, mid)
    idx = tl.minimum(lo, N_CENTROIDS - 1)

    # ── 2. PACK MSE INDICES into data region at data_base[0:MSE_BYTES] ──
    if MSE_BITS == 4:
        idx_pairs = tl.reshape(idx, [BLOCK_D // 2, 2])
        shifts_4 = tl.arange(0, 2) * 4
        packed = tl.sum((idx_pairs & 0xF) << shifts_4[None, :], axis=1).to(tl.uint8)
        mse_offs = tl.arange(0, BLOCK_D // 2)
        mse_mask = mse_offs < MSE_BYTES
        tl.store(KV_cache_ptr + data_base + mse_offs, packed, mask=mse_mask)

    elif MSE_BITS == 3:
        grp_offs = tl.arange(0, BLOCK_GRP)
        grp_mask = grp_offs < (D // 8)
        idx_grp = tl.reshape(idx, [BLOCK_GRP, 8])
        shifts_3 = tl.arange(0, 8) * 3
        packed_24 = tl.sum((idx_grp & 0x7) << shifts_3[None, :], axis=1)
        b0 = (packed_24 & 0xFF).to(tl.uint8)
        b1 = ((packed_24 >> 8) & 0xFF).to(tl.uint8)
        b2 = ((packed_24 >> 16) & 0xFF).to(tl.uint8)
        tl.store(KV_cache_ptr + data_base + grp_offs * 3, b0, mask=grp_mask)
        tl.store(KV_cache_ptr + data_base + grp_offs * 3 + 1, b1, mask=grp_mask)
        tl.store(KV_cache_ptr + data_base + grp_offs * 3 + 2, b2, mask=grp_mask)

    # ── 3. STORE K-norm in SoA region (single u16 store) ─────────────
    # Norm-correction (Opt#1) still folds 1/||c_vec|| into the stored value;
    # only the destination moved to the SoA K-norm array.
    vn_f32 = tl.load(Norms_ptr + pid)  # ||k_t||, fp32

    if NORM_CORRECTION:
        safe_idx = tl.minimum(tl.maximum(idx, 0), N_CENTROIDS - 1)
        c_vals = tl.load(Centroids_ptr + safe_idx, mask=d_mask, other=0.0)
        c_norm_sq = tl.sum(tl.where(d_mask, c_vals * c_vals, 0.0), axis=0)
        c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
        vn_f32 = vn_f32 * c_inv_norm

    vn_u16 = vn_f32.to(tl.float16).to(tl.uint16, bitcast=True)
    tl.store(KV_cache_u16_ptr + knorm_u16_addr, vn_u16)

    # ── 4. VALUE QUANTIZE + PACK + SoA scale/zero store ──────────────
    _store_quantized_value(
        Value_ptr,
        KV_cache_ptr,
        KV_cache_u16_ptr,
        base,
        data_base,
        vscale_u16_addr,
        vzero_u16_addr,
        d_offs,
        d_mask,
        D=D,
        V_DATA_OFFSET=MSE_BYTES,  # V follows K indices in the data region
        VQB=VQB,
        VAL_DATA_BYTES=VAL_DATA_BYTES,
        BLOCK_D=BLOCK_D,
        BLOCK_VAL=BLOCK_VAL,
        BLOCK_GRP=BLOCK_GRP,
    )


# ═══════════════════════════════════════════════════════════════════════
# Launcher
# ═══════════════════════════════════════════════════════════════════════


def triton_turboquant_store(
    key: torch.Tensor,  # [N, H, D] — raw keys (post-RoPE)
    value: torch.Tensor,  # [N, H, D] — raw values
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, padded_slot] uint8
    slot_mapping: torch.Tensor,  # [N] int32
    PiT: torch.Tensor,  # [D, D] float32
    midpoints: torch.Tensor,  # [n_centroids-1] float32
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_fp8: bool = False,
    centroids: torch.Tensor
    | None = None,  # [n_centroids] float32 — required when norm_correction=True
    norm_correction: bool = False,
):
    """Launch TQ store kernel (FP8 or MSE path).

    When ``norm_correction=True`` (default off; set by MSE norm-correction
    presets), the centroid-vector norm ||c_t|| is folded into the stored
    per-token scalar at store time (stored value = ||k_t|| / ||c_t||). The
    load-side attention kernel no longer computes per-tile sum+sqrt+divide,
    saving ~26% of TQ overhead on gpt-oss decode (ctx=4096). ``centroids``
    must be provided in this mode.
    """
    if norm_correction and not key_fp8:
        assert centroids is not None, (
            "norm_correction=True requires centroids tensor; got None. "
            "Pass the same [n_centroids] fp32 tensor used to derive midpoints."
        )
    N, H, D = key.shape
    NH = N * H
    block_size = kv_cache.shape[1]
    BLOCK_D = triton.next_power_of_2(D)
    mse_bytes = math.ceil(D * mse_bits / 8)
    n_centroids = 2**mse_bits

    val_data_bytes = math.ceil(D * value_quant_bits / 8)

    BLOCK_VAL = triton.next_power_of_2(val_data_bytes)

    # Cache strides (element_size=1 for uint8, so stride in bytes = stride()).
    # Only stride_cache_block (== bs*H*slot_aligned) is used under Opt#3;
    # per-slot and per-head strides are computed from DATA_BYTES_PER_SLOT.
    stride_block = kv_cache.stride(0)

    block_grp = triton.next_power_of_2(D // 8) if D >= 8 else 1

    # ── Opt#3 SoA layout constants (derived locally so the launcher API
    # stays unchanged). Invariant: key_packed_size + value_packed_size
    # == slot_size_aligned == (key_data + val_data) + (k_meta + v_meta).
    key_data_bytes = D if key_fp8 else mse_bytes
    data_bytes_per_slot = key_data_bytes + val_data_bytes
    # Data region occupies [0, block_size * H * data_bytes_per_slot) bytes
    # at the start of each block; metadata region starts at that offset.
    meta_region_offset = block_size * H * data_bytes_per_slot
    num_soa_fields = 2 if key_fp8 else 3
    soa_v_scale = 0 if key_fp8 else 1
    soa_v_zero = 1 if key_fp8 else 2

    # uint16-aliased view of the same storage — enables single-instruction
    # u16 writes for the SoA K-norm / V-scale / V-zero stores. The cache
    # bytes are always contiguous and 2-byte aligned under the Opt#3 layout
    # (MSE_BYTES, VAL_DATA_BYTES, meta_region_offset are all even).
    kv_cache_u16 = kv_cache.view(torch.uint16)

    # ── FP8 PATH: in-kernel FP8 cast + scatter via fp8 kernel ──
    if key_fp8:
        k_flat = key.reshape(NH, D).contiguous()
        v_flat = value.reshape(NH, D).contiguous()

        fp8_e4b15 = _use_fp8_e4b15(key.device.index or 0)

        grid = (NH,)
        _tq_fused_store_fp8[grid](
            k_flat,
            v_flat,
            kv_cache.view(-1),
            kv_cache_u16.view(-1),
            slot_mapping,
            stride_cache_block=stride_block,
            D=D,
            H=H,
            BLOCK_SIZE=block_size,
            BLOCK_D=BLOCK_D,
            KEY_DATA_BYTES=key_data_bytes,
            VAL_DATA_BYTES=val_data_bytes,
            META_REGION_OFFSET=meta_region_offset,
            NUM_SOA_FIELDS=num_soa_fields,
            SOA_V_SCALE=soa_v_scale,
            SOA_V_ZERO=soa_v_zero,
            VQB=value_quant_bits,
            BLOCK_VAL=BLOCK_VAL,
            BLOCK_GRP=block_grp,
            FP8_E4B15=fp8_e4b15,
            num_warps=4,
            num_stages=1,
        )
        return

    # ── MSE PATH: external GEMM + fused bucketize/pack kernel ──
    # Normalize + rotation GEMM externally (cuBLAS is faster than in-kernel)
    k_flat = key.float().reshape(NH, D)
    norms = k_flat.norm(dim=1, keepdim=True)
    x_hat = k_flat / (norms + 1e-8)
    y = x_hat @ PiT

    v_flat = value.float().reshape(NH, D)

    # When norm_correction=True, the kernel folds 1/||c_t|| into the
    # stored per-token K-norm so the load kernel can skip per-tile norm
    # math (Opt#1). centroids is only dereferenced inside the kernel when
    # NORM_CORRECTION=1; pass midpoints as a harmless placeholder otherwise.
    centroids_ptr = centroids if centroids is not None else midpoints
    grid = (NH,)
    _tq_fused_store_mse[grid](
        y,
        norms.squeeze(1),
        v_flat,
        midpoints,
        centroids_ptr,
        kv_cache.view(-1),
        kv_cache_u16.view(-1),
        slot_mapping,
        stride_cache_block=stride_block,
        D=D,
        H=H,
        BLOCK_SIZE=block_size,
        BLOCK_D=BLOCK_D,
        MSE_BYTES=mse_bytes,
        VAL_DATA_BYTES=val_data_bytes,
        META_REGION_OFFSET=meta_region_offset,
        NUM_SOA_FIELDS=num_soa_fields,
        SOA_K_NORM=0,
        SOA_V_SCALE=soa_v_scale,
        SOA_V_ZERO=soa_v_zero,
        VQB=value_quant_bits,
        BLOCK_VAL=BLOCK_VAL,
        MSE_BITS=mse_bits,
        N_CENTROIDS=n_centroids,
        NORM_CORRECTION=1 if norm_correction else 0,
        BLOCK_GRP=block_grp,
        num_warps=4,
        num_stages=1,
    )
