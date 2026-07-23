# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton store kernel for OSCAR INT2 KV cache.

The expensive, dense rotation (``K @ R_k``, ``V @ R_v``) and the optional
percentile clip are done externally with cuBLAS/PyTorch — the same split
TurboQuant uses for its rotation GEMM. This kernel takes the already
rotated-and-clipped K/V and, per (token, head):

  1. computes the per-vector asymmetric INT2 quantizer (min/max -> scale,
     zero);
  2. packs four 2-bit indices per byte;
  3. scatters the packed bytes plus an fp16 ``(scale, zero)`` pair into the
     combined KV cache slot ``[key_packed | value_packed]``.

Single quantization group per vector (``group_size >= head_dim``); this is
the ``head_dim <= 128`` regime that the OSCAR presets target.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _store_int2_vec(
    Src_ptr,  # [NH, D] fp32 — rotated (+ clipped) K or V
    KV_cache_ptr,  # flattened uint8 cache
    base,  # pid * D into Src_ptr
    region_base,  # byte offset of this region within the slot (0 or key_packed)
    slot_base,  # byte offset of this slot+head in the cache
    d_offs,  # tl.arange(0, BLOCK_D)
    d_mask,  # d_offs < D
    D: tl.constexpr,
    LEVELS: tl.constexpr,  # 2 ** quant_bits (== 4 for INT2)
    DATA_BYTES: tl.constexpr,  # ceil(D * bits / 8) == D // 4 for INT2
    BLOCK_D: tl.constexpr,
    BLOCK_PACK: tl.constexpr,  # next_pow2(DATA_BYTES)
):
    """Asymmetric INT2 quantize + 4-per-byte pack + scale/zero store."""
    vec = tl.load(Src_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    vmin = tl.min(tl.where(d_mask, vec, float("inf")), axis=0)
    vmax = tl.max(tl.where(d_mask, vec, -float("inf")), axis=0)
    scale = (vmax - vmin) / (LEVELS - 1)
    scale = tl.where(scale > 1e-8, scale, 1e-8)

    # Quantize against the *fp16-rounded* scale/zero that we store, so the
    # store→load round-trip is self-consistent (avoids off-by-one-level errors
    # at bin boundaries for the very coarse INT2 grid).
    scale_f16 = scale.to(tl.float16)
    zero_f16 = vmin.to(tl.float16)
    scale = scale_f16.to(tl.float32)
    zero = zero_f16.to(tl.float32)

    # q = clamp(round((x - zero) / scale), 0, LEVELS - 1)
    q = tl.minimum(tl.maximum(((vec - zero) / scale + 0.5).to(tl.int32), 0), LEVELS - 1)

    # Pack 4 two-bit indices per byte: byte b = q[4b] | q[4b+1]<<2 | ...
    q_grp = tl.reshape(q, [BLOCK_D // 4, 4])
    shifts = tl.arange(0, 4) * 2
    packed = tl.sum((q_grp & 0x3) << shifts[None, :], axis=1).to(tl.uint8)
    pack_offs = tl.arange(0, BLOCK_PACK)
    pack_mask = pack_offs < DATA_BYTES
    tl.store(
        KV_cache_ptr + slot_base + region_base + pack_offs,
        packed,
        mask=pack_mask,
    )

    # Store fp16 scale and zero (== vmin) right after the packed data.
    meta = region_base + DATA_BYTES
    sc_u16 = scale_f16.to(tl.uint16, bitcast=True)
    tl.store(KV_cache_ptr + slot_base + meta, (sc_u16 & 0xFF).to(tl.uint8))
    tl.store(KV_cache_ptr + slot_base + meta + 1, ((sc_u16 >> 8) & 0xFF).to(tl.uint8))
    zr_u16 = zero_f16.to(tl.uint16, bitcast=True)
    tl.store(KV_cache_ptr + slot_base + meta + 2, (zr_u16 & 0xFF).to(tl.uint8))
    tl.store(KV_cache_ptr + slot_base + meta + 3, ((zr_u16 >> 8) & 0xFF).to(tl.uint8))


@triton.jit
def _oscar_store_kernel(
    Key_ptr,  # [NH, D] fp32 — rotated+clipped keys
    Value_ptr,  # [NH, D] fp32 — rotated+clipped values
    KV_cache_ptr,  # flattened uint8
    Slot_mapping_ptr,  # [N] int
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    KEY_PACKED: tl.constexpr,  # bytes of the key region (incl. its meta)
    KEY_LEVELS: tl.constexpr,
    VALUE_LEVELS: tl.constexpr,
    DATA_BYTES: tl.constexpr,  # == D // 4 for INT2 (shared by K and V)
    BLOCK_PACK: tl.constexpr,
):
    pid = tl.program_id(0)
    token_idx = pid // H
    head_idx = pid % H

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    blk = (slot // BLOCK_SIZE).to(tl.int64)
    off = (slot % BLOCK_SIZE).to(tl.int64)
    slot_base = (
        blk * stride_cache_block
        + off * stride_cache_pos
        + tl.cast(head_idx, tl.int64) * stride_cache_head
    )

    base = pid * D
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # Key region at offset 0, value region at offset KEY_PACKED.
    _store_int2_vec(
        Key_ptr,
        KV_cache_ptr,
        base,
        0,
        slot_base,
        d_offs,
        d_mask,
        D=D,
        LEVELS=KEY_LEVELS,
        DATA_BYTES=DATA_BYTES,
        BLOCK_D=BLOCK_D,
        BLOCK_PACK=BLOCK_PACK,
    )
    _store_int2_vec(
        Value_ptr,
        KV_cache_ptr,
        base,
        KEY_PACKED,
        slot_base,
        d_offs,
        d_mask,
        D=D,
        LEVELS=VALUE_LEVELS,
        DATA_BYTES=DATA_BYTES,
        BLOCK_D=BLOCK_D,
        BLOCK_PACK=BLOCK_PACK,
    )


def oscar_store(
    key_rot: torch.Tensor,  # [N, H, D] fp32/fp16 — rotated (+clipped) keys
    value_rot: torch.Tensor,  # [N, H, D] — rotated (+clipped) values
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, slot_size] uint8
    slot_mapping: torch.Tensor,  # [N]
    key_levels: int,
    value_levels: int,
    key_packed_size: int,
    data_bytes: int,
) -> None:
    """Quantize rotated K/V to INT2 and scatter into the combined cache."""
    N, H, D = key_rot.shape
    if N == 0:
        return
    NH = N * H
    block_size = kv_cache.shape[1]
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_PACK = triton.next_power_of_2(data_bytes)

    k_flat = key_rot.reshape(NH, D).contiguous().float()
    v_flat = value_rot.reshape(NH, D).contiguous().float()

    grid = (NH,)
    _oscar_store_kernel[grid](
        k_flat,
        v_flat,
        kv_cache.view(-1),
        slot_mapping,
        stride_cache_block=kv_cache.stride(0),
        stride_cache_pos=kv_cache.stride(1),
        stride_cache_head=kv_cache.stride(2),
        D=D,
        H=H,
        BLOCK_SIZE=block_size,
        BLOCK_D=BLOCK_D,
        KEY_PACKED=key_packed_size,
        KEY_LEVELS=key_levels,
        VALUE_LEVELS=value_levels,
        DATA_BYTES=data_bytes,
        BLOCK_PACK=BLOCK_PACK,
        num_warps=4,
        num_stages=1,
    )
