# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full KV dequant for the UltraQuant cache format.

Used by continuation prefill: when a chunk brings many new query tokens on
top of a long cached prefix, dequanting the prefix once and running a dense
prefill kernel beats replaying the decode kernel per query token.

Reads FP4 codes plus UE8M0 group scales and writes K (Hadamard-rotated, as
stored) and V (raw) into pre-allocated fp16/bf16 buffers.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.ultraquant.format import (
    FP4_BITS_TO_VALUE,
    UE8M0_BIAS,
    get_group_size,
    k_scales_offset,
    n_groups,
    v_codes_offset,
    v_scales_offset,
)
from vllm.v1.attention.ops.ultraquant.triton_store import _kv_cache_flat

_FP4_DECODE_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _get_fp4_decode_table(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """16-entry FP4 E2M1 bit-pattern -> value table (cached per device)."""
    key = (device, dtype)
    t = _FP4_DECODE_CACHE.get(key)
    if t is None:
        t = torch.tensor(FP4_BITS_TO_VALUE, device=device, dtype=dtype).contiguous()
        _FP4_DECODE_CACHE[key] = t
    return t


@triton.jit
def _ultraquant_full_dequant_kv(
    KV_cache_ptr,  # uint8 view, flat
    Block_table_ptr,  # [B, max_num_blocks] int32
    Fp4_decode_ptr,  # [16] FP4 bit pattern -> value
    K_out_ptr,
    V_out_ptr,
    stride_ko_b: tl.int64,
    stride_ko_h: tl.int64,
    stride_ko_s: tl.int64,
    stride_vo_b: tl.int64,
    stride_vo_h: tl.int64,
    stride_vo_s: tl.int64,
    stride_cache_block: tl.int64,
    stride_cache_pos: tl.int64,
    stride_cache_head: tl.int64,
    stride_bt_b: tl.int64,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    K_SCALES_OFFSET: tl.constexpr,
    V_CODES_OFFSET: tl.constexpr,
    V_SCALES_OFFSET: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    OUT_BF16: tl.constexpr,
    UE8M0_BIAS_C: tl.constexpr,
):
    pos = tl.program_id(0)
    bh = tl.program_id(1)
    bid = bh // NUM_KV_HEADS
    hid = bh % NUM_KV_HEADS

    page_idx = pos // BLOCK_SIZE
    page_off = pos % BLOCK_SIZE
    block_num = tl.load(Block_table_ptr + bid * stride_bt_b + page_idx).to(tl.int64)
    slot_base = (
        block_num * stride_cache_block
        + page_off * stride_cache_pos
        + tl.cast(hid, tl.int64) * stride_cache_head
    )

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    byte_idx = (d_offs // 2).to(tl.int32)
    nibble_shift = ((d_offs % 2) * 4).to(tl.int32)
    g_offs = tl.arange(0, N_GROUPS_C)

    out_dtype = tl.bfloat16 if OUT_BF16 else tl.float16

    # K side: codes are packed nibbles, one UE8M0 byte per group of 32.
    k_byte_raw = tl.load(KV_cache_ptr + slot_base + byte_idx, mask=d_mask, other=0).to(
        tl.int32
    )
    k_codes = (k_byte_raw >> nibble_shift) & 0xF
    k_dec = tl.load(Fp4_decode_ptr + k_codes).to(tl.float32)
    k_dec = tl.where(d_mask, k_dec, 0.0)

    # UE8M0 scale: value = 2^(byte - bias); byte == 0 encodes exact zero.
    k_scale_bytes = tl.load(KV_cache_ptr + slot_base + K_SCALES_OFFSET + g_offs).to(
        tl.int32
    )
    k_scales = tl.where(
        k_scale_bytes == 0,
        0.0,
        tl.exp2(tl.cast(k_scale_bytes - UE8M0_BIAS_C, tl.float32)),
    )

    k_g = tl.reshape(k_dec, [N_GROUPS_C, GROUP_SIZE_C])
    k_recon = tl.reshape(k_g * k_scales[:, None], [BLOCK_D])
    k_recon = tl.where(d_mask, k_recon, 0.0)

    ko_base = bid * stride_ko_b + hid * stride_ko_h + pos * stride_ko_s
    tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(out_dtype), mask=d_mask)

    # V side mirrors K but is stored unrotated.
    v_byte_raw = tl.load(
        KV_cache_ptr + slot_base + V_CODES_OFFSET + byte_idx,
        mask=d_mask,
        other=0,
    ).to(tl.int32)
    v_codes = (v_byte_raw >> nibble_shift) & 0xF
    v_dec = tl.load(Fp4_decode_ptr + v_codes).to(tl.float32)
    v_dec = tl.where(d_mask, v_dec, 0.0)

    v_scale_bytes = tl.load(KV_cache_ptr + slot_base + V_SCALES_OFFSET + g_offs).to(
        tl.int32
    )
    v_scales = tl.where(
        v_scale_bytes == 0,
        0.0,
        tl.exp2(tl.cast(v_scale_bytes - UE8M0_BIAS_C, tl.float32)),
    )

    v_g = tl.reshape(v_dec, [N_GROUPS_C, GROUP_SIZE_C])
    v_recon = tl.reshape(v_g * v_scales[:, None], [BLOCK_D])
    v_recon = tl.where(d_mask, v_recon, 0.0)

    vo_base = bid * stride_vo_b + hid * stride_vo_h + pos * stride_vo_s
    tl.store(V_out_ptr + vo_base + d_offs, v_recon.to(out_dtype), mask=d_mask)


def ultraquant_full_dequant_kv(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    alloc_len: int,
) -> None:
    """Dequant ``alloc_len`` cached positions into ``k_out`` / ``v_out``.

    ``k_out`` / ``v_out`` are ``[B, Hk, alloc_len, D]`` in fp16 or bf16.
    """
    B = block_table.shape[0]
    Hk = kv_cache.shape[2]
    D = k_out.shape[3]
    block_size = kv_cache.shape[1]

    gs = get_group_size()
    _ultraquant_full_dequant_kv[(alloc_len, B * Hk)](
        _kv_cache_flat(kv_cache),
        block_table,
        _get_fp4_decode_table(kv_cache.device, k_out.dtype),
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
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        NUM_KV_HEADS=Hk,
        K_SCALES_OFFSET=k_scales_offset(D, gs),
        V_CODES_OFFSET=v_codes_offset(D, gs),
        V_SCALES_OFFSET=v_scales_offset(D, gs),
        GROUP_SIZE_C=gs,
        N_GROUPS_C=n_groups(D, gs),
        BLOCK_D=triton.next_power_of_2(D),
        OUT_BF16=1 if k_out.dtype == torch.bfloat16 else 0,
        UE8M0_BIAS_C=UE8M0_BIAS,
        num_warps=4,
    )
