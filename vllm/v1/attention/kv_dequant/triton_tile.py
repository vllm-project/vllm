# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton KV dequantization helpers used by unified attention kernels."""

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import INT4_CODEBOOK_LEVELS
from vllm.v1.kv_cache_interface import INT4_CHANNELS_PER_SCALE, KVQuantMode

SUPPORTED_MODES: frozenset[KVQuantMode] = frozenset(
    {
        KVQuantMode.INT8_PER_TOKEN_HEAD,
        KVQuantMode.FP8_PER_TOKEN_HEAD,
        KVQuantMode.INT4_PER_TOKEN_HEAD,
    }
)

(
    INT4_CODEBOOK_0,
    INT4_CODEBOOK_1,
    INT4_CODEBOOK_2,
    INT4_CODEBOOK_3,
    INT4_CODEBOOK_4,
    INT4_CODEBOOK_5,
    INT4_CODEBOOK_6,
    INT4_CODEBOOK_7,
    INT4_CODEBOOK_8,
    INT4_CODEBOOK_9,
    INT4_CODEBOOK_10,
    INT4_CODEBOOK_11,
    INT4_CODEBOOK_12,
    INT4_CODEBOOK_13,
    INT4_CODEBOOK_14,
    INT4_CODEBOOK_15,
) = INT4_CODEBOOK_LEVELS


@triton.jit
def _int4_codebook_lookup(
    idx,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    C4: tl.constexpr,
    C5: tl.constexpr,
    C6: tl.constexpr,
    C7: tl.constexpr,
    C9: tl.constexpr,
    C10: tl.constexpr,
    C11: tl.constexpr,
    C12: tl.constexpr,
    C13: tl.constexpr,
    C14: tl.constexpr,
    C15: tl.constexpr,
):
    value = idx.to(tl.float32) * 0.0
    value = tl.where(idx == 1, C1, value)
    value = tl.where(idx == 2, C2, value)
    value = tl.where(idx == 3, C3, value)
    value = tl.where(idx == 4, C4, value)
    value = tl.where(idx == 5, C5, value)
    value = tl.where(idx == 6, C6, value)
    value = tl.where(idx == 7, C7, value)
    value = tl.where(idx == 9, C9, value)
    value = tl.where(idx == 10, C10, value)
    value = tl.where(idx == 11, C11, value)
    value = tl.where(idx == 12, C12, value)
    value = tl.where(idx == 13, C13, value)
    value = tl.where(idx == 14, C14, value)
    value = tl.where(idx == 15, C15, value)
    return value


@triton.jit
def dequant_int8_fp8_kv_tile(
    data,
    Q,
    scale_cache_ptr,
    physical_block_idx,
    seq_offset,
    kv_head_idx,
    stride_s_blk,
    stride_s_slot,
    stride_s_head,
    tile_mask,
    BLOCK_SIZE: tl.constexpr,
):
    scale_idx = (
        physical_block_idx * stride_s_blk
        + (seq_offset % BLOCK_SIZE) * stride_s_slot
        + kv_head_idx * stride_s_head
    )
    token_head_scales = tl.load(scale_cache_ptr + scale_idx, mask=tile_mask, other=1.0)
    return data.to(Q.dtype), token_head_scales.to(tl.float32)


@triton.jit
def dequant_int4_kv_tile(
    data,
    Q,
    scale_cache_ptr,
    physical_block_idx,
    seq_offset,
    kv_head_idx,
    stride_s_blk,
    stride_s_slot,
    stride_s_head,
    stride_s_group,
    offs_d,
    tile_mask,
    IS_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_SCALE: tl.constexpr,
    INT4_C1: tl.constexpr,
    INT4_C2: tl.constexpr,
    INT4_C3: tl.constexpr,
    INT4_C4: tl.constexpr,
    INT4_C5: tl.constexpr,
    INT4_C6: tl.constexpr,
    INT4_C7: tl.constexpr,
    INT4_C9: tl.constexpr,
    INT4_C10: tl.constexpr,
    INT4_C11: tl.constexpr,
    INT4_C12: tl.constexpr,
    INT4_C13: tl.constexpr,
    INT4_C14: tl.constexpr,
    INT4_C15: tl.constexpr,
):
    dim_groups = offs_d // CHANNELS_PER_SCALE
    if IS_VALUE:
        scale_idx = (
            physical_block_idx[:, None] * stride_s_blk
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_s_slot
            + kv_head_idx * stride_s_head
            + dim_groups[None, :] * stride_s_group
        )
        scale_mask = tile_mask[:, None]
        shift = ((offs_d % 2) * 4).to(tl.int32)[None, :]
    else:
        scale_idx = (
            physical_block_idx[None, :] * stride_s_blk
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_s_slot
            + kv_head_idx * stride_s_head
            + dim_groups[:, None] * stride_s_group
        )
        scale_mask = tile_mask[None, :]
        shift = ((offs_d % 2) * 4).to(tl.int32)[:, None]

    token_dim_scales = tl.load(scale_cache_ptr + scale_idx, mask=scale_mask, other=1.0)
    idx = (data.to(tl.int32) >> shift) & 0xF
    decoded = _int4_codebook_lookup(
        idx,
        INT4_C1,
        INT4_C2,
        INT4_C3,
        INT4_C4,
        INT4_C5,
        INT4_C6,
        INT4_C7,
        INT4_C9,
        INT4_C10,
        INT4_C11,
        INT4_C12,
        INT4_C13,
        INT4_C14,
        INT4_C15,
    )
    return (decoded * token_dim_scales.to(tl.float32)).to(Q.dtype)


@triton.jit
def prepare_kv_tile(
    data,
    Q,
    tensor_scale,
    scale_cache_ptr,
    physical_block_idx,
    seq_offset,
    kv_head_idx,
    stride_s_blk,
    stride_s_slot,
    stride_s_head,
    stride_s_group,
    offs_d,
    tile_mask,
    IS_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    KV_QUANT_MODE: tl.constexpr,
    CHANNELS_PER_SCALE: tl.constexpr = INT4_CHANNELS_PER_SCALE,
    INT4_C1: tl.constexpr = INT4_CODEBOOK_1,
    INT4_C2: tl.constexpr = INT4_CODEBOOK_2,
    INT4_C3: tl.constexpr = INT4_CODEBOOK_3,
    INT4_C4: tl.constexpr = INT4_CODEBOOK_4,
    INT4_C5: tl.constexpr = INT4_CODEBOOK_5,
    INT4_C6: tl.constexpr = INT4_CODEBOOK_6,
    INT4_C7: tl.constexpr = INT4_CODEBOOK_7,
    INT4_C9: tl.constexpr = INT4_CODEBOOK_9,
    INT4_C10: tl.constexpr = INT4_CODEBOOK_10,
    INT4_C11: tl.constexpr = INT4_CODEBOOK_11,
    INT4_C12: tl.constexpr = INT4_CODEBOOK_12,
    INT4_C13: tl.constexpr = INT4_CODEBOOK_13,
    INT4_C14: tl.constexpr = INT4_CODEBOOK_14,
    INT4_C15: tl.constexpr = INT4_CODEBOOK_15,
):
    unused_scales = tile_mask.to(tl.float32)

    if KV_QUANT_MODE == 1:
        if Q.dtype.is_fp8():
            return data.to(Q.dtype), unused_scales
        return (data.to(tl.float32) * tl.load(tensor_scale)).to(Q.dtype), unused_scales
    if KV_QUANT_MODE == 4:
        return (
            dequant_int4_kv_tile(
                data,
                Q,
                scale_cache_ptr,
                physical_block_idx,
                seq_offset,
                kv_head_idx,
                stride_s_blk,
                stride_s_slot,
                stride_s_head,
                stride_s_group,
                offs_d,
                tile_mask,
                IS_VALUE,
                BLOCK_SIZE,
                CHANNELS_PER_SCALE,
                INT4_C1,
                INT4_C2,
                INT4_C3,
                INT4_C4,
                INT4_C5,
                INT4_C6,
                INT4_C7,
                INT4_C9,
                INT4_C10,
                INT4_C11,
                INT4_C12,
                INT4_C13,
                INT4_C14,
                INT4_C15,
            ),
            unused_scales,
        )
    if KV_QUANT_MODE >= 2:
        return dequant_int8_fp8_kv_tile(
            data,
            Q,
            scale_cache_ptr,
            physical_block_idx,
            seq_offset,
            kv_head_idx,
            stride_s_blk,
            stride_s_slot,
            stride_s_head,
            tile_mask,
            BLOCK_SIZE,
        )
    return data.to(Q.dtype), unused_scales
