# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import is_quantized_kv_cache, nvfp4_kv_cache_split_views

FP8_MIN, FP8_MAX = get_fp8_min_max()

_NATIVE_KV_CACHE_DTYPES = {"auto", "float16", "bfloat16", "float32", "half", "float"}
_NVFP4_GROUP_SIZE = 16


def _is_supported_kv_cache_dtype(kv_cache_dtype: str) -> bool:
    return kv_cache_dtype in _NATIVE_KV_CACHE_DTYPES or is_quantized_kv_cache(
        kv_cache_dtype
    )


@triton.jit
def _round_to_e2m1_bits(x):
    sign = tl.where(x < 0.0, 8, 0)
    abs_x = tl.abs(x)
    mag = tl.full(x.shape, 0, dtype=tl.int32)
    mag = tl.where((abs_x > 0.25) & (abs_x < 0.75), 1, mag)
    mag = tl.where((abs_x >= 0.75) & (abs_x <= 1.25), 2, mag)
    mag = tl.where((abs_x > 1.25) & (abs_x < 1.75), 3, mag)
    mag = tl.where((abs_x >= 1.75) & (abs_x <= 2.5), 4, mag)
    mag = tl.where((abs_x > 2.5) & (abs_x < 3.5), 5, mag)
    mag = tl.where((abs_x >= 3.5) & (abs_x <= 5.0), 6, mag)
    mag = tl.where(abs_x > 5.0, 7, mag)
    return sign | mag


@triton.jit
def _float_to_e4m3fn_bits(x):
    x = tl.clamp(x, 0.0, 448.0)
    x_safe = tl.maximum(x, 0.0000000001)

    subnormal_mant = tl.floor(x * 512.0 + 0.5).to(tl.int32)
    subnormal_mant = tl.minimum(subnormal_mant, 7)

    exp_unbiased = tl.floor(tl.log2(x_safe)).to(tl.int32)
    exp_bits = exp_unbiased + 7
    base = tl.exp2(exp_unbiased.to(tl.float32))
    mant = tl.floor(((x / base) - 1.0) * 8.0 + 0.5).to(tl.int32)
    exp_bits += mant >> 3
    mant = mant & 7

    normal_bits = (exp_bits << 3) | mant
    bits = tl.where(x < 0.015625, subnormal_mant, normal_bits)
    bits = tl.where(x == 0.0, 0, bits)
    return bits.to(tl.uint8)


@triton.jit
def _e4m3fn_bits_to_float(bits):
    bits_i32 = bits.to(tl.int32)
    payload = bits_i32 & 0x7F
    exp_bits = (payload >> 3) & 0x0F
    mant = payload & 0x07

    subnormal = mant.to(tl.float32) / 512.0
    normal = (1.0 + mant.to(tl.float32) * 0.125) * tl.exp2(
        (exp_bits - 7).to(tl.float32)
    )
    value = tl.where(exp_bits == 0, subnormal, normal)
    value = tl.where(payload == 0, 0.0, value)
    return tl.where((bits_i32 & 0x80) != 0, -value, value)


@triton.jit
def _nvfp4_swizzled_scale_coord(
    slot_in_block,
    scale_group_idx,
    SCALE_DIM: tl.constexpr,
):
    SWIZZLE_GROUP: tl.constexpr = SCALE_DIM // 4
    swizzled_slot = (slot_in_block // 4) * 4 + (scale_group_idx // SWIZZLE_GROUP)
    swizzled_scale = (scale_group_idx % SWIZZLE_GROUP) * 4 + (slot_in_block % 4)
    return swizzled_slot, swizzled_scale


@triton.jit
def _reshape_cache_nvfp4_kernel(
    key_ptr,  # [num_tokens, num_heads, head_size]
    value_ptr,  # [num_tokens, num_heads, head_size]
    key_data_cache_ptr,  # [num_blocks, block_size, num_heads, head_size // 2]
    value_data_cache_ptr,  # [num_blocks, block_size, num_heads, head_size // 2]
    key_scale_cache_ptr,  # [num_blocks, block_size, num_heads, head_size // 16]
    value_scale_cache_ptr,  # [num_blocks, block_size, num_heads, head_size // 16]
    slot_mapping_ptr,  # [num_tokens]
    k_scale,  # float32 global dequant scale
    v_scale,  # float32 global dequant scale
    stride_key_tok: tl.int64,
    stride_key_head: tl.int64,
    stride_value_tok: tl.int64,
    stride_value_head: tl.int64,
    stride_k_data_blk: tl.int64,
    stride_k_data_slot: tl.int64,
    stride_k_data_head: tl.int64,
    stride_v_data_blk: tl.int64,
    stride_v_data_slot: tl.int64,
    stride_v_data_head: tl.int64,
    stride_k_scale_blk: tl.int64,
    stride_k_scale_slot: tl.int64,
    stride_k_scale_head: tl.int64,
    stride_k_scale_dim: tl.int64,
    stride_v_scale_blk: tl.int64,
    stride_v_scale_slot: tl.int64,
    stride_v_scale_head: tl.int64,
    stride_v_scale_dim: tl.int64,
    block_size: tl.constexpr,
    head_size: tl.constexpr,
):
    tl.static_assert(head_size % 16 == 0)
    SCALE_DIM: tl.constexpr = head_size // 16
    tl.static_assert(SCALE_DIM % 4 == 0)
    tl.static_assert(block_size % 4 == 0)

    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    scale_group_idx = tl.program_id(2)

    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    slot_in_block = slot_idx % block_size

    byte_offsets = tl.arange(0, 8)
    even_offsets = byte_offsets * 2
    odd_offsets = even_offsets + 1
    dim_base = scale_group_idx * 16

    key_vals_even = tl.load(
        key_ptr
        + token_idx * stride_key_tok
        + head_idx * stride_key_head
        + dim_base
        + even_offsets,
    ).to(tl.float32)
    key_vals_odd = tl.load(
        key_ptr
        + token_idx * stride_key_tok
        + head_idx * stride_key_head
        + dim_base
        + odd_offsets,
    ).to(tl.float32)
    value_vals_even = tl.load(
        value_ptr
        + token_idx * stride_value_tok
        + head_idx * stride_value_head
        + dim_base
        + even_offsets,
    ).to(tl.float32)
    value_vals_odd = tl.load(
        value_ptr
        + token_idx * stride_value_tok
        + head_idx * stride_value_head
        + dim_base
        + odd_offsets,
    ).to(tl.float32)

    k_global_scale = tl.load(k_scale).to(tl.float32)
    v_global_scale = tl.load(v_scale).to(tl.float32)
    k_quant_scale = tl.where(k_global_scale == 0.0, 1.0, 1.0 / k_global_scale)
    v_quant_scale = tl.where(v_global_scale == 0.0, 1.0, 1.0 / v_global_scale)

    k_vec_max = tl.maximum(
        tl.max(tl.abs(key_vals_even), axis=0),
        tl.max(tl.abs(key_vals_odd), axis=0),
    )
    v_vec_max = tl.maximum(
        tl.max(tl.abs(value_vals_even), axis=0),
        tl.max(tl.abs(value_vals_odd), axis=0),
    )

    k_block_scale_bits = _float_to_e4m3fn_bits((k_quant_scale * k_vec_max) / 6.0)
    v_block_scale_bits = _float_to_e4m3fn_bits((v_quant_scale * v_vec_max) / 6.0)

    k_block_scale_f32 = _e4m3fn_bits_to_float(k_block_scale_bits)
    v_block_scale_f32 = _e4m3fn_bits_to_float(v_block_scale_bits)
    k_output_scale = tl.where(
        k_block_scale_f32 == 0.0, 0.0, k_quant_scale / k_block_scale_f32
    )
    v_output_scale = tl.where(
        v_block_scale_f32 == 0.0, 0.0, v_quant_scale / v_block_scale_f32
    )

    key_low = _round_to_e2m1_bits(tl.clamp(key_vals_even * k_output_scale, -6.0, 6.0))
    key_high = _round_to_e2m1_bits(tl.clamp(key_vals_odd * k_output_scale, -6.0, 6.0))
    value_low = _round_to_e2m1_bits(
        tl.clamp(value_vals_even * v_output_scale, -6.0, 6.0)
    )
    value_high = _round_to_e2m1_bits(
        tl.clamp(value_vals_odd * v_output_scale, -6.0, 6.0)
    )
    key_packed = key_low | (key_high << 4)
    value_packed = value_low | (value_high << 4)

    key_data_base = (
        block_idx * stride_k_data_blk
        + slot_in_block * stride_k_data_slot
        + head_idx * stride_k_data_head
        + scale_group_idx * 8
    )
    value_data_base = (
        block_idx * stride_v_data_blk
        + slot_in_block * stride_v_data_slot
        + head_idx * stride_v_data_head
        + scale_group_idx * 8
    )
    tl.store(key_data_cache_ptr + key_data_base + byte_offsets, key_packed)
    tl.store(value_data_cache_ptr + value_data_base + byte_offsets, value_packed)

    swizzled_slot, swizzled_scale = _nvfp4_swizzled_scale_coord(
        slot_in_block, scale_group_idx, SCALE_DIM
    )
    tl.store(
        key_scale_cache_ptr
        + block_idx * stride_k_scale_blk
        + swizzled_slot * stride_k_scale_slot
        + head_idx * stride_k_scale_head
        + swizzled_scale * stride_k_scale_dim,
        k_block_scale_bits,
    )
    tl.store(
        value_scale_cache_ptr
        + block_idx * stride_v_scale_blk
        + swizzled_slot * stride_v_scale_slot
        + head_idx * stride_v_scale_head
        + swizzled_scale * stride_v_scale_dim,
        v_block_scale_bits,
    )


def triton_reshape_and_cache_flash_nvfp4(
    key: torch.Tensor,
    value: torch.Tensor,
    key_data_cache: torch.Tensor,
    value_data_cache: torch.Tensor,
    key_scale_cache: torch.Tensor,
    value_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    num_tokens, num_heads, head_size = key.shape
    if key_scale_cache.dtype != torch.uint8:
        key_scale_cache = key_scale_cache.view(torch.uint8)
    if value_scale_cache.dtype != torch.uint8:
        value_scale_cache = value_scale_cache.view(torch.uint8)

    if value.shape != key.shape:
        raise NotImplementedError("Triton NVFP4 KV cache requires K/V same shape.")
    if key_data_cache.ndim != 4 or value_data_cache.ndim != 4:
        raise NotImplementedError("Triton NVFP4 KV cache only supports NHD layout.")
    if key_data_cache.shape[1] % 4 != 0:
        raise ValueError("NVFP4 requires block_size divisible by 4.")
    if head_size % _NVFP4_GROUP_SIZE != 0:
        raise ValueError("NVFP4 requires head_size divisible by 16.")
    if (head_size // _NVFP4_GROUP_SIZE) % 4 != 0:
        raise ValueError(
            "NVFP4 requires (head_size // 16) divisible by 4 "
            "for 4x4 block scale swizzle."
        )
    if key.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("NVFP4 KV cache only supports fp16/bf16 K/V inputs.")

    block_size = key_data_cache.shape[1]
    num_scale_groups = head_size // _NVFP4_GROUP_SIZE
    grid = (num_tokens, num_heads, num_scale_groups)

    _reshape_cache_nvfp4_kernel[grid](
        key_ptr=key,
        value_ptr=value,
        key_data_cache_ptr=key_data_cache,
        value_data_cache_ptr=value_data_cache,
        key_scale_cache_ptr=key_scale_cache,
        value_scale_cache_ptr=value_scale_cache,
        slot_mapping_ptr=slot_mapping,
        k_scale=k_scale,
        v_scale=v_scale,
        stride_key_tok=key.stride(0),
        stride_key_head=key.stride(1),
        stride_value_tok=value.stride(0),
        stride_value_head=value.stride(1),
        stride_k_data_blk=key_data_cache.stride(0),
        stride_k_data_slot=key_data_cache.stride(1),
        stride_k_data_head=key_data_cache.stride(2),
        stride_v_data_blk=value_data_cache.stride(0),
        stride_v_data_slot=value_data_cache.stride(1),
        stride_v_data_head=value_data_cache.stride(2),
        stride_k_scale_blk=key_scale_cache.stride(0),
        stride_k_scale_slot=key_scale_cache.stride(1),
        stride_k_scale_head=key_scale_cache.stride(2),
        stride_k_scale_dim=key_scale_cache.stride(3),
        stride_v_scale_blk=value_scale_cache.stride(0),
        stride_v_scale_slot=value_scale_cache.stride(1),
        stride_v_scale_head=value_scale_cache.stride(2),
        stride_v_scale_dim=value_scale_cache.stride(3),
        block_size=block_size,
        head_size=head_size,
        num_warps=1,
        num_stages=4,
    )


@triton.jit
def reshape_and_cache_kernel_flash(
    key_ptr,  # [num_tokens, num_heads, head_size]
    value_ptr,  # [num_tokens, num_heads, head_size]
    key_cache_ptr,  # [num_blocks, block_size, num_heads, head_size]
    value_cache_ptr,  # [num_blocks, block_size, num_heads, head_size]
    slot_mapping_ptr,  # [num_tokens]
    k_scale,  # float32
    v_scale,  # float32
    # strides
    key_stride: tl.int64,
    value_stride: tl.int64,
    block_stride: tl.int64,
    head_stride: tl.int64,
    dim_stride_k: tl.int64,
    dim_stride_v: tl.int64,
    page_stride: tl.int64,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    x: tl.constexpr,
    USE_HEAD_MAJOR_LAYOUT: tl.constexpr,
    # FP8 flags
    FP8_KV_CACHE: tl.constexpr,
    # tune parameters
    TILE_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        # Padding token that should be ignored.
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    tile_i = tl.program_id(axis=1)
    tile_offs = tl.arange(0, TILE_SIZE)
    tile_pos = tile_i * TILE_SIZE + tile_offs
    src_key_idx = token_idx * key_stride
    src_value_idx = token_idx * value_stride

    if USE_HEAD_MAJOR_LAYOUT:
        # Decompose the tile index back into head and dim coordinates.
        cur_head = tile_pos // head_size
        cur_dim = tile_pos % head_size
        # Value addressing (4D): [Block, Head, Dim, Slot]
        tgt_idx_v = (
            block_idx * block_stride
            + cur_head * head_stride
            + cur_dim * dim_stride_v
            + block_offset * 1
        )
        # Key addressing (5D): [Block, Head, Dim//8, Slot, 8]
        tgt_idx_k = (
            block_idx * block_stride
            + cur_head * head_stride
            + (cur_dim // x) * dim_stride_k
            + block_offset * x
            + (cur_dim % x)
        )
    else:
        cur_head = tile_pos // head_size
        cur_dim = tile_pos % head_size
        tgt_idx_k = (
            block_idx * block_stride
            + block_offset * page_stride
            + cur_head * head_stride
            + cur_dim
        )
        tgt_idx_v = tgt_idx_k

    # [TILE_SIZE]
    key_load = tl.load(
        key_ptr + src_key_idx + tile_pos, mask=tile_pos < (num_heads * head_size)
    )
    if FP8_KV_CACHE:
        # tl.store will do the correct implicit cast to fp8,
        # based on the key_cache_ptr.dtype.element_ty
        key_tile = key_load if key_load.dtype.is_fp8() else key_load / tl.load(k_scale)
    else:
        key_tile = key_load

    # [TILE_SIZE]
    value_load = tl.load(
        value_ptr + src_value_idx + tile_pos, mask=tile_pos < (num_heads * head_size)
    )
    if FP8_KV_CACHE:
        if value_load.dtype.is_fp8():
            value_tile = value_load
        else:
            # tl.store will do the correct implicit cast to fp8,
            #  based on the value_cache_ptr.dtype.element_ty
            value_tile = value_load / tl.load(v_scale)
    else:
        value_tile = value_load

    tl.store(
        key_cache_ptr + tgt_idx_k,
        key_tile,
        mask=tile_pos < (num_heads * head_size),
    )
    tl.store(
        value_cache_ptr + tgt_idx_v,
        value_tile,
        mask=tile_pos < (num_heads * head_size),
    )
    return


# ---------------------------------------------------------------------------
# Per-token-head dynamic quantization kernel
# Grid: (num_tokens, NUM_KV_HEADS)
# Each program handles one (token, head) pair:
#   1. Loads K (or V) for that single head
#   2. Computes absmax across head_size → scale = absmax / QUANT_MAX
#   3. Quantizes and stores the data + per-head scale
#
# Parametrised by QUANT_MAX / QUANT_MIN so the same code path works
# for int8 (±127/128), fp8_e4m3 (±448), and other formats.
# ---------------------------------------------------------------------------
@triton.jit
def _reshape_cache_per_token_head(
    key_ptr,  # [num_tokens, num_kv_heads, head_size]
    value_ptr,  # [num_tokens, num_kv_heads, head_size_v]
    key_cache_ptr,  # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blocks, block_size, num_kv_heads, head_size_v]
    k_scale_cache_ptr,  # [num_blocks, block_size, num_kv_heads] float32
    v_scale_cache_ptr,  # [num_blocks, block_size, num_kv_heads] float32
    slot_mapping_ptr,  # [num_tokens]
    stride_key_tok: tl.int64,
    stride_key_head: tl.int64,
    stride_val_tok: tl.int64,
    stride_val_head: tl.int64,
    stride_kc_blk: tl.int64,  # key_cache stride over blocks
    stride_kc_slot: tl.int64,  # key_cache stride over slots
    stride_kc_head: tl.int64,  # key_cache stride over heads
    stride_vc_blk: tl.int64,
    stride_vc_slot: tl.int64,
    stride_vc_head: tl.int64,
    stride_ks_blk: tl.int64,  # k_scale_cache stride[0] (blocks)
    stride_ks_slot: tl.int64,  # k_scale_cache stride[1] (slots)
    stride_ks_head: tl.int64,  # k_scale_cache stride[2] (heads)
    stride_vs_blk: tl.int64,  # v_scale_cache stride[0] (blocks)
    stride_vs_slot: tl.int64,  # v_scale_cache stride[1] (slots)
    stride_vs_head: tl.int64,  # v_scale_cache stride[2] (heads)
    block_size: tl.constexpr,
    head_size: tl.constexpr,
    head_size_v: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,  # next_power_of_2(max(head_size, head_size_v))
    QUANT_MAX: tl.constexpr = 127.0,
    QUANT_MIN: tl.constexpr = -128.0,
):
    tok = tl.program_id(0)
    head = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if slot < 0:
        return

    blk = slot // block_size
    slot_in_blk = slot % block_size

    dim_offs = tl.arange(0, HEAD_SIZE_PADDED)

    # ---- Key: load one head → absmax → quantize → store -------------------
    k_mask = dim_offs < head_size
    k_h = tl.load(
        key_ptr + tok * stride_key_tok + head * stride_key_head + dim_offs,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)

    k_scale = tl.maximum(tl.max(tl.abs(k_h)) / QUANT_MAX, 1e-6)
    tl.store(
        k_scale_cache_ptr
        + blk * stride_ks_blk
        + slot_in_blk * stride_ks_slot
        + head * stride_ks_head,
        k_scale,
    )

    k_q = tl.clamp(k_h * (1.0 / k_scale), QUANT_MIN, QUANT_MAX)
    tl.store(
        key_cache_ptr
        + blk * stride_kc_blk
        + slot_in_blk * stride_kc_slot
        + head * stride_kc_head
        + dim_offs,
        k_q,
        mask=k_mask,
    )

    # ---- Value: same per-head approach ------------------------------------
    v_mask = dim_offs < head_size_v
    v_h = tl.load(
        value_ptr + tok * stride_val_tok + head * stride_val_head + dim_offs,
        mask=v_mask,
        other=0.0,
    ).to(tl.float32)

    v_scale = tl.maximum(tl.max(tl.abs(v_h)) / QUANT_MAX, 1e-6)
    tl.store(
        v_scale_cache_ptr
        + blk * stride_vs_blk
        + slot_in_blk * stride_vs_slot
        + head * stride_vs_head,
        v_scale,
    )

    v_q = tl.clamp(v_h * (1.0 / v_scale), QUANT_MIN, QUANT_MAX)
    tl.store(
        value_cache_ptr
        + blk * stride_vc_blk
        + slot_in_blk * stride_vc_slot
        + head * stride_vc_head
        + dim_offs,
        v_q,
        mask=v_mask,
    )


def triton_reshape_and_cache_flash_per_token_head_quant(
    key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
    value: torch.Tensor,  # [num_tokens, num_kv_heads, head_size_v]
    key_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_size_v]
    k_scale_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads] float32
    v_scale_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads] float32
    slot_mapping: torch.Tensor,  # [num_tokens]
):
    """Quantize key/value per (token, head) and write to paged cache.

    Computes one scale = absmax / QUANT_MAX per (token, head), stores
    quantized data in key_cache/value_cache, and stores the float32
    scale in k_scale_cache/v_scale_cache.

    The quantization range (QUANT_MAX, QUANT_MIN) is derived from the
    cache tensor dtype so the same code path works for int8 and fp8.
    """
    cache_dtype = key_cache.dtype
    if cache_dtype == torch.int8:
        quant_max, quant_min = 127.0, -128.0
    elif cache_dtype == current_platform.fp8_dtype():
        quant_max, quant_min = FP8_MAX, FP8_MIN
    else:
        raise ValueError(
            f"Per-token-head quantization not supported for cache dtype "
            f"{cache_dtype}.  Supported: "
            f"{[torch.int8, current_platform.fp8_dtype()]}"
        )

    num_tokens, num_kv_heads, head_size = key.shape
    head_size_v = value.shape[2]
    head_size_padded = triton.next_power_of_2(max(head_size, head_size_v))

    block_size = key_cache.shape[1]

    if current_platform.is_rocm() or current_platform.is_xpu():
        num_warps = 4
    else:
        num_warps = min(16, max(1, head_size_padded // 32))

    _reshape_cache_per_token_head[(num_tokens, num_kv_heads)](
        key_ptr=key,
        value_ptr=value,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        k_scale_cache_ptr=k_scale_cache,
        v_scale_cache_ptr=v_scale_cache,
        slot_mapping_ptr=slot_mapping,
        stride_key_tok=key.stride(0),
        stride_key_head=key.stride(1),
        stride_val_tok=value.stride(0),
        stride_val_head=value.stride(1),
        stride_kc_blk=key_cache.stride(0),
        stride_kc_slot=key_cache.stride(1),
        stride_kc_head=key_cache.stride(2),
        stride_vc_blk=value_cache.stride(0),
        stride_vc_slot=value_cache.stride(1),
        stride_vc_head=value_cache.stride(2),
        stride_ks_blk=k_scale_cache.stride(0),
        stride_ks_slot=k_scale_cache.stride(1),
        stride_ks_head=k_scale_cache.stride(2),
        stride_vs_blk=v_scale_cache.stride(0),
        stride_vs_slot=v_scale_cache.stride(1),
        stride_vs_head=v_scale_cache.stride(2),
        block_size=block_size,
        head_size=head_size,
        head_size_v=head_size_v,
        HEAD_SIZE_PADDED=head_size_padded,
        QUANT_MAX=quant_max,
        QUANT_MIN=quant_min,
        num_warps=num_warps,
    )


def triton_reshape_and_cache_flash(
    key: torch.Tensor,  # [num_tokens, num_heads, head_size]
    value: torch.Tensor,  # [num_tokens, num_heads, head_size]
    # [num_blocks, block_size, num_heads, head_size]
    key_cache: torch.Tensor,
    # [num_blocks, block_size, num_heads, head_size]
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,  # [num_tokens]
    kv_cache_dtype: str,  # "auto", "fp8"
    k_scale: torch.Tensor,  # float32
    v_scale: torch.Tensor,  # float32
):
    num_heads = key.shape[1]
    head_size = key.shape[2]

    if kv_cache_dtype == "nvfp4":
        if key_cache.ndim != 4 or value_cache.ndim != 4:
            raise NotImplementedError("Triton NVFP4 KV cache only supports NHD layout.")
        if key_cache.shape[2] != num_heads:
            raise NotImplementedError("Triton NVFP4 KV cache only supports NHD layout.")
        (key_data_cache,), (key_scale_cache,) = nvfp4_kv_cache_split_views(key_cache)
        (value_data_cache,), (value_scale_cache,) = nvfp4_kv_cache_split_views(
            value_cache
        )
        triton_reshape_and_cache_flash_nvfp4(
            key,
            value,
            key_data_cache,
            value_data_cache,
            key_scale_cache.view(torch.uint8),
            value_scale_cache.view(torch.uint8),
            slot_mapping,
            k_scale,
            v_scale,
        )
        return

    use_head_major_layout = key_cache.ndim == 5
    if use_head_major_layout:
        block_size = key_cache.shape[3]
        x = key_cache.shape[4]
        head_stride = key_cache.stride(1)
        dim_stride_k = key_cache.stride(2)
        dim_stride_v = value_cache.stride(2)
    else:
        block_size = key_cache.shape[1]
        x = 1
        dim_stride_k = 0
        dim_stride_v = 0
        head_stride = key_cache.stride()[2]
    n = num_heads * head_size
    key_stride = key.stride()[0]
    value_stride = value.stride()[0]
    block_stride = key_cache.stride()[0]
    page_stride = key_cache.stride()[1]

    assert _is_supported_kv_cache_dtype(kv_cache_dtype), (
        f"unsupported kv_cache_dtype (str), got {kv_cache_dtype}."
    )
    kv_cache_torch_dtype = (
        current_platform.fp8_dtype()
        if is_quantized_kv_cache(kv_cache_dtype)
        else key_cache.dtype
    )

    if key_cache.dtype != kv_cache_torch_dtype and is_quantized_kv_cache(
        kv_cache_dtype
    ):
        # to avoid erounous implicit cast in triton kernel (tl.store to uint8)
        # (e.g. explicit cast to fp8e4m3fnuz is not supported in triton 3.4)
        key_cache = key_cache.view(kv_cache_torch_dtype)
        value_cache = value_cache.view(kv_cache_torch_dtype)
    assert kv_cache_dtype != torch.uint8, (
        "explicit fp8 cast and store to "
        "uint8 is not supported by triton reshape_and_cache_flash"
    )

    FP8_KV_CACHE = is_quantized_kv_cache(kv_cache_dtype)
    assert (not FP8_KV_CACHE) or kv_cache_torch_dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.uint8,
        torch.float8_e4m3fnuz,
    ], (
        "unsupported dtype of KV cache tensor, got "
        "{kv_cache_torch_dtype}. Supported kv cache dtypes: fp8e4m3fn, "
        "fp8e5m2, uint8, bfloat16, float16, float32, fp8e4m3fnuz."
    )

    # heuristics instead of autotuning
    TILE_SIZE = min(2048, triton.next_power_of_2(n))
    if current_platform.is_rocm() or current_platform.is_xpu():
        num_stages = 4
        num_warps = 8
    else:  # cuda
        num_stages = 10
        num_warps = 16
        if torch.cuda.get_device_capability(key.device)[0] < 9:
            TILE_SIZE = min(512, TILE_SIZE)

    # TODO(ngl): maybe replace with static launch grid to avoid overhead if
    #   using cudagraphs
    grid = lambda meta: (
        slot_mapping.shape[0],
        triton.cdiv(n, meta["TILE_SIZE"]),
    )

    reshape_and_cache_kernel_flash[grid](
        key_ptr=key,
        value_ptr=value,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        slot_mapping_ptr=slot_mapping,
        k_scale=k_scale,
        v_scale=v_scale,
        # strides
        key_stride=key_stride,
        value_stride=value_stride,
        block_stride=block_stride,
        head_stride=head_stride,
        dim_stride_k=dim_stride_k,
        dim_stride_v=dim_stride_v,
        page_stride=page_stride,
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        x=x,
        USE_HEAD_MAJOR_LAYOUT=use_head_major_layout,
        FP8_KV_CACHE=FP8_KV_CACHE,
        # autotune parameters
        TILE_SIZE=TILE_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@triton.jit
def reshape_and_cache_kernel_flash_diffkv(
    key_ptr,  # [num_tokens, num_heads, head_size]
    value_ptr,  # [num_tokens, num_heads, head_size_v]
    kv_cache_ptr,  # [num_blocks, block_size, num_heads, head_size + head_size_v]
    slot_mapping_ptr,  # [num_tokens]
    k_scale,  # float32
    v_scale,  # float32
    # strides
    key_stride: tl.int64,
    value_stride: tl.int64,
    block_stride: tl.int64,
    page_stride: tl.int64,
    num_heads: tl.constexpr,
    head_size_k: tl.constexpr,
    head_size_v: tl.constexpr,
    block_size: tl.constexpr,
    # FP8 flags
    FP8_KV_CACHE: tl.constexpr,
    # tune parameters
    TILE_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        # Padding token that should be ignored.
        return

    tile_i = tl.program_id(axis=1)
    tile_offs = tl.arange(0, TILE_SIZE)

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    src_key_idx = token_idx * key_stride + tile_i * head_size_k
    src_value_idx = token_idx * value_stride + tile_i * head_size_v

    tgt_idx = (
        block_idx * block_stride
        + block_offset * page_stride
        + tile_i * (head_size_k + head_size_v)
    )

    # [TILE_SIZE]
    key_load = tl.load(key_ptr + src_key_idx + tile_offs, mask=tile_offs < head_size_k)
    if FP8_KV_CACHE:
        # tl.store will do the correct implicit cast to fp8,
        # based on the key_cache_ptr.dtype.element_ty
        key_tile = key_load if key_load.dtype.is_fp8() else key_load / tl.load(k_scale)
    else:
        key_tile = key_load

    # [TILE_SIZE]
    value_load = tl.load(
        value_ptr + src_value_idx + tile_offs, mask=tile_offs < head_size_v
    )
    if FP8_KV_CACHE:
        if value_load.dtype.is_fp8():
            value_tile = value_load
        else:
            # tl.store will do the correct implicit cast to fp8,
            #  based on the value_cache_ptr.dtype.element_ty
            value_tile = value_load / tl.load(v_scale)
    else:
        value_tile = value_load

    tl.store(
        kv_cache_ptr + tgt_idx + tile_offs,
        key_tile,
        mask=tile_offs < head_size_k,
    )
    tl.store(
        kv_cache_ptr + tgt_idx + head_size_k + tile_offs,
        value_tile,
        mask=tile_offs < head_size_v,
    )
    return


def triton_reshape_and_cache_flash_diffkv(
    key: torch.Tensor,  # [num_tokens, num_heads, head_size]
    value: torch.Tensor,  # [num_tokens, num_heads, head_size_v]
    # [num_blocks, block_size, num_heads, head_size + head_size_v]
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,  # [num_tokens]
    kv_cache_dtype: str,  # "auto", "fp8"
    k_scale: torch.Tensor,  # float32
    v_scale: torch.Tensor,  # float32
):
    num_heads = key.shape[1]
    head_size_k = key.shape[2]
    head_size_v = value.shape[2]
    block_size = kv_cache.shape[1]

    k_stride = key.stride()[0]
    v_stride = value.stride()[0]
    block_stride = kv_cache.stride()[0]
    page_stride = kv_cache.stride()[1]

    assert _is_supported_kv_cache_dtype(kv_cache_dtype), (
        f"unsupported kv_cache_dtype (str), got {kv_cache_dtype}."
    )
    kv_cache_torch_dtype = (
        current_platform.fp8_dtype()
        if is_quantized_kv_cache(kv_cache_dtype)
        else kv_cache.dtype
    )

    if kv_cache.dtype != kv_cache_torch_dtype and is_quantized_kv_cache(kv_cache_dtype):
        # to avoid erounous implicit cast in triton kernel (tl.store to uint8)
        # (e.g. explicit cast to fp8e4m3fnuz is not supported in triton 3.4)
        kv_cache = kv_cache.view(kv_cache_torch_dtype)
    assert kv_cache_dtype != torch.uint8, (
        "explicit fp8 cast and store to "
        "uint8 is not supported by triton reshape_and_cache_flash_diffkv"
    )

    FP8_KV_CACHE = is_quantized_kv_cache(kv_cache_dtype)
    assert (not FP8_KV_CACHE) or kv_cache_torch_dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.uint8,
        torch.float8_e4m3fnuz,
    ], (
        "unsupported dtype of KV cache tensor, got "
        "{kv_cache_torch_dtype}. Supported kv cache dtypes: fp8e4m3fn, "
        "fp8e5m2, uint8, bfloat16, float16, float32, fp8e4m3fnuz."
    )

    # heuristics instead of autotuning
    TILE_SIZE = max(head_size_k, head_size_v)
    TILE_SIZE = triton.next_power_of_2(TILE_SIZE)
    if current_platform.is_rocm() or current_platform.is_xpu():
        num_stages = 4
        num_warps = 8
    else:  # cuda
        num_stages = 10
        num_warps = 16

    # TODO(ngl): maybe replace with static launch grid to avoid overhead if
    #   using cudagraphs
    grid = lambda meta: (
        slot_mapping.shape[0],
        num_heads,
    )

    reshape_and_cache_kernel_flash_diffkv[grid](
        key_ptr=key,
        value_ptr=value,
        kv_cache_ptr=kv_cache,
        slot_mapping_ptr=slot_mapping,
        k_scale=k_scale,
        v_scale=v_scale,
        # strides
        key_stride=k_stride,
        value_stride=v_stride,
        block_stride=block_stride,
        page_stride=page_stride,
        num_heads=num_heads,
        head_size_k=head_size_k,
        head_size_v=head_size_v,
        block_size=block_size,
        # FP8 flags
        FP8_KV_CACHE=FP8_KV_CACHE,
        # autotune parameters
        TILE_SIZE=TILE_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
