# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import NVFP4_QUANT_BLOCK_SIZE


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
        tgt_base = block_idx * block_stride + block_offset * page_stride
        tgt_idx_k = tgt_base + tile_pos
        tgt_idx_v = tgt_base + tile_pos

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

    assert kv_cache_dtype == "auto" or kv_cache_dtype.startswith("fp8"), (
        f"unsupported kv_cache_dtype (str), got {kv_cache_dtype}."
    )
    kv_cache_torch_dtype = (
        current_platform.fp8_dtype()
        if kv_cache_dtype.startswith("fp8")
        else key_cache.dtype
    )

    if key_cache.dtype != kv_cache_torch_dtype and kv_cache_dtype.startswith("fp8"):
        # to avoid erounous implicit cast in triton kernel (tl.store to uint8)
        # (e.g. explicit cast to fp8e4m3fnuz is not supported in triton 3.4)
        key_cache = key_cache.view(kv_cache_torch_dtype)
        value_cache = value_cache.view(kv_cache_torch_dtype)
    assert kv_cache_dtype != torch.uint8, (
        "explicit fp8 cast and store to "
        "uint8 is not supported by triton reshape_and_cache_flash"
    )

    FP8_KV_CACHE = kv_cache_dtype.startswith("fp8")
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
        # FP8 flags
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

    assert kv_cache_dtype == "auto" or kv_cache_dtype.startswith("fp8"), (
        f"unsupported kv_cache_dtype (str), got {kv_cache_dtype}."
    )
    kv_cache_torch_dtype = (
        current_platform.fp8_dtype()
        if kv_cache_dtype.startswith("fp8")
        else kv_cache.dtype
    )

    if kv_cache.dtype != kv_cache_torch_dtype and kv_cache_dtype.startswith("fp8"):
        # to avoid erounous implicit cast in triton kernel (tl.store to uint8)
        # (e.g. explicit cast to fp8e4m3fnuz is not supported in triton 3.4)
        kv_cache = kv_cache.view(kv_cache_torch_dtype)
    assert kv_cache_dtype != torch.uint8, (
        "explicit fp8 cast and store to "
        "uint8 is not supported by triton reshape_and_cache_flash_diffkv"
    )

    FP8_KV_CACHE = kv_cache_dtype.startswith("fp8")
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


# ===================== NVFP4 KV Cache Support =====================
# NVFP4 (E2M1) uses 4 bits per element, packed 2 per byte as uint8.
# Block scales are FP8 (E4M3), one per NVFP4_QUANT_BLOCK_SIZE elements.
#
# Cache layout per head per token (contiguous in the last dim):
#   [packed_fp4_data (head_size // 2 bytes)] [block_scales (head_size // 16 bytes)]
#
# FP4 E2M1 encoding (4 bits): [sign(1) | exponent(2) | mantissa(1)]
#   Representable magnitudes: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0

FP4_E2M1_MAX = 6.0


@triton.jit
def _fp4_e2m1_quantize(val):
    """Quantize a float value to FP4 E2M1 (4-bit), returning 4-bit codes.

    E2M1 format: [sign | exp(2) | mantissa(1)]
    Representable magnitudes: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    """
    sign = tl.where(val < 0, 8, 0).to(tl.int32)
    abs_val = tl.abs(val)

    # Clamp to representable range
    abs_val = tl.minimum(abs_val, 6.0)

    # Round to nearest representable value using midpoint thresholds
    code = tl.zeros_like(abs_val, dtype=tl.int32)
    code = tl.where(abs_val >= 0.25, 1, code)   # 0.5
    code = tl.where(abs_val >= 0.75, 2, code)   # 1.0
    code = tl.where(abs_val >= 1.25, 3, code)   # 1.5
    code = tl.where(abs_val >= 1.75, 4, code)   # 2.0
    code = tl.where(abs_val >= 2.5, 5, code)    # 3.0
    code = tl.where(abs_val >= 3.5, 6, code)    # 4.0
    code = tl.where(abs_val >= 5.0, 7, code)    # 6.0

    return sign | code


@triton.jit
def reshape_and_cache_nvfp4_kernel(
    key_ptr,             # [num_tokens, num_heads, head_size] in bf16/fp16
    value_ptr,           # [num_tokens, num_heads, head_size] in bf16/fp16
    key_cache_ptr,       # [num_blocks, block_size, num_heads, eff_head_size] uint8
    value_cache_ptr,     # [num_blocks, block_size, num_heads, eff_head_size] uint8
    slot_mapping_ptr,    # [num_tokens]
    k_scale_ptr,         # global K scale (float32 tensor, scalar)
    v_scale_ptr,         # global V scale (float32 tensor, scalar)
    # strides (for source tensors, in elements)
    key_stride: tl.int64,
    value_stride: tl.int64,
    # strides (for cache tensors, in bytes since uint8)
    cache_block_stride: tl.int64,
    cache_page_stride: tl.int64,
    cache_head_stride: tl.int64,
    # constexpr
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    packed_head_size: tl.constexpr,    # head_size // 2
    scale_head_size: tl.constexpr,     # head_size // QUANT_GROUP_SIZE
):
    """Reshape, quantize to NVFP4, and cache K/V vectors.

    For each head, the cache stores (contiguously):
      - packed FP4 data: head_size // 2 bytes (2 elements per byte)
      - block scales: head_size // QUANT_GROUP_SIZE bytes (FP8 E4M3)

    Each program instance handles one token and one quantization group
    (QUANT_GROUP_SIZE elements) across all heads.
    """
    token_idx = tl.program_id(axis=0)
    head_idx = tl.program_id(axis=1)
    group_idx = tl.program_id(axis=2)

    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    # Source offset for this head's group
    src_key_base = (token_idx * key_stride
                    + head_idx * head_size
                    + group_idx * QUANT_GROUP_SIZE)
    src_val_base = (token_idx * value_stride
                    + head_idx * head_size
                    + group_idx * QUANT_GROUP_SIZE)

    # Cache base offset for this token's head
    cache_base_k = (block_idx * cache_block_stride
                    + block_offset * cache_page_stride
                    + head_idx * cache_head_stride)
    cache_base_v = (block_idx * cache_block_stride
                    + block_offset * cache_page_stride
                    + head_idx * cache_head_stride)

    # Load global scale factors
    global_k_sf = tl.load(k_scale_ptr)
    global_v_sf = tl.load(v_scale_ptr)

    # Load the group of elements
    group_offs = tl.arange(0, QUANT_GROUP_SIZE)
    k_vals = tl.load(key_ptr + src_key_base + group_offs).to(tl.float32)
    v_vals = tl.load(value_ptr + src_val_base + group_offs).to(tl.float32)

    # Compute per-group absmax
    k_absmax = tl.max(tl.abs(k_vals))
    v_absmax = tl.max(tl.abs(v_vals))

    # Compute block scales: scale = absmax / (FP4_MAX * global_sf)
    # Dequant formula: value = fp4_code * block_scale * global_sf
    k_block_scale = k_absmax / (6.0 * global_k_sf + 1e-12)
    v_block_scale = v_absmax / (6.0 * global_v_sf + 1e-12)

    # Scale values to FP4 range
    k_scaled = k_vals / (k_block_scale * global_k_sf + 1e-12)
    v_scaled = v_vals / (v_block_scale * global_v_sf + 1e-12)

    # Quantize to FP4 E2M1
    k_codes = _fp4_e2m1_quantize(k_scaled)
    v_codes = _fp4_e2m1_quantize(v_scaled)

    # Pack pairs of FP4 values into uint8: byte[i] = code[2i] | (code[2i+1] << 4)
    pair_offs = tl.arange(0, QUANT_GROUP_SIZE // 2)
    k_lo = k_codes[pair_offs * 2] & 0xF
    k_hi = (k_codes[pair_offs * 2 + 1] & 0xF) << 4
    k_packed = (k_lo | k_hi).to(tl.uint8)

    v_lo = v_codes[pair_offs * 2] & 0xF
    v_hi = (v_codes[pair_offs * 2 + 1] & 0xF) << 4
    v_packed = (v_lo | v_hi).to(tl.uint8)

    # Store packed FP4 data
    packed_offset = group_idx * (QUANT_GROUP_SIZE // 2)
    tl.store(key_cache_ptr + cache_base_k + packed_offset + pair_offs,
             k_packed)
    tl.store(value_cache_ptr + cache_base_v + packed_offset + pair_offs,
             v_packed)

    # Store block scales as FP8 E4M3 (bitcast to uint8)
    scale_offset = packed_head_size + group_idx
    k_scale_fp8 = k_block_scale.to(tl.float8e4nv)
    v_scale_fp8 = v_block_scale.to(tl.float8e4nv)
    tl.store(key_cache_ptr + cache_base_k + scale_offset,
             k_scale_fp8.to(tl.uint8, bitcast=True))
    tl.store(value_cache_ptr + cache_base_v + scale_offset,
             v_scale_fp8.to(tl.uint8, bitcast=True))


def triton_reshape_and_cache_nvfp4(
    key: torch.Tensor,         # [num_tokens, num_heads, head_size]
    value: torch.Tensor,       # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor,   # [num_blocks, block_size, num_heads, eff_head_size]
    value_cache: torch.Tensor, # [num_blocks, block_size, num_heads, eff_head_size]
    slot_mapping: torch.Tensor,  # [num_tokens]
    k_scale: torch.Tensor,    # global K scale (float32)
    v_scale: torch.Tensor,    # global V scale (float32)
):
    """Quantize K/V to NVFP4 and store in paged KV cache.

    The cache layout per head per token is:
      [packed_fp4_data (head_size//2 bytes)] [block_scales (head_size//16 bytes)]

    Args:
        key: Input key tensor in BF16/FP16.
        value: Input value tensor in BF16/FP16.
        key_cache: Pre-allocated key cache (uint8).
        value_cache: Pre-allocated value cache (uint8).
        slot_mapping: Maps token index to cache slot.
        k_scale: Global key scale factor (second-level SF).
        v_scale: Global value scale factor (second-level SF).
    """
    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_size = key.shape[2]

    assert head_size % NVFP4_QUANT_BLOCK_SIZE == 0
    packed_head_size = head_size // 2
    scale_head_size = head_size // NVFP4_QUANT_BLOCK_SIZE
    num_groups = head_size // NVFP4_QUANT_BLOCK_SIZE

    block_size = key_cache.shape[1]
    cache_block_stride = key_cache.stride(0)
    cache_page_stride = key_cache.stride(1)
    cache_head_stride = key_cache.stride(2)

    grid = (num_tokens, num_heads, num_groups)

    reshape_and_cache_nvfp4_kernel[grid](
        key_ptr=key,
        value_ptr=value,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        slot_mapping_ptr=slot_mapping,
        k_scale_ptr=k_scale,
        v_scale_ptr=v_scale,
        key_stride=key.stride(0),
        value_stride=value.stride(0),
        cache_block_stride=cache_block_stride,
        cache_page_stride=cache_page_stride,
        cache_head_stride=cache_head_stride,
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        QUANT_GROUP_SIZE=NVFP4_QUANT_BLOCK_SIZE,
        packed_head_size=packed_head_size,
        scale_head_size=scale_head_size,
        num_warps=4,
        num_stages=2,
    )
