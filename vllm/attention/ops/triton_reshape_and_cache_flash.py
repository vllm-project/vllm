# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import torch
import triton
import triton.language as tl
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE


@triton.jit
def quant_symmetric_per_tensor_fp8e5(x, scale=None):
    if scale is None:
        # Compute scale
        max_val = tl.max(tl.abs(x))
        scale = max_val / 57.0
        scale = tl.where(scale == 0.0, 1.0, scale)  # Avoid div-by-zero

    # Quantize to float8e5
    x_scaled = x / scale
    x_clipped = tl.clamp(x_scaled, -57.0, 57.0)
    return x_clipped.to(tl.float8e5)

@triton.jit
def quant_symmetric_per_tensor_fp8e4nv(x, scale=None):
    if scale is None:
        # Compute scale
        max_val = tl.max(tl.abs(x))
        scale = max_val / 448.0
        scale = tl.where(scale == 0.0, 1.0, scale)  # Avoid div-by-zero

    # Quantize to float8e4nv
    x_scaled = x / scale
    x_clipped = tl.clamp(x_scaled, -448.0, 448.0)
    return x_clipped.to(tl.float8e4nv)


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
    page_stride: tl.int64,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    # FP8 flags
    FP8_KV_CACHE: tl.constexpr,
    KV_CACHE_FP8E4M3_DTYPE: tl.constexpr,
    KV_CACHE_FP8E5M2_DTYPE: tl.constexpr,
    KV_CACHE_UINT8_CAST_REQUIRED: tl.constexpr,
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
    tile_pos = tile_i * TILE_SIZE + tile_offs

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    src_key_idx = token_idx * key_stride
    src_value_idx = token_idx * value_stride

    tgt_idx = block_idx * block_stride + block_offset * page_stride

    # [TILE_SIZE]
    key_load = tl.load(
        key_ptr + src_key_idx + tile_pos,
        mask=tile_pos < (num_heads * head_size)
    ) 
    if FP8_KV_CACHE:
        if key_load.dtype.is_fp8():
            key_tile = key_load
        elif KV_CACHE_FP8E4M3_DTYPE:
            key_tile = quant_symmetric_per_tensor_fp8e4nv(
                    key_load.to(tl.float32), tl.load(k_scale)
                )
        elif KV_CACHE_FP8E5M2_DTYPE:
            key_tile = quant_symmetric_per_tensor_fp8e5(
                    key_load.to(tl.float32), tl.load(k_scale)
                )
        if KV_CACHE_UINT8_CAST_REQUIRED:
            # here, the dtype of the pytorch tensor kv_cache is uint8, so 
            #  we need to cast it to uint8 with bitcast=True to avoid the
            #  implicit cast in tl.store
            key_tile = key_tile.to(tl.uint8, bitcast=True)
    else:
        key_tile = key_load

    # [TILE_SIZE]
    value_load = tl.load(
        value_ptr + src_value_idx + tile_pos,
        mask=tile_pos < (num_heads * head_size)
    )
    if FP8_KV_CACHE:
        if value_load.dtype.is_fp8():
            value_tile = value_load
        elif KV_CACHE_FP8E4M3_DTYPE:
            value_tile = quant_symmetric_per_tensor_fp8e4nv(
                    value_load.to(tl.float32), tl.load(v_scale)
                )
        elif KV_CACHE_FP8E5M2_DTYPE:
            value_tile = quant_symmetric_per_tensor_fp8e5(
                    value_load.to(tl.float32), tl.load(v_scale)
                )
        if KV_CACHE_UINT8_CAST_REQUIRED:
            # here, the dtype of the pytorch tensor kv_cache is uint8, so 
            #  we need to cast it to uint8 with bitcast=True to avoid the
            #  implicit cast in tl.store
            value_tile = value_tile.to(tl.uint8, bitcast=True)
    else:
        value_tile = value_load

    tl.store(
        key_cache_ptr + tgt_idx + tile_pos,
        key_tile,
        mask=tile_pos < (num_heads * head_size),
    )
    tl.store(
        value_cache_ptr + tgt_idx + tile_pos,
        value_tile,
        mask=tile_pos < (num_heads * head_size),
    )
    return


def reshape_and_cache_flash(
    key: torch.Tensor,  # [num_tokens, num_heads, head_size]
    value: torch.Tensor,  # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, block_size, num_heads, head_size]
    value_cache: torch.Tensor,  # [num_blocks, block_size, num_heads, head_size]
    slot_mapping: torch.Tensor,  # [num_tokens]
    kv_cache_dtype: str,  # "auto", "fp8"
    k_scale: torch.Tensor,  # float32
    v_scale: torch.Tensor,  # float32
):
    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_size = key.shape[2]
    block_size = key_cache.shape[1]
    n = num_heads * head_size

    key_stride = key.stride()[0]
    value_stride = key.stride()[0]
    block_stride = key_cache.stride()[0]
    page_stride = key_cache.stride()[1]

    head_stride = key_cache.stride()[2]
    assert head_stride == head_size, "only continous heads are supported"

    kv_cache_torch_dtype = key_cache.dtype if kv_cache_dtype == "auto" else STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
    FP8_KV_CACHE = kv_cache_torch_dtype in [torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8]
    # if the pytorch dtype of the kv_cache is uint8, fp8e4m3 is used 
    #  (see `DISPATCH_BY_KV_CACHE_DTYPE` in csrc/quantization/fp8/nvidia/quant_utils.cuh)
    KV_CACHE_FP8E4M3_DTYPE = kv_cache_torch_dtype == torch.float8_e4m3fn or kv_cache_torch_dtype == torch.uint8
    KV_CACHE_FP8E5M2_DTYPE = kv_cache_torch_dtype == torch.float8_e5m2
    KV_CACHE_UINT8_CAST_REQUIRED = kv_cache_torch_dtype == torch.uint8
    assert (not FP8_KV_CACHE) or (KV_CACHE_FP8E4M3_DTYPE or KV_CACHE_FP8E5M2_DTYPE), \
        f"unsupported kv cache dtype, got {kv_cache_torch_dtype}"

    # heuristics instead of autotuning
    TILE_SIZE = min(2048, triton.next_power_of_2(n))
    if torch.version.hip:
        num_stages = 4
        num_warps = 8
    else: # cuda
        num_stages = 10
        num_warps = 16
        if torch.cuda.get_device_capability(key.device)[0] < 9:
            TILE_SIZE = min(512, TILE_SIZE)
    
    # TODO(ngl): maybe replace with static launch grid to avoid overhead if
    #   using cudagraphs
    grid = lambda meta: (int(num_tokens), triton.cdiv(n, meta["TILE_SIZE"]))

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
        page_stride=page_stride,
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        # FP8 flags
        FP8_KV_CACHE=FP8_KV_CACHE,
        KV_CACHE_FP8E4M3_DTYPE=KV_CACHE_FP8E4M3_DTYPE,
        KV_CACHE_FP8E5M2_DTYPE=KV_CACHE_FP8E5M2_DTYPE,
        KV_CACHE_UINT8_CAST_REQUIRED=KV_CACHE_UINT8_CAST_REQUIRED,
        # autotune parameters
        TILE_SIZE=TILE_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

