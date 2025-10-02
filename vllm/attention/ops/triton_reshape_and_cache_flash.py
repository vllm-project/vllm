# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


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
    key_load = tl.load(key_ptr + src_key_idx + tile_pos,
                       mask=tile_pos < (num_heads * head_size))
    if FP8_KV_CACHE:
        if key_load.dtype.is_fp8():
            key_tile = key_load
        else:
            # tl.store will do the correct implicit cast to fp8,
            #  based on the key_cache_ptr.dtype.element_ty
            key_tile = key_load / tl.load(k_scale)
    else:
        key_tile = key_load

    # [TILE_SIZE]
    value_load = tl.load(value_ptr + src_value_idx + tile_pos,
                         mask=tile_pos < (num_heads * head_size))
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
    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_size = key.shape[2]
    block_size = key_cache.shape[1]
    n = num_heads * head_size

    key_stride = key.stride()[0]
    value_stride = value.stride()[0]
    block_stride = key_cache.stride()[0]
    page_stride = key_cache.stride()[1]

    head_stride = key_cache.stride()[2]
    assert head_stride == head_size, "only continous heads are supported"

    assert kv_cache_dtype == "auto" or kv_cache_dtype.startswith("fp8"), \
        f"unsupported kv_cache_dtype (str), got {kv_cache_dtype}."
    kv_cache_torch_dtype = current_platform.fp8_dtype() if \
        kv_cache_dtype.startswith("fp8") else key_cache.dtype

    if key_cache.dtype != kv_cache_torch_dtype and kv_cache_dtype.startswith(
            "fp8"):
        # to avoid erounous implicit cast in triton kernel (tl.store to uint8)
        # (e.g. explicit cast to fp8e4m3fnuz is not supported in triton 3.4)
        key_cache = key_cache.view(kv_cache_torch_dtype)
        value_cache = value_cache.view(kv_cache_torch_dtype)
    assert kv_cache_dtype != torch.uint8, "explicit fp8 cast and store to "\
        "uint8 is not supported by triton reshape_and_cache_flash"

    FP8_KV_CACHE = kv_cache_dtype.startswith("fp8")
    assert (not FP8_KV_CACHE) or kv_cache_torch_dtype in [
        torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8,
        torch.float8_e4m3fnuz], \
            "unsupported dtype of KV cache tensor, got "\
            "{kv_cache_torch_dtype}. Supported kv cache dtypes: fp8e4m3fn, " \
            "fp8e5m2, uint8, bfloat16, float16, float32, fp8e4m3fnuz."

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
        # autotune parameters
        TILE_SIZE=TILE_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
