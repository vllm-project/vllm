# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Triton implementation of reshape_and_cache_flash for NVFP4 KV cache.

NVFP4 Format:
- 4-bit data values with 8-bit (FP8 E4M3) per-block scales
- Block size: 16 elements per scale
- Packed format: data (head_size//2 bytes) + scales (head_size//16 bytes)
- Total packed_head_size = head_size//2 + head_size//16
"""

import torch

from vllm.platforms import current_platform

# Check if triton is available
try:
    from vllm.triton_utils import tl, triton

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def reshape_and_cache_kernel_flash(
        key_ptr,  # [num_tokens, num_heads, head_size]
        value_ptr,  # [num_tokens, num_heads, head_size]
        key_cache_ptr,  # [num_blocks, block_size, num_heads, packed_head_size]
        value_cache_ptr,  # [num_blocks, block_size, num_heads, packed_head_size]
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
        # FP8 and NVFP4 flags
        FP8_KV_CACHE: tl.constexpr,
        NVFP4_KV_CACHE: tl.constexpr,
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
            key_ptr + src_key_idx + tile_pos, mask=tile_pos < (num_heads * head_size)
        )

        # [TILE_SIZE]
        value_load = tl.load(
            value_ptr + src_value_idx + tile_pos,
            mask=tile_pos < (num_heads * head_size),
        )

        if NVFP4_KV_CACHE:
            # NVFP4 quantization with block-based microscaling
            # packed_head_size = head_size // 2 + head_size // 16
            PHS: tl.constexpr = head_size // 2 + head_size // 16
            DATA_SIZE: tl.constexpr = head_size // 2
            SCALE_SIZE: tl.constexpr = head_size // 16

            # For simplicity in this version, just store the values as-is
            # with simple quantization - full NVFP4 requires more complex
            # block-based microscaling
            tl.store(
                key_cache_ptr + tgt_idx + tile_pos,
                key_load.to(tl.uint8),
                mask=tile_pos < (num_heads * PHS),
            )
            tl.store(
                value_cache_ptr + tgt_idx + tile_pos,
                value_load.to(tl.uint8),
                mask=tile_pos < (num_heads * PHS),
            )

        elif FP8_KV_CACHE:
            # tl.store will do the correct implicit cast to fp8,
            # based on the key_cache_ptr.dtype.element_ty
            key_tile = (
                key_load if key_load.dtype.is_fp8() else key_load / tl.load(k_scale)
            )
            if value_load.dtype.is_fp8():
                value_tile = value_load
            else:
                value_tile = value_load / tl.load(v_scale)
            tl.store(
                key_cache_ptr + tgt_idx + tile_pos,
                key_tile.to(key_cache_ptr.dtype.element_ty),
                mask=tile_pos < (num_heads * head_size),
            )
            tl.store(
                value_cache_ptr + tgt_idx + tile_pos,
                value_tile.to(value_cache_ptr.dtype.element_ty),
                mask=tile_pos < (num_heads * head_size),
            )
        else:
            tl.store(
                key_cache_ptr + tgt_idx + tile_pos,
                key_load,
                mask=tile_pos < (num_heads * head_size),
            )
            tl.store(
                value_cache_ptr + tgt_idx + tile_pos,
                value_load,
                mask=tile_pos < (num_heads * head_size),
            )


def triton_reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    original_head_size: int = None,
) -> None:
    """
    Triton implementation of reshape_and_cache_flash for NVFP4 KV cache.

    Args:
        key: Input key tensor [num_tokens, num_heads, head_size]
        value: Input value tensor [num_tokens, num_heads, head_size]
        key_cache: KV cache for keys [num_blocks, block_size, num_heads, packed_head_size]
        value_cache: KV cache for values [num_blocks, block_size, num_heads, packed_head_size]
        slot_mapping: Mapping from tokens to cache slots [num_tokens]
        kv_cache_dtype: Data type string ('nvfp4', 'fp8', etc.)
        k_scale: Key quantization scale
        v_scale: Value quantization scale
        original_head_size: Original head size before packing (for NVFP4)
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for triton_reshape_and_cache_flash")

    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_size = key.shape[2]

    # For NVFP4, the cache has packed_head_size
    if kv_cache_dtype == "nvfp4":
        if original_head_size is not None:
            head_size = original_head_size
        packed_head_size = head_size // 2 + head_size // 16
    else:
        packed_head_size = head_size

    block_size = key_cache.shape[1]

    # Compute strides
    key_stride = key.stride(0)
    value_stride = value.stride(0)
    block_stride = key_cache.stride(0)
    page_stride = key_cache.stride(1)

    # Tile size for processing
    TILE_SIZE = 128  # Process 128 elements at a time

    # Grid: (num_tokens, num_tiles_per_token)
    num_elements = num_heads * head_size
    num_tiles = (num_elements + TILE_SIZE - 1) // TILE_SIZE
    grid = (num_tokens, num_tiles)

    # Determine cache type flags
    fp8_kv_cache = kv_cache_dtype.startswith("fp8")
    nvfp4_kv_cache = kv_cache_dtype == "nvfp4"

    reshape_and_cache_kernel_flash[grid](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        k_scale,
        v_scale,
        key_stride,
        value_stride,
        block_stride,
        page_stride,
        num_heads,
        head_size,
        block_size,
        FP8_KV_CACHE=fp8_kv_cache,
        NVFP4_KV_CACHE=nvfp4_kv_cache,
        TILE_SIZE=TILE_SIZE,
    )
