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


@triton.jit
def quantize_nvfp4(val, scale):
    # NVFP4 magnitude set: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    # Rounded midpoints: [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    norm_val = tl.abs(val / scale)
    sign = tl.where(val < 0, 8, 0)  # 4th bit is sign

    idx = tl.where(
        norm_val < 0.25,
        0,
        tl.where(
            norm_val < 0.75,
            1,
            tl.where(
                norm_val < 1.25,
                2,
                tl.where(
                    norm_val < 1.75,
                    3,
                    tl.where(
                        norm_val < 2.5,
                        4,
                        tl.where(
                            norm_val < 3.5,
                            5,
                            tl.where(norm_val < 5.0, 6, 7),
                        ),
                    ),
                ),
            ),
        ),
    )

    return (sign | idx).to(tl.uint8)


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
    key_stride_tok: tl.int64,
    key_stride_head: tl.int64,
    value_stride_tok: tl.int64,
    value_stride_head: tl.int64,
    block_stride: tl.int64,
    page_stride: tl.int64,
    head_stride: tl.int64,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    # FP8 and NVFP4 flags
    FP8_KV_CACHE: tl.constexpr,
    NVFP4_KV_CACHE: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)
    head_idx = tl.program_id(axis=1)

    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    src_key_ptr = key_ptr + token_idx * key_stride_tok + head_idx * key_stride_head
    src_val_ptr = (
        value_ptr + token_idx * value_stride_tok + head_idx * value_stride_head
    )

    tgt_key_base = (
        key_cache_ptr
        + block_idx * block_stride
        + block_offset * page_stride
        + head_idx * head_stride
    )
    tgt_val_base = (
        value_cache_ptr
        + block_idx * block_stride
        + block_offset * page_stride
        + head_idx * head_stride
    )

    if NVFP4_KV_CACHE:
        # Block-based microscaling (16 elements per block)
        DATA_SIZE: tl.constexpr = head_size // 2
        SCALE_SIZE: tl.constexpr = head_size // 16

        # Process in blocks of 16
        for b in range(SCALE_SIZE):
            offs = b * 16 + tl.arange(0, 16)
            k_vals = tl.load(src_key_ptr + offs)
            v_vals = tl.load(src_val_ptr + offs)

            # Compute exponential scales (SGLang style: MXFP4 compatible)
            # scale = 2^ceil(log2(max_val / 6.0))
            k_exp = tl.math.ceil(tl.math.log2(tl.max(tl.abs(k_vals)) / 6.0 + 1e-10))
            v_exp = tl.math.ceil(tl.math.log2(tl.max(tl.abs(v_vals)) / 6.0 + 1e-10))

            k_s_val = tl.math.exp2(k_exp)
            v_s_val = tl.math.exp2(v_exp)

            # Store scales as 1-byte uint8 (offset by 127) at the end of the head
            # Layout: head_size//2 (data) + head_size//16 (scales)
            tl.store(tgt_key_base + DATA_SIZE + b, (k_exp + 127).to(tl.uint8))
            tl.store(tgt_val_base + DATA_SIZE + b, (v_exp + 127).to(tl.uint8))

            # Quantize and pack using separated even/odd loading to avoid Triton indexing errors
            # Even indices: 0, 2, 4, 6, 8, 10, 12, 14
            even_offs = b * 16 + tl.arange(0, 8) * 2
            k_even_vals = tl.load(src_key_ptr + even_offs)
            v_even_vals = tl.load(src_val_ptr + even_offs)

            # Odd indices: 1, 3, 5, 7, 9, 11, 13, 15
            odd_offs = even_offs + 1
            k_odd_vals = tl.load(src_key_ptr + odd_offs)
            v_odd_vals = tl.load(src_val_ptr + odd_offs)

            # Quantize even and odd values with the shared scale
            k_q_even = quantize_nvfp4(k_even_vals, k_s_val)
            k_q_odd = quantize_nvfp4(k_odd_vals, k_s_val)
            v_q_even = quantize_nvfp4(v_even_vals, v_s_val)
            v_q_odd = quantize_nvfp4(v_odd_vals, v_s_val)

            # Pack: low nibble from even, high nibble from odd (Matches SGLang)
            k_packed = (k_q_even & 0x0F) | ((k_q_odd & 0x0F) << 4)
            v_packed = (v_q_even & 0x0F) | ((v_q_odd & 0x0F) << 4)

            # Store packed bytes (8 bytes per block of 16 values)
            pack_offs = tl.arange(0, 8)
            tl.store(tgt_key_base + b * 8 + pack_offs, k_packed.to(tl.uint8))
            tl.store(tgt_val_base + b * 8 + pack_offs, v_packed.to(tl.uint8))

    elif FP8_KV_CACHE:
        tile_pos = tl.arange(0, head_size)
        # Simple cast for FP8 path in this version for stability
        tl.store(
            tgt_key_base + tile_pos,
            tl.load(src_key_ptr + tile_pos).to(tl.float8e4nv),
        )
        tl.store(
            tgt_val_base + tile_pos,
            tl.load(src_val_ptr + tile_pos).to(tl.float8e4nv),
        )
    else:
        tile_pos = tl.arange(0, head_size)
        tl.store(tgt_key_base + tile_pos, tl.load(src_key_ptr + tile_pos))
        tl.store(tgt_val_base + tile_pos, tl.load(src_val_ptr + tile_pos))


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

    block_size = key_cache.shape[1]

    # Compute strides
    key_stride_tok = key.stride(0)
    key_stride_head = key.stride(1)
    value_stride_tok = value.stride(0)
    value_stride_head = value.stride(1)

    block_stride = key_cache.stride(0)
    page_stride = key_cache.stride(1)
    head_stride = key_cache.stride(2)

    # Grid: (num_tokens, num_heads)
    grid = (num_tokens, num_heads)

    # Determine cache type flags
    fp8_kv_cache = kv_cache_dtype.startswith("fp8")
    nvfp4_kv_cache = kv_cache_dtype == "nvfp4"

    if nvfp4_kv_cache:
        try:
            import vllm_nvfp4

            print("DEBUG: Using vllm_nvfp4.reshape_and_cache CUDA kernel")
            vllm_nvfp4.reshape_and_cache(
                key, value, key_cache, value_cache, slot_mapping
            )
            return
        except ImportError:
            print("DEBUG: vllm_nvfp4 NOT found, falling back to Triton")
            pass
        except Exception as e:
            print(f"DEBUG: vllm_nvfp4 kernel failed: {e}, falling back to Triton")
            pass

    reshape_and_cache_kernel_flash[grid](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        k_scale,
        v_scale,
        key_stride_tok,
        key_stride_head,
        value_stride_tok,
        value_stride_head,
        block_stride,
        page_stride,
        head_stride,
        num_heads,
        head_size,
        block_size,
        FP8_KV_CACHE=fp8_kv_cache,
        NVFP4_KV_CACHE=nvfp4_kv_cache,
    )
