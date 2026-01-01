# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
NVFP4 Dequantization Kernel for KV Cache

NVFP4 Format:
- 4-bit signed values with block-based microscaling
- Block size: 16 elements per scale (fp16 scale)
- Packed format: data (head_size//2 bytes) + scales (head_size//16 * 2 bytes)
- packed_head_size = head_size//2 + head_size//16

NVFP4 Value Mapping (4-bit signed):
Bits 0-2: magnitude index (0-7)
Bit 3: sign bit (0=positive, 1=negative)

Magnitude lookup: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
"""

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    # NVFP4 magnitude lookup table values
    # Index 0-7 maps to these magnitudes
    NVFP4_MAGNITUDES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

    @triton.jit
    def nvfp4_dequant_cached_kv_kernel(
        packed_ptr,
        output_ptr,
        total_slots: tl.constexpr,
        num_heads: tl.constexpr,
        head_size: tl.constexpr,
        stride_p_slot: tl.int64,
        stride_p_head: tl.int64,
        stride_o_slot: tl.int64,
        stride_o_head: tl.int64,
    ):
        pid = tl.program_id(0)

        # pid maps to (slot_idx, head_idx)
        head_idx = pid % num_heads
        slot_idx = pid // num_heads

        p_base = slot_idx * stride_p_slot + head_idx * stride_p_head
        o_base = slot_idx * stride_o_slot + head_idx * stride_o_head

        DATA_SIZE: tl.constexpr = head_size // 2
        SCALE_SIZE: tl.constexpr = head_size // 16

        # Process each 16-element block within the head
        for b in range(SCALE_SIZE):
            # Load 8 packed bytes (16 elements)
            packed_bytes = tl.load(packed_ptr + p_base + b * 8 + tl.arange(0, 8))

            # Load scale (uint8 exp + 127)
            scale_u8 = tl.load(packed_ptr + p_base + DATA_SIZE + b).to(tl.float32)
            scale = tl.math.exp2(scale_u8 - 127.0)

            # Extract nibbles
            low = packed_bytes & 0x0F
            high = (packed_bytes >> 4) & 0x0F

            # Dequantize (E2M1 logic matching nvfp4_reshape_and_cache)
            # Sign bit is 8 (1000)
            # Dequantize first 8 (low nibbles)
            sign_low = tl.where((low & 8) != 0, -1.0, 1.0)
            mag_idx_low = low & 7
            mag_low = tl.where(
                mag_idx_low == 0,
                0.0,
                tl.where(
                    mag_idx_low == 1,
                    0.5,
                    tl.where(
                        mag_idx_low == 2,
                        1.0,
                        tl.where(
                            mag_idx_low == 3,
                            1.5,
                            tl.where(
                                mag_idx_low == 4,
                                2.0,
                                tl.where(
                                    mag_idx_low == 5,
                                    3.0,
                                    tl.where(mag_idx_low == 6, 4.0, 6.0),
                                ),
                            ),
                        ),
                    ),
                ),
            )
            dq_low = sign_low * mag_low * scale

            # Dequantize next 8 (high nibbles)
            sign_high = tl.where((high & 8) != 0, -1.0, 1.0)
            mag_idx_high = high & 7
            mag_high = tl.where(
                mag_idx_high == 0,
                0.0,
                tl.where(
                    mag_idx_high == 1,
                    0.5,
                    tl.where(
                        mag_idx_high == 2,
                        1.0,
                        tl.where(
                            mag_idx_high == 3,
                            1.5,
                            tl.where(
                                mag_idx_high == 4,
                                2.0,
                                tl.where(
                                    mag_idx_high == 5,
                                    3.0,
                                    tl.where(mag_idx_high == 6, 4.0, 6.0),
                                ),
                            ),
                        ),
                    ),
                ),
            )
            dq_high = sign_high * mag_high * scale

            # Interleave into output (low, high, low, high...)
            offs_low = b * 16 + tl.arange(0, 8) * 2
            offs_high = b * 16 + tl.arange(0, 8) * 2 + 1

            tl.store(
                output_ptr + o_base + offs_low, dq_low.to(output_ptr.dtype.element_ty)
            )
            tl.store(
                output_ptr + o_base + offs_high, dq_high.to(output_ptr.dtype.element_ty)
            )


def dequantize_nvfp4_kv_cache(
    packed_cache: torch.Tensor,
    head_size: int,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize NVFP4 packed KV cache to bfloat16.

    Args:
        packed_cache: Packed NVFP4 cache tensor
            Shape: [2, num_blocks, block_size, num_heads, packed_head_size] or
                   [num_blocks, block_size, num_heads, packed_head_size]
        head_size: Original head size before packing
        output_dtype: Output dtype (default bfloat16)

    Returns:
        Dequantized tensor with original head_size
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for NVFP4 dequantization")

    # Handle shape variations
    has_kv_dim = packed_cache.dim() == 5
    if has_kv_dim:
        # [2, num_blocks, block_size, num_heads, packed_head_size]
        kv_dim, num_blocks, block_size, num_heads, packed_head_size = packed_cache.shape
        packed_cache_flat = packed_cache.reshape(-1, num_heads, packed_head_size)
        total_slots = kv_dim * num_blocks * block_size
    else:
        # [num_blocks, block_size, num_heads, packed_head_size]
        num_blocks, block_size, num_heads, packed_head_size = packed_cache.shape
        packed_cache_flat = packed_cache.reshape(-1, num_heads, packed_head_size)
        total_slots = num_blocks * block_size
        kv_dim = 1

    # Allocate output
    output = torch.empty(
        total_slots,
        num_heads,
        head_size,
        dtype=output_dtype,
        device=packed_cache.device,
    )

    # Launch kernel
    grid = (total_slots * num_heads,)

    nvfp4_dequant_cached_kv_kernel[grid](
        packed_cache_flat,
        output,
        total_slots=total_slots,
        num_heads=num_heads,
        head_size=head_size,
        stride_p_slot=packed_cache_flat.stride(0),
        stride_p_head=packed_cache_flat.stride(1),
        stride_o_slot=output.stride(0),
        stride_o_head=output.stride(1),
    )

    # Reshape output
    if has_kv_dim:
        output = output.view(kv_dim, num_blocks, block_size, num_heads, head_size)
    else:
        output = output.view(num_blocks, block_size, num_heads, head_size)

    return output


def dequantize_nvfp4_kv_cache_simple(
    packed_cache: torch.Tensor,
    head_size: int,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Simple PyTorch implementation of NVFP4 dequantization for debugging.

    NVFP4 Value Mapping (4-bit signed):
    Bits 0-2: magnitude index (0-7) -> [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    Bit 3: sign bit (0=positive, 1=negative)

    Scales: uint8 exponential format (1 byte per 16 elements)
    - Stored as exp + 127
    """
    # Packed layout: data (head_size//2) + scales (head_size//16 bytes as uint8 exp)
    data_size = head_size // 2
    scale_size = head_size // 16
    packed_head_size = data_size + scale_size

    # Get shape info
    orig_shape = packed_cache.shape
    device = packed_cache.device

    # Flatten for processing
    *leading_dims, phs = packed_cache.shape
    assert (
        phs == packed_head_size
    ), f"Expected packed_head_size={packed_head_size}, got {phs}"

    total_positions = 1
    for d in leading_dims:
        total_positions *= d
    flat = packed_cache.view(total_positions, packed_head_size)

    # Split data and scales
    data_bytes = flat[:, :data_size]  # [N, data_size]
    scale_bytes = flat[:, data_size:]  # [N, scale_size]

    # Convert uint8 exponential scales to float32
    # scale = 2^(val - 127)
    scale_exp = scale_bytes.to(torch.float32) - 127.0
    scales = torch.pow(2.0, scale_exp)  # [N, scale_size]

    # FAIL-FAST: Check for suspicious scales (ignore zero blocks)
    if scales.max() > 1e10 or (scales.min() < 1e-40 and scales.min() > 0):
        print(
            f"SUSPICIOUS NVFP4 scales detected! Max: {scales.max()}, Min: {scales.min()}"
        )
        # We don't raise here to allow logging, but we will check results later

    # Expand scales to match elements (each scale covers 16 elements)
    scales_expanded = (
        scales.unsqueeze(-1).expand(-1, -1, 16).reshape(total_positions, head_size)
    )  # [N, head_size]

    # Unpack 4-bit values from bytes
    low_nibbles = data_bytes & 0x0F  # [N, data_size] -> even indices
    high_nibbles = (data_bytes >> 4) & 0x0F  # [N, data_size] -> odd indices

    # Interleave to get [N, head_size] tensor of 4-bit indices
    nibbles = torch.stack([low_nibbles, high_nibbles], dim=-1).view(
        total_positions, head_size
    )

    # Extract sign and magnitude
    signs = torch.where((nibbles & 0x08) != 0, -1.0, 1.0).to(device)
    mag_indices = (nibbles & 0x07).to(torch.int64)

    # NVFP4 magnitude lookup table
    mag_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device, dtype=torch.float32
    )
    magnitudes = mag_lut[mag_indices]

    # Dequantize: sign * magnitude * scale
    dequantized = signs * magnitudes * scales_expanded

    # Reshape back and convert to output dtype
    output_shape = list(leading_dims) + [head_size]
    return dequantized.view(*output_shape).to(output_dtype)
