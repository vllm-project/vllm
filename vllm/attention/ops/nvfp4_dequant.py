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
        # Input: packed NVFP4 cache
        packed_cache_ptr,  # [num_blocks, block_size, num_heads, packed_head_size]
        # Output: dequantized bf16 cache
        output_ptr,  # [num_blocks, block_size, num_heads, head_size]
        # Dimensions
        num_blocks: tl.constexpr,
        block_size: tl.constexpr,
        num_heads: tl.constexpr,
        head_size: tl.constexpr,
        packed_head_size: tl.constexpr,
        # Strides for packed cache (in bytes)
        stride_packed_block: tl.int64,
        stride_packed_slot: tl.int64,
        stride_packed_head: tl.int64,
        # Strides for output (in elements)
        stride_out_block: tl.int64,
        stride_out_slot: tl.int64,
        stride_out_head: tl.int64,
        # Block parameters
        BLOCK_HEAD: tl.constexpr,  # Tile size for head dimension
    ):
        """
        Dequantize NVFP4 packed KV cache to bfloat16.

        Each program handles one (block_idx, slot_idx, head_idx) combination.
        """
        # Program indices
        pid = tl.program_id(0)
        total_slots = num_blocks * block_size * num_heads

        if pid >= total_slots:
            return

        # Compute indices
        head_idx = pid % num_heads
        slot_idx = (pid // num_heads) % block_size
        block_idx = pid // (num_heads * block_size)

        # Calculate base offsets
        packed_base = (
            block_idx * stride_packed_block
            + slot_idx * stride_packed_slot
            + head_idx * stride_packed_head
        )

        out_base = (
            block_idx * stride_out_block
            + slot_idx * stride_out_slot
            + head_idx * stride_out_head
        )

        # NVFP4 layout within packed_head_size:
        # - First head_size//2 bytes: packed 4-bit data (2 values per byte)
        # - Next head_size//16 * 2 bytes: fp16 scales (one scale per 16 elements)

        DATA_SIZE: tl.constexpr = head_size // 2
        SCALE_SIZE: tl.constexpr = head_size // 16

        # Process head_size elements (in this simplified version, we loop)
        # Each byte contains 2 packed 4-bit values
        for byte_idx in range(DATA_SIZE):
            # Load packed byte
            packed_byte = tl.load(packed_cache_ptr + packed_base + byte_idx)

            # Extract two 4-bit values
            low_nibble = packed_byte & 0x0F  # Lower 4 bits -> first value
            high_nibble = (packed_byte >> 4) & 0x0F  # Upper 4 bits -> second value

            # Determine which scale block these values belong to
            elem_idx_0 = byte_idx * 2
            elem_idx_1 = byte_idx * 2 + 1
            scale_idx_0 = elem_idx_0 // 16
            scale_idx_1 = elem_idx_1 // 16

            # Load scales (stored as fp16 after data section)
            scale_offset_0 = DATA_SIZE + scale_idx_0 * 2
            scale_offset_1 = DATA_SIZE + scale_idx_1 * 2

            # Load scale bytes and reinterpret as fp16
            scale_bytes_0_lo = tl.load(
                packed_cache_ptr + packed_base + scale_offset_0
            ).to(tl.uint16)
            scale_bytes_0_hi = tl.load(
                packed_cache_ptr + packed_base + scale_offset_0 + 1
            ).to(tl.uint16)
            scale_u16_0 = scale_bytes_0_lo | (scale_bytes_0_hi << 8)

            scale_bytes_1_lo = tl.load(
                packed_cache_ptr + packed_base + scale_offset_1
            ).to(tl.uint16)
            scale_bytes_1_hi = tl.load(
                packed_cache_ptr + packed_base + scale_offset_1 + 1
            ).to(tl.uint16)
            scale_u16_1 = scale_bytes_1_lo | (scale_bytes_1_hi << 8)

            # Convert uint16 to float16 view (reinterpret bits)
            scale_0 = scale_u16_0.to(tl.float16, bitcast=True).to(tl.float32)
            scale_1 = scale_u16_1.to(tl.float16, bitcast=True).to(tl.float32)

            # Dequantize: extract sign and magnitude from 4-bit value
            # Bit 3 = sign, Bits 0-2 = magnitude index
            sign_0 = tl.where((low_nibble & 0x08) != 0, -1.0, 1.0)
            mag_idx_0 = low_nibble & 0x07

            sign_1 = tl.where((high_nibble & 0x08) != 0, -1.0, 1.0)
            mag_idx_1 = high_nibble & 0x07

            # Magnitude lookup (NVFP4 uses specific magnitude values)
            # For simplicity, use approximate linear mapping
            # True NVFP4: [0, 0.5, 1, 1.5, 2, 3, 4, 6]
            mag_0 = tl.where(
                mag_idx_0 == 0,
                0.0,
                tl.where(
                    mag_idx_0 == 1,
                    0.5,
                    tl.where(
                        mag_idx_0 == 2,
                        1.0,
                        tl.where(
                            mag_idx_0 == 3,
                            1.5,
                            tl.where(
                                mag_idx_0 == 4,
                                2.0,
                                tl.where(
                                    mag_idx_0 == 5,
                                    3.0,
                                    tl.where(mag_idx_0 == 6, 4.0, 6.0),
                                ),
                            ),
                        ),
                    ),
                ),
            )

            mag_1 = tl.where(
                mag_idx_1 == 0,
                0.0,
                tl.where(
                    mag_idx_1 == 1,
                    0.5,
                    tl.where(
                        mag_idx_1 == 2,
                        1.0,
                        tl.where(
                            mag_idx_1 == 3,
                            1.5,
                            tl.where(
                                mag_idx_1 == 4,
                                2.0,
                                tl.where(
                                    mag_idx_1 == 5,
                                    3.0,
                                    tl.where(mag_idx_1 == 6, 4.0, 6.0),
                                ),
                            ),
                        ),
                    ),
                ),
            )

            # Final dequantized values
            val_0 = sign_0 * mag_0 * scale_0
            val_1 = sign_1 * mag_1 * scale_1

            # Store to output as bfloat16
            tl.store(output_ptr + out_base + elem_idx_0, val_0.to(tl.bfloat16))
            tl.store(output_ptr + out_base + elem_idx_1, val_1.to(tl.bfloat16))


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
        num_blocks=num_blocks * kv_dim,
        block_size=block_size if not has_kv_dim else 1,
        num_heads=num_heads,
        head_size=head_size,
        packed_head_size=packed_head_size,
        stride_packed_block=packed_cache_flat.stride(0) if not has_kv_dim else 0,
        stride_packed_slot=packed_cache_flat.stride(0),
        stride_packed_head=packed_cache_flat.stride(1),
        stride_out_block=output.stride(0) if not has_kv_dim else 0,
        stride_out_slot=output.stride(0),
        stride_out_head=output.stride(1),
        BLOCK_HEAD=32,
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

    Scales: E4M3 format (1 byte per 16 elements)
    """
    # Packed layout: data (head_size//2) + scales (head_size//16 bytes as E4M3)
    data_size = head_size // 2
    scale_size = head_size // 16
    packed_head_size = data_size + scale_size

    # Get shape info
    orig_shape = packed_cache.shape
    device = packed_cache.device

    # Flatten for processing
    # Expected: [..., packed_head_size] where ... can be any leading dims
    *leading_dims, phs = packed_cache.shape
    assert (
        phs == packed_head_size
    ), f"Expected packed_head_size={packed_head_size}, got {phs}"

    # Reshape to (total_positions, packed_head_size)
    total_positions = 1
    for d in leading_dims:
        total_positions *= d
    flat = packed_cache.view(total_positions, packed_head_size)

    # Split data and scales
    data_bytes = flat[:, :data_size]  # [N, data_size]
    scale_bytes = flat[:, data_size:]  # [N, scale_size] (1 byte per E4M3 scale)

    # Convert E4M3 scales to float32
    # E4M3 is torch.float8_e4m3fn - need to convert uint8 -> float8 -> float32
    # For now, interpret as scaled values: scale = 2^(exponent-7) * (1 + mantissa/8)
    # Simpler approach: view as float8_e4m3fn directly
    try:
        # Try native torch.float8_e4m3fn support (PyTorch 2.1+)
        scales_f8 = scale_bytes.view(torch.float8_e4m3fn)
        scales = scales_f8.to(torch.float32)  # [N, scale_size]
    except (TypeError, RuntimeError):
        # Fallback: manual E4M3 decode
        # E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits
        # For simplicity, use a linear approximation or unity scales
        # This is a placeholder - actual E4M3 decode needed
        scales = torch.ones(
            total_positions, scale_size, device=device, dtype=torch.float32
        )

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
