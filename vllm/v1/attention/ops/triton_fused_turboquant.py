# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels for TurboQuant encode and decode.

Encode kernel fuses: normalize → sign_flip → Hadamard → quantize →
4-bit pack → scatter packed bytes to paged KV cache.
Norms and outlier bytes are handled in Python (simple, small).

Decode kernel fuses: load packed bytes → 4-bit unpack → codebook →
inverse Hadamard → sign_flip → scale by norm → write output.
Norms and outlier channels extracted in Python before kernel call.

Both eliminate the expensive intermediate indices tensor and Python
bit-packing overhead.
"""

import math

import torch

from vllm.triton_utils import tl, triton

# ===========================================================================
# Fused encode: normalize → Hadamard → quantize → 4-bit pack → scatter
# ===========================================================================


@triton.jit
def _fused_encode_and_store_kernel(
    # Input (normal channels): [num_tokens, num_kv_heads, normal_size]
    x_ptr,
    # Sign flips: [BLOCK_D] float32
    signs_ptr,
    # Boundaries: [num_centroids - 1] float32
    boundaries_ptr,
    # Scratch: [num_tokens * num_kv_heads, BLOCK_D] float32
    scratch_ptr,
    # Cache slice: [num_blocks, block_size, num_kv_heads, slot_bytes] uint8
    cache_ptr,
    # Block indices/offsets: [num_tokens]
    block_indices_ptr,
    block_offsets_ptr,
    # Output norms: [num_tokens, num_kv_heads] float16
    norms_ptr,
    # Shapes
    normal_size: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_boundaries: tl.constexpr,
    LOG2_D: tl.constexpr,
    packed_start: tl.constexpr,  # byte offset where packed data starts
    packed_bytes: tl.constexpr,
    # Input strides
    x_stride_token: tl.int64,
    x_stride_head: tl.int64,
    # Cache strides
    cache_stride_block: tl.int64,
    cache_stride_bs: tl.int64,
    cache_stride_head: tl.int64,
    # Norm strides
    norm_stride_token: tl.int64,
    BLOCK_D: tl.constexpr,
    BLOCK_PACKED: tl.constexpr,
):
    """Fused encode + 4-bit pack + scatter packed bytes to cache.

    Norms are output separately (no Triton bitcast needed).
    Outlier bytes are written to cache by the Python wrapper.
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    scratch_row = token_idx * num_kv_heads + head_idx

    dim_offs = tl.arange(0, BLOCK_D)
    mask = dim_offs < normal_size

    # ---- Normalize ----
    x_base = token_idx * x_stride_token + head_idx * x_stride_head
    x = tl.load(x_ptr + x_base + dim_offs, mask=mask, other=0.0).to(tl.float32)
    norm_sq = tl.sum(x * x, axis=0)
    norm_val = tl.sqrt(norm_sq + 1e-16)
    x = x / norm_val

    # ---- Sign flips ----
    signs = tl.load(signs_ptr + dim_offs)
    x = x * signs

    # ---- Hadamard butterfly ----
    scratch_base = scratch_row * BLOCK_D
    tl.store(scratch_ptr + scratch_base + dim_offs, x)

    h = 1
    for _level in range(LOG2_D):
        partner = dim_offs ^ h
        val_self = tl.load(scratch_ptr + scratch_base + dim_offs)
        val_partner = tl.load(scratch_ptr + scratch_base + partner)
        is_lower = (dim_offs & h) == 0
        result = tl.where(is_lower, val_self + val_partner, val_partner - val_self)
        tl.store(scratch_ptr + scratch_base + dim_offs, result)
        h = h * 2

    tl.debug_barrier()
    x = tl.load(scratch_ptr + scratch_base + dim_offs)
    had_scale = 1.0 / tl.sqrt(float(BLOCK_D))
    x = x * had_scale

    # ---- Quantize ----
    idx = tl.zeros([BLOCK_D], dtype=tl.int32)
    for b in range(num_boundaries):
        bnd = tl.load(boundaries_ptr + b)
        idx = idx + (x > bnd).to(tl.int32)

    # ---- 4-bit pack via scratch ----
    tl.store(scratch_ptr + scratch_base + dim_offs, idx.to(tl.float32))
    tl.debug_barrier()

    pair_offs = tl.arange(0, BLOCK_PACKED)
    pair_mask = pair_offs < packed_bytes
    even_pos = pair_offs * 2
    odd_pos = pair_offs * 2 + 1

    even_val = tl.load(
        scratch_ptr + scratch_base + even_pos,
        mask=(even_pos < normal_size) & pair_mask,
        other=0,
    ).to(tl.int32)
    odd_val = tl.load(
        scratch_ptr + scratch_base + odd_pos,
        mask=(odd_pos < normal_size) & pair_mask,
        other=0,
    ).to(tl.int32)

    packed_byte = ((even_val & 0xF) | ((odd_val & 0xF) << 4)).to(tl.uint8)

    # ---- Scatter packed bytes to cache ----
    block_idx = tl.load(block_indices_ptr + token_idx)
    block_off = tl.load(block_offsets_ptr + token_idx)
    cache_base = (
        block_idx * cache_stride_block
        + block_off * cache_stride_bs
        + head_idx * cache_stride_head
    )

    tl.store(
        cache_ptr + cache_base + packed_start + pair_offs,
        packed_byte,
        mask=pair_mask,
    )

    # ---- Output norm separately (no bitcast) ----
    tl.store(
        norms_ptr + token_idx * norm_stride_token + head_idx,
        norm_val.to(tl.float16),
    )


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def fused_hadamard_encode_and_store(
    normal_x: torch.Tensor,  # [num_tokens, num_kv_heads, normal_size]
    outlier_x: torch.Tensor | None,  # [num_tokens, num_kv_heads, n_outliers]
    sign_flips: torch.Tensor,  # [BLOCK_D] float32
    boundaries: torch.Tensor,  # [num_centroids - 1]
    cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, slot_bytes]
    block_indices: torch.Tensor,  # [num_tokens]
    block_offsets: torch.Tensor,  # [num_tokens]
    bit_width: int = 4,
) -> None:
    """Fused encode + 4-bit pack + scatter to paged KV cache.

    The kernel handles quantization + packing + scatter of packed bytes.
    Outlier bytes and norm bytes are written to cache by Python (simple ops).
    """
    assert bit_width == 4, "Fused kernel only supports 4-bit packing"

    num_tokens, num_kv_heads, normal_size = normal_x.shape
    BLOCK_D = sign_flips.shape[0]
    LOG2_D = int(math.log2(BLOCK_D))
    num_boundaries = boundaries.shape[0]

    n_outliers = outlier_x.shape[2] if outlier_x is not None else 0
    outlier_u8_count = n_outliers * 2
    packed_bytes = math.ceil(normal_size * bit_width / 8)

    scratch = torch.empty(
        (num_tokens * num_kv_heads, BLOCK_D),
        dtype=torch.float32,
        device=normal_x.device,
    )
    norms = torch.empty(
        (num_tokens, num_kv_heads),
        dtype=torch.float16,
        device=normal_x.device,
    )

    BLOCK_PACKED = _next_power_of_2(packed_bytes)
    grid = (num_tokens, num_kv_heads)

    # Kernel writes packed indices to cache and norms to separate tensor
    _fused_encode_and_store_kernel[grid](
        x_ptr=normal_x,
        signs_ptr=sign_flips,
        boundaries_ptr=boundaries,
        scratch_ptr=scratch,
        cache_ptr=cache,
        block_indices_ptr=block_indices,
        block_offsets_ptr=block_offsets,
        norms_ptr=norms,
        normal_size=normal_size,
        num_kv_heads=num_kv_heads,
        num_boundaries=num_boundaries,
        LOG2_D=LOG2_D,
        packed_start=outlier_u8_count,
        packed_bytes=packed_bytes,
        x_stride_token=normal_x.stride(0),
        x_stride_head=normal_x.stride(1),
        cache_stride_block=cache.stride(0),
        cache_stride_bs=cache.stride(1),
        cache_stride_head=cache.stride(2),
        norm_stride_token=norms.stride(0),
        BLOCK_D=BLOCK_D,
        BLOCK_PACKED=BLOCK_PACKED,
        num_warps=4,
        num_stages=1,
    )

    # Python: write outlier bf16 bytes and norm bytes to cache
    N = num_tokens * num_kv_heads
    norm_offset = outlier_u8_count + packed_bytes

    # Norm bytes → cache
    norm_u8 = norms.reshape(N).view(torch.uint8).reshape(N, 2)
    norm_3d = norm_u8.reshape(num_tokens, num_kv_heads, 2)
    cache[block_indices, block_offsets, :, norm_offset : norm_offset + 2] = norm_3d

    # Outlier bytes → cache
    if outlier_x is not None and n_outliers > 0:
        outlier_u8 = (
            outlier_x.to(torch.bfloat16)
            .contiguous()
            .view(torch.uint8)
            .reshape(num_tokens, num_kv_heads, outlier_u8_count)
        )
        cache[block_indices, block_offsets, :, :outlier_u8_count] = outlier_u8


# ===========================================================================
# Fused decode: unpack → codebook → inverse Hadamard → scale → write
# ===========================================================================


@triton.jit
def _fused_decode_kernel(
    # Packed index bytes: [N, packed_bytes] uint8
    packed_ptr,
    # Sign flips: [BLOCK_D] float32
    signs_ptr,
    # Codebook: [num_centroids] float32
    codebook_ptr,
    # Scratch: [N, BLOCK_D] float32
    scratch_ptr,
    # Norms: [N] float16
    norms_ptr,
    # Output: [N, normal_size] bfloat16
    out_ptr,
    # Shapes
    normal_size: tl.constexpr,
    packed_bytes: tl.constexpr,
    LOG2_D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_PACKED: tl.constexpr,
):
    """Fused decode: unpack 4-bit → codebook → inv Hadamard → scale → write.

    Norms are passed as a separate float16 tensor (no bitcast needed).
    Outlier channels are handled by the Python caller.
    """
    row_idx = tl.program_id(0)

    dim_offs = tl.arange(0, BLOCK_D)
    mask = dim_offs < normal_size

    # ---- Unpack 4-bit indices ----
    pair_offs = tl.arange(0, BLOCK_PACKED)
    pair_mask = pair_offs < packed_bytes
    packed_data = tl.load(
        packed_ptr + row_idx * packed_bytes + pair_offs,
        mask=pair_mask,
        other=0,
    ).to(tl.int32)

    low_idx = packed_data & 0xF
    high_idx = (packed_data >> 4) & 0xF

    # Store unpacked values to scratch at interleaved positions
    scratch_base = row_idx * BLOCK_D
    even_pos = pair_offs * 2
    odd_pos = pair_offs * 2 + 1
    tl.store(
        scratch_ptr + scratch_base + even_pos,
        low_idx.to(tl.float32),
        mask=pair_mask & (even_pos < normal_size),
    )
    tl.store(
        scratch_ptr + scratch_base + odd_pos,
        high_idx.to(tl.float32),
        mask=pair_mask & (odd_pos < normal_size),
    )
    tl.debug_barrier()

    # Reload as contiguous indices
    indices = tl.load(
        scratch_ptr + scratch_base + dim_offs,
        mask=mask,
        other=0,
    ).to(tl.int32)

    # ---- Codebook lookup ----
    reconstructed = tl.load(codebook_ptr + indices)
    reconstructed = tl.where(mask, reconstructed, 0.0)

    # ---- Inverse Hadamard butterfly ----
    tl.store(scratch_ptr + scratch_base + dim_offs, reconstructed)

    h = 1
    for _level in range(LOG2_D):
        partner = dim_offs ^ h
        val_self = tl.load(scratch_ptr + scratch_base + dim_offs)
        val_partner = tl.load(scratch_ptr + scratch_base + partner)
        is_lower = (dim_offs & h) == 0
        result = tl.where(is_lower, val_self + val_partner, val_partner - val_self)
        tl.store(scratch_ptr + scratch_base + dim_offs, result)
        h = h * 2

    tl.debug_barrier()
    x = tl.load(scratch_ptr + scratch_base + dim_offs)
    had_scale = 1.0 / tl.sqrt(float(BLOCK_D))
    x = x * had_scale

    # ---- Sign flips ----
    signs = tl.load(signs_ptr + dim_offs)
    x = x * signs

    # ---- Scale by norm (loaded from float16 tensor, no bitcast) ----
    norm_val = tl.load(norms_ptr + row_idx).to(tl.float32)
    x = x * norm_val

    # ---- Write output ----
    out_base = row_idx * normal_size
    tl.store(
        out_ptr + out_base + dim_offs,
        x.to(tl.bfloat16),
        mask=mask,
    )


def fused_hadamard_decode_from_slots(
    flat_slots: torch.Tensor,  # [N, slot_bytes] uint8
    sign_flips: torch.Tensor,  # [BLOCK_D] float32
    codebook: torch.Tensor,  # [num_centroids] float32
    normal_idx: torch.Tensor | None,  # [normal_size] int64
    outlier_idx: torch.Tensor | None,  # [n_outliers] int64
    head_size: int,
    normal_size: int,
    n_outliers: int,
    packed_bytes: int,
) -> torch.Tensor:
    """Fused decode from flat slot data to bf16 output.

    The kernel handles: unpack → codebook → inv Hadamard → scale → write.
    Python handles: norm extraction and outlier channel interleaving.

    Returns: [N, head_size] bfloat16
    """
    N = flat_slots.shape[0]
    BLOCK_D = sign_flips.shape[0]
    LOG2_D = int(math.log2(BLOCK_D))
    outlier_u8_count = n_outliers * 2
    norm_offset = outlier_u8_count + packed_bytes

    # Extract norms from slots (Python — simple slice + view)
    norms = (
        flat_slots[:, norm_offset : norm_offset + 2]
        .contiguous()
        .view(torch.float16)
        .reshape(N)
    )

    # Extract packed index bytes
    packed_data = flat_slots[
        :, outlier_u8_count : outlier_u8_count + packed_bytes
    ].contiguous()

    # Kernel decodes normal channels
    normal_out = torch.empty(
        N, normal_size, dtype=torch.bfloat16, device=flat_slots.device
    )
    scratch = torch.empty(N, BLOCK_D, dtype=torch.float32, device=flat_slots.device)

    BLOCK_PACKED = _next_power_of_2(packed_bytes)
    grid = (N,)

    _fused_decode_kernel[grid](
        packed_ptr=packed_data,
        signs_ptr=sign_flips,
        codebook_ptr=codebook,
        scratch_ptr=scratch,
        norms_ptr=norms,
        out_ptr=normal_out,
        normal_size=normal_size,
        packed_bytes=packed_bytes,
        LOG2_D=LOG2_D,
        BLOCK_D=BLOCK_D,
        BLOCK_PACKED=BLOCK_PACKED,
        num_warps=4,
        num_stages=1,
    )

    # Assemble full output with outlier channels
    has_outliers = normal_idx is not None and n_outliers > 0
    if has_outliers:
        out = torch.empty(N, head_size, dtype=torch.bfloat16, device=flat_slots.device)
        out[:, normal_idx] = normal_out
        # Extract outlier bf16 values from slots
        outlier_vals = (
            flat_slots[:, :outlier_u8_count]
            .contiguous()
            .view(torch.bfloat16)
            .reshape(N, n_outliers)
        )
        out[:, outlier_idx] = outlier_vals
        return out
    else:
        return normal_out
