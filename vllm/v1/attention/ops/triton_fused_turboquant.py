# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels for TurboQuant encode and decode.

Encode: normalize → sign_flip → Hadamard → quantize → 4-bit pack → scatter
Decode: load slot → 4-bit unpack → codebook → inv Hadamard → scale → write

Both operate on a single (token, head) per program, eliminating all
intermediate tensors and Python-side pack/unpack overhead.
"""

import math

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_encode_and_store_kernel(
    # Input (normal channels): [num_tokens, num_kv_heads, normal_size]
    x_ptr,
    # Outlier channels as uint8 view of bf16:
    #   [num_tokens, num_kv_heads, n_outliers * 2] uint8
    outlier_u8_ptr,
    # Sign flips: [BLOCK_D] float32
    signs_ptr,
    # Boundaries: [num_centroids - 1] float32
    boundaries_ptr,
    # Scratch: [num_tokens * num_kv_heads, BLOCK_D] float32
    scratch_ptr,
    # Cache slice for this kv_idx: [num_blocks, block_size, num_kv_heads,
    #                                slot_bytes] uint8
    cache_ptr,
    # Block indices: [num_tokens] int32/int64
    block_indices_ptr,
    # Block offsets: [num_tokens] int32/int64
    block_offsets_ptr,
    # Shapes
    normal_size: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_boundaries: tl.constexpr,
    LOG2_D: tl.constexpr,
    outlier_u8_count: tl.constexpr,  # n_outliers * 2
    slot_bytes: tl.constexpr,
    packed_bytes: tl.constexpr,
    # Input strides
    x_stride_token: tl.int64,
    x_stride_head: tl.int64,
    # Outlier strides (over uint8 view)
    outlier_stride_token: tl.int64,
    outlier_stride_head: tl.int64,
    # Cache strides
    cache_stride_block: tl.int64,
    cache_stride_bs: tl.int64,
    cache_stride_head: tl.int64,
    BLOCK_D: tl.constexpr,
    # Max outlier bytes rounded up to power of 2 for tl.arange
    BLOCK_OUTLIER: tl.constexpr,
    # Max packed_bytes rounded up to power of 2 for tl.arange
    BLOCK_PACKED: tl.constexpr,
):
    """Fused encode + 4-bit pack + scatter to paged cache.

    Each program handles one (token, head) pair.
    Slot layout: [outlier_bf16_bytes | packed_4bit_indices | norm_fp16_bytes]
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

    # ---- Cache write address ----
    block_idx = tl.load(block_indices_ptr + token_idx)
    block_off = tl.load(block_offsets_ptr + token_idx)
    cache_base = (
        block_idx * cache_stride_block
        + block_off * cache_stride_bs
        + head_idx * cache_stride_head
    )

    # ---- Write outlier bf16 bytes (already as uint8) ----
    if outlier_u8_count > 0:
        out_byte_offs = tl.arange(0, BLOCK_OUTLIER)
        out_mask = out_byte_offs < outlier_u8_count
        outlier_base = token_idx * outlier_stride_token + head_idx * outlier_stride_head
        out_bytes = tl.load(
            outlier_u8_ptr + outlier_base + out_byte_offs,
            mask=out_mask,
            other=0,
        )
        tl.store(
            cache_ptr + cache_base + out_byte_offs,
            out_bytes,
            mask=out_mask,
        )

    outlier_byte_offset = outlier_u8_count

    # ---- 4-bit pack and write ----
    # Store idx to scratch so we can reload as pairs
    tl.store(scratch_ptr + scratch_base + dim_offs, idx.to(tl.float32))
    tl.debug_barrier()

    pair_offs = tl.arange(0, BLOCK_PACKED)
    pair_mask = pair_offs < packed_bytes
    even_pos = pair_offs * 2
    odd_pos = pair_offs * 2 + 1
    even_in_range = even_pos < normal_size
    odd_in_range = odd_pos < normal_size

    even_val = tl.load(
        scratch_ptr + scratch_base + even_pos,
        mask=even_in_range & pair_mask,
        other=0,
    ).to(tl.int32)
    odd_val = tl.load(
        scratch_ptr + scratch_base + odd_pos,
        mask=odd_in_range & pair_mask,
        other=0,
    ).to(tl.int32)

    packed_byte = ((even_val & 0xF) | ((odd_val & 0xF) << 4)).to(tl.uint8)
    tl.store(
        cache_ptr + cache_base + outlier_byte_offset + pair_offs,
        packed_byte,
        mask=pair_mask,
    )

    # ---- Write norm as fp16 (2 bytes) ----
    # Use scratch memory to reinterpret float16 as 2 uint8 bytes,
    # avoiding tl.to(bitcast=True) which may not be supported.
    norm_offset = outlier_byte_offset + packed_bytes
    scratch_f16_ptr = (scratch_ptr + scratch_base).to(tl.pointer_type(tl.float16))
    tl.store(scratch_f16_ptr, norm_val.to(tl.float16))
    scratch_u8_ptr = (scratch_ptr + scratch_base).to(tl.pointer_type(tl.uint8))
    norm_byte0 = tl.load(scratch_u8_ptr)
    norm_byte1 = tl.load(scratch_u8_ptr + 1)
    tl.store(cache_ptr + cache_base + norm_offset, norm_byte0)
    tl.store(cache_ptr + cache_base + norm_offset + 1, norm_byte1)


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


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

    Single kernel per (token, head): normalize → sign_flip → FWHT →
    quantize → 4-bit pack → scatter write.

    Only supports bit_width=4 in the fused path. Other bit widths
    fall back to the non-fused encode.
    """
    assert bit_width == 4, "Fused kernel only supports 4-bit packing"

    num_tokens, num_kv_heads, normal_size = normal_x.shape
    BLOCK_D = sign_flips.shape[0]
    LOG2_D = int(math.log2(BLOCK_D))
    num_boundaries = boundaries.shape[0]

    n_outliers = outlier_x.shape[2] if outlier_x is not None else 0
    outlier_u8_count = n_outliers * 2
    slot_bytes = cache.shape[3]
    packed_bytes = math.ceil(normal_size * bit_width / 8)

    scratch = torch.empty(
        (num_tokens * num_kv_heads, BLOCK_D),
        dtype=torch.float32,
        device=normal_x.device,
    )

    # Convert outlier bf16 to uint8 view for byte copying
    if outlier_x is not None:
        outlier_u8 = (
            outlier_x.to(torch.bfloat16)
            .contiguous()
            .view(torch.uint8)
            .reshape(num_tokens, num_kv_heads, outlier_u8_count)
        )
        outlier_stride_token = outlier_u8.stride(0)
        outlier_stride_head = outlier_u8.stride(1)
    else:
        outlier_u8 = normal_x  # dummy pointer, won't be read
        outlier_stride_token = 0
        outlier_stride_head = 0

    BLOCK_OUTLIER = max(_next_power_of_2(outlier_u8_count), 1)
    BLOCK_PACKED = _next_power_of_2(packed_bytes)

    grid = (num_tokens, num_kv_heads)

    _fused_encode_and_store_kernel[grid](
        x_ptr=normal_x,
        outlier_u8_ptr=outlier_u8,
        signs_ptr=sign_flips,
        boundaries_ptr=boundaries,
        scratch_ptr=scratch,
        cache_ptr=cache,
        block_indices_ptr=block_indices,
        block_offsets_ptr=block_offsets,
        normal_size=normal_size,
        num_kv_heads=num_kv_heads,
        num_boundaries=num_boundaries,
        LOG2_D=LOG2_D,
        outlier_u8_count=outlier_u8_count,
        slot_bytes=slot_bytes,
        packed_bytes=packed_bytes,
        x_stride_token=normal_x.stride(0),
        x_stride_head=normal_x.stride(1),
        outlier_stride_token=outlier_stride_token,
        outlier_stride_head=outlier_stride_head,
        cache_stride_block=cache.stride(0),
        cache_stride_bs=cache.stride(1),
        cache_stride_head=cache.stride(2),
        BLOCK_D=BLOCK_D,
        BLOCK_OUTLIER=BLOCK_OUTLIER,
        BLOCK_PACKED=BLOCK_PACKED,
        num_warps=4,
        num_stages=1,
    )


# ===========================================================================
# Fused decode: load slot → unpack → codebook → inv Hadamard → write bf16
# ===========================================================================


@triton.jit
def _fused_decode_kernel(
    # Input: flat cache slots [N, slot_bytes] uint8
    # N = num_entries * block_size * num_kv_heads
    slot_ptr,
    # Sign flips: [BLOCK_D] float32
    signs_ptr,
    # Codebook: [num_centroids] float32
    codebook_ptr,
    # Scratch: [N, BLOCK_D] float32
    scratch_ptr,
    # Normal channel indices: [normal_size] int64 (scatter positions)
    # If None (normal_size == head_size), write sequentially
    normal_idx_ptr,
    # Outlier channel indices: [n_outliers] int64 (scatter positions)
    outlier_idx_ptr,
    # Output: [N, head_size] bfloat16
    out_ptr,
    # Shapes
    normal_size: tl.constexpr,
    head_size: tl.constexpr,
    n_outliers: tl.constexpr,
    outlier_u8_count: tl.constexpr,  # n_outliers * 2
    packed_bytes: tl.constexpr,
    slot_bytes: tl.constexpr,
    LOG2_D: tl.constexpr,
    HAS_OUTLIERS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_PACKED: tl.constexpr,
    BLOCK_OUTLIER: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    """Fused decode one slot: unpack → codebook → inv Hadamard → write.

    Each program handles one flat element (one token-head combination).
    """
    row_idx = tl.program_id(0)
    slot_base = row_idx * slot_bytes
    out_base = row_idx * head_size

    dim_offs = tl.arange(0, BLOCK_D)
    mask = dim_offs < normal_size

    # ---- Read packed 4-bit indices from slot ----
    pair_offs = tl.arange(0, BLOCK_PACKED)
    pair_mask = pair_offs < packed_bytes
    packed_data = tl.load(
        slot_ptr + slot_base + outlier_u8_count + pair_offs,
        mask=pair_mask,
        other=0,
    ).to(tl.int32)

    # Unpack 4-bit: low nibble at even positions, high nibble at odd
    low_idx = packed_data & 0xF
    high_idx = (packed_data >> 4) & 0xF

    # Store unpacked indices to scratch for sequential access
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

    # Load unpacked indices as contiguous vector
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

    # ---- Sign flips (inverse = same signs) ----
    signs = tl.load(signs_ptr + dim_offs)
    x = x * signs

    # ---- Scale by norm ----
    # Use scratch memory to reinterpret 2 uint8 bytes as float16,
    # avoiding tl.to(bitcast=True) which may not be supported.
    norm_offset = outlier_u8_count + packed_bytes
    scratch_u8_ptr = (scratch_ptr + scratch_base).to(tl.pointer_type(tl.uint8))
    norm_byte0 = tl.load(slot_ptr + slot_base + norm_offset)
    norm_byte1 = tl.load(slot_ptr + slot_base + norm_offset + 1)
    tl.store(scratch_u8_ptr, norm_byte0)
    tl.store(scratch_u8_ptr + 1, norm_byte1)
    scratch_f16_ptr = (scratch_ptr + scratch_base).to(tl.pointer_type(tl.float16))
    norm_val = tl.load(scratch_f16_ptr).to(tl.float32)
    x = x * norm_val

    # ---- Write output ----
    if HAS_OUTLIERS:
        # Write normal channels to scattered positions
        normal_positions = tl.load(
            normal_idx_ptr + dim_offs,
            mask=mask,
            other=0,
        )
        tl.store(
            out_ptr + out_base + normal_positions,
            x.to(tl.bfloat16),
            mask=mask,
        )

        # Copy outlier bf16 bytes from slot to output.
        # Reinterpret 2 uint8 bytes as bfloat16 via scratch memory.
        outlier_dim = tl.arange(0, BLOCK_OUTLIER)
        outlier_mask = outlier_dim < n_outliers
        outlier_positions = tl.load(
            outlier_idx_ptr + outlier_dim,
            mask=outlier_mask,
            other=0,
        )
        # Copy outlier bytes (2 per value) to scratch, reload as bf16
        byte_offs = outlier_dim * 2
        lo = tl.load(
            slot_ptr + slot_base + byte_offs,
            mask=outlier_mask,
            other=0,
        )
        hi = tl.load(
            slot_ptr + slot_base + byte_offs + 1,
            mask=outlier_mask,
            other=0,
        )
        # Write byte pairs to scratch, then load as bf16
        scratch_outlier_u8 = (scratch_ptr + scratch_base).to(tl.pointer_type(tl.uint8))
        tl.store(
            scratch_outlier_u8 + byte_offs,
            lo,
            mask=outlier_mask,
        )
        tl.store(
            scratch_outlier_u8 + byte_offs + 1,
            hi,
            mask=outlier_mask,
        )
        scratch_outlier_bf16 = (scratch_ptr + scratch_base).to(
            tl.pointer_type(tl.bfloat16)
        )
        outlier_bf16 = tl.load(
            scratch_outlier_bf16 + outlier_dim,
            mask=outlier_mask,
            other=0.0,
        )
        tl.store(
            out_ptr + out_base + outlier_positions,
            outlier_bf16,
            mask=outlier_mask,
        )
    else:
        # No outliers: write sequentially
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

    Single kernel: unpack 4-bit → codebook → inv Hadamard → norm scale
    → write bf16 with outlier interleaving.

    Args:
        flat_slots: [N, slot_bytes] uint8, each row is one cache slot
        Returns: [N, head_size] bfloat16
    """
    N = flat_slots.shape[0]
    slot_bytes = flat_slots.shape[1]
    BLOCK_D = sign_flips.shape[0]
    LOG2_D = int(math.log2(BLOCK_D))
    outlier_u8_count = n_outliers * 2

    out = torch.empty(N, head_size, dtype=torch.bfloat16, device=flat_slots.device)
    scratch = torch.empty(N, BLOCK_D, dtype=torch.float32, device=flat_slots.device)

    has_outliers = normal_idx is not None and n_outliers > 0
    # Dummy pointers for no-outlier case
    if normal_idx is None:
        normal_idx = torch.empty(0, dtype=torch.int64, device=flat_slots.device)
    if outlier_idx is None:
        outlier_idx = torch.empty(0, dtype=torch.int64, device=flat_slots.device)

    BLOCK_PACKED = _next_power_of_2(packed_bytes)
    BLOCK_OUTLIER = max(_next_power_of_2(n_outliers), 1)
    BLOCK_HEAD = _next_power_of_2(head_size)

    grid = (N,)

    _fused_decode_kernel[grid](
        slot_ptr=flat_slots,
        signs_ptr=sign_flips,
        codebook_ptr=codebook,
        scratch_ptr=scratch,
        normal_idx_ptr=normal_idx,
        outlier_idx_ptr=outlier_idx,
        out_ptr=out,
        normal_size=normal_size,
        head_size=head_size,
        n_outliers=n_outliers,
        outlier_u8_count=outlier_u8_count,
        packed_bytes=packed_bytes,
        slot_bytes=slot_bytes,
        LOG2_D=LOG2_D,
        HAS_OUTLIERS=has_outliers,
        BLOCK_D=BLOCK_D,
        BLOCK_PACKED=BLOCK_PACKED,
        BLOCK_OUTLIER=BLOCK_OUTLIER,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=4,
        num_stages=1,
    )

    return out
