# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels for TurboQuant with Hadamard rotation.

Single-kernel encode: normalize → sign_flip → Hadamard → quantize
Single-kernel decode: codebook lookup → Hadamard → sign_flip → scale

The Hadamard butterfly uses XOR-based partner indexing with a small
scratch buffer in global memory (stays in L1 cache, ~512 bytes per
thread block).
"""

import math

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_hadamard_encode_kernel(
    # Input: [num_tokens, num_kv_heads, head_size]
    x_ptr,
    # Sign flips: [BLOCK_D] float32
    signs_ptr,
    # Boundaries: [num_centroids - 1] float32
    boundaries_ptr,
    # Scratch buffer: [num_tokens * num_kv_heads, BLOCK_D] float32
    scratch_ptr,
    # Output indices: [num_tokens, num_kv_heads, head_size] uint8
    indices_ptr,
    # Output norms: [num_tokens, num_kv_heads] float16
    norms_ptr,
    # Shapes
    head_size: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_boundaries: tl.constexpr,
    LOG2_D: tl.constexpr,
    # Strides
    x_stride_token: tl.int64,
    x_stride_head: tl.int64,
    idx_stride_token: tl.int64,
    idx_stride_head: tl.int64,
    norm_stride_token: tl.int64,
    BLOCK_D: tl.constexpr,
):
    """Fused encode: norm → sign_flip → Hadamard → quantize."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    scratch_row = token_idx * num_kv_heads + head_idx

    dim_offs = tl.arange(0, BLOCK_D)
    mask = dim_offs < head_size

    # Load and normalize
    x_base = token_idx * x_stride_token + head_idx * x_stride_head
    x = tl.load(x_ptr + x_base + dim_offs, mask=mask, other=0.0).to(tl.float32)
    norm_sq = tl.sum(x * x, axis=0)
    norm = tl.sqrt(norm_sq + 1e-16)
    x = x / norm

    # Sign flips
    signs = tl.load(signs_ptr + dim_offs)
    x = x * signs

    # Store to scratch for Hadamard butterfly
    scratch_base = scratch_row * BLOCK_D
    tl.store(scratch_ptr + scratch_base + dim_offs, x)

    # Hadamard butterfly: log2(BLOCK_D) passes.
    # Barrier after each store to prevent inter-warp races
    # (element i in warp 0 may partner with element j in warp 1).
    h = 1
    for _level in range(LOG2_D):
        partner = dim_offs ^ h
        val_self = tl.load(scratch_ptr + scratch_base + dim_offs)
        val_partner = tl.load(scratch_ptr + scratch_base + partner)
        is_lower = (dim_offs & h) == 0
        result = tl.where(is_lower, val_self + val_partner, val_partner - val_self)
        tl.debug_barrier()
        tl.store(scratch_ptr + scratch_base + dim_offs, result)
        tl.debug_barrier()
        h = h * 2

    # Load result, scale
    x = tl.load(scratch_ptr + scratch_base + dim_offs)
    scale = 1.0 / tl.sqrt(float(BLOCK_D))
    x = x * scale

    # Quantize: vectorized bucketize
    # For each element, count boundaries exceeded
    idx = tl.zeros([BLOCK_D], dtype=tl.int32)
    for b in range(num_boundaries):
        bnd = tl.load(boundaries_ptr + b)
        idx = idx + (x > bnd).to(tl.int32)

    # Store indices and norm
    idx_base = token_idx * idx_stride_token + head_idx * idx_stride_head
    tl.store(indices_ptr + idx_base + dim_offs, idx.to(tl.uint8), mask=mask)
    tl.store(norms_ptr + token_idx * norm_stride_token + head_idx, norm.to(tl.float16))


@triton.jit
def _fused_hadamard_decode_kernel(
    # Input indices: [num_tokens, num_kv_heads, head_size] uint8
    indices_ptr,
    # Input norms: [num_tokens, num_kv_heads] float16
    norms_ptr,
    # Sign flips: [BLOCK_D] float32
    signs_ptr,
    # Codebook: [num_centroids] float32
    codebook_ptr,
    # Scratch buffer: [num_tokens * num_kv_heads, BLOCK_D] float32
    scratch_ptr,
    # Output: [num_tokens, num_kv_heads, head_size]
    out_ptr,
    # Shapes
    head_size: tl.constexpr,
    num_kv_heads: tl.constexpr,
    LOG2_D: tl.constexpr,
    # Strides
    idx_stride_token: tl.int64,
    idx_stride_head: tl.int64,
    norm_stride_token: tl.int64,
    out_stride_token: tl.int64,
    out_stride_head: tl.int64,
    BLOCK_D: tl.constexpr,
    OUTPUT_BF16: tl.constexpr,
):
    """Fused decode: codebook → Hadamard → sign_flip → scale."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    scratch_row = token_idx * num_kv_heads + head_idx

    dim_offs = tl.arange(0, BLOCK_D)
    mask = dim_offs < head_size

    # Load indices and codebook lookup
    idx_base = token_idx * idx_stride_token + head_idx * idx_stride_head
    indices = tl.load(indices_ptr + idx_base + dim_offs, mask=mask, other=0).to(
        tl.int32
    )
    reconstructed = tl.load(codebook_ptr + indices)
    reconstructed = tl.where(mask, reconstructed, 0.0)

    # Store to scratch for Hadamard butterfly
    scratch_base = scratch_row * BLOCK_D
    tl.store(scratch_ptr + scratch_base + dim_offs, reconstructed)

    # Hadamard butterfly (inverse = same as forward, just scale)
    h = 1
    for _level in range(LOG2_D):
        partner = dim_offs ^ h
        val_self = tl.load(scratch_ptr + scratch_base + dim_offs)
        val_partner = tl.load(scratch_ptr + scratch_base + partner)
        is_lower = (dim_offs & h) == 0
        result = tl.where(is_lower, val_self + val_partner, val_partner - val_self)
        tl.store(scratch_ptr + scratch_base + dim_offs, result)
        h = h * 2

    # Load, scale, sign flip
    tl.debug_barrier()
    x = tl.load(scratch_ptr + scratch_base + dim_offs)
    scale = 1.0 / tl.sqrt(float(BLOCK_D))
    x = x * scale

    # Sign flips (inverse = same signs)
    signs = tl.load(signs_ptr + dim_offs)
    x = x * signs

    # Scale by norm
    norm = tl.load(norms_ptr + token_idx * norm_stride_token + head_idx).to(tl.float32)
    x = x * norm

    # Store output
    out_base = token_idx * out_stride_token + head_idx * out_stride_head
    if OUTPUT_BF16:
        tl.store(out_ptr + out_base + dim_offs, x.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + out_base + dim_offs, x.to(tl.float16), mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def hadamard_turboquant_encode(
    x: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
    sign_flips: torch.Tensor,  # [BLOCK_D] float32
    codebook: torch.Tensor,  # [num_centroids] (unused, kept for API compat)
    boundaries: torch.Tensor,  # [num_centroids - 1]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Hadamard encode: normalize → sign_flip → FWHT → quantize."""
    num_tokens, num_kv_heads, head_size = x.shape
    BLOCK_D = sign_flips.shape[0]  # padded to power of 2
    LOG2_D = int(math.log2(BLOCK_D))
    num_boundaries = boundaries.shape[0]

    indices = torch.empty(
        (num_tokens, num_kv_heads, head_size), dtype=torch.uint8, device=x.device
    )
    norms = torch.empty(
        (num_tokens, num_kv_heads), dtype=torch.float16, device=x.device
    )
    scratch = torch.empty(
        (num_tokens * num_kv_heads, BLOCK_D), dtype=torch.float32, device=x.device
    )

    grid = (num_tokens, num_kv_heads)

    _fused_hadamard_encode_kernel[grid](
        x_ptr=x,
        signs_ptr=sign_flips,
        boundaries_ptr=boundaries,
        scratch_ptr=scratch,
        indices_ptr=indices,
        norms_ptr=norms,
        head_size=head_size,
        num_kv_heads=num_kv_heads,
        num_boundaries=num_boundaries,
        LOG2_D=LOG2_D,
        x_stride_token=x.stride(0),
        x_stride_head=x.stride(1),
        idx_stride_token=indices.stride(0),
        idx_stride_head=indices.stride(1),
        norm_stride_token=norms.stride(0),
        BLOCK_D=BLOCK_D,
        # num_warps=1: required for Hadamard butterfly correctness.
        # Multi-warp causes inter-warp races on scratch buffer.
        num_warps=1,
        num_stages=1,
    )

    return indices, norms


def hadamard_turboquant_decode(
    indices: torch.Tensor,  # [num_tokens, num_kv_heads, head_size] uint8
    norms: torch.Tensor,  # [num_tokens, num_kv_heads] float16
    sign_flips: torch.Tensor,  # [BLOCK_D] float32
    codebook: torch.Tensor,  # [num_centroids]
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Fused Hadamard decode: codebook → FWHT → sign_flip → scale."""
    num_tokens, num_kv_heads, head_size = indices.shape
    BLOCK_D = sign_flips.shape[0]
    LOG2_D = int(math.log2(BLOCK_D))

    out = torch.empty(
        (num_tokens, num_kv_heads, head_size), dtype=output_dtype, device=indices.device
    )
    scratch = torch.empty(
        (num_tokens * num_kv_heads, BLOCK_D), dtype=torch.float32, device=indices.device
    )

    grid = (num_tokens, num_kv_heads)

    _fused_hadamard_decode_kernel[grid](
        indices_ptr=indices,
        norms_ptr=norms,
        signs_ptr=sign_flips,
        codebook_ptr=codebook,
        scratch_ptr=scratch,
        out_ptr=out,
        head_size=head_size,
        num_kv_heads=num_kv_heads,
        LOG2_D=LOG2_D,
        idx_stride_token=indices.stride(0),
        idx_stride_head=indices.stride(1),
        norm_stride_token=norms.stride(0),
        out_stride_token=out.stride(0),
        out_stride_head=out.stride(1),
        BLOCK_D=BLOCK_D,
        OUTPUT_BF16=(output_dtype == torch.bfloat16),
        num_warps=1,
        num_stages=1,
    )

    return out
