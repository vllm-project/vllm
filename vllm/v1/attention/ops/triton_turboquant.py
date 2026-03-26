# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for TurboQuant KV cache encode/decode.

Encode: rotate -> scalar quantize -> store indices + norm
Decode: load indices + norm -> codebook lookup -> unrotate

All tl.arange calls use BLOCK_D (next power of 2 >= head_size) with masking
to satisfy Triton's power-of-2 requirement.
"""

import math

import torch

from vllm.triton_utils import tl, triton


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# Encode kernel: rotate + quantize + store
# ---------------------------------------------------------------------------

@triton.jit
def _turboquant_encode_kernel(
    # Input K or V: [num_tokens, num_kv_heads, head_size]
    x_ptr,
    # Rotation matrix Pi^T: [head_size, head_size], row-major
    pit_ptr,
    # Boundaries: [num_centroids - 1]
    boundaries_ptr,
    # Output indices: [num_tokens, num_kv_heads, head_size] as uint8
    indices_ptr,
    # Output norms: [num_tokens, num_kv_heads] as float16
    norms_ptr,
    # Shapes
    head_size: tl.constexpr,
    num_boundaries: tl.constexpr,
    # Strides
    x_stride_token: tl.int64,
    x_stride_head: tl.int64,
    idx_stride_token: tl.int64,
    idx_stride_head: tl.int64,
    norm_stride_token: tl.int64,
    # Padded dimension (power of 2)
    BLOCK_D: tl.constexpr,
):
    """Encode one (token, head) pair: rotate, quantize to codebook indices."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Load input vector x[token, head, :] with masking
    dim_offs = tl.arange(0, BLOCK_D)
    mask = dim_offs < head_size
    x_base = token_idx * x_stride_token + head_idx * x_stride_head
    x_vec = tl.load(x_ptr + x_base + dim_offs, mask=mask, other=0.0).to(
        tl.float32)

    # Compute L2 norm
    norm_sq = tl.sum(x_vec * x_vec, axis=0)
    norm = tl.sqrt(norm_sq + 1e-16)
    x_normed = x_vec / norm

    idx_base = token_idx * idx_stride_token + head_idx * idx_stride_head

    # For each output dimension j: compute rotated value and quantize
    for j in range(head_size):
        # Load j-th column of Pi^T: PiT[i, j] = pit_ptr[i * head_size + j]
        pit_col = tl.load(pit_ptr + dim_offs * head_size + j, mask=mask,
                          other=0.0)
        y_j = tl.sum(x_normed * pit_col, axis=0)

        # Scalar quantize: count how many boundaries y_j exceeds
        # This is equivalent to torch.bucketize(y_j, boundaries)
        idx = 0  # Python int, promoted on first addition
        for b in range(num_boundaries):
            bnd = tl.load(boundaries_ptr + b)
            idx = idx + (y_j > bnd).to(tl.int32)

        tl.store(indices_ptr + idx_base + j, idx.to(tl.uint8))

    # Store norm
    tl.store(norms_ptr + token_idx * norm_stride_token + head_idx,
             norm.to(tl.float16))


# ---------------------------------------------------------------------------
# Decode kernel: load indices + norm -> codebook lookup -> unrotate
# ---------------------------------------------------------------------------

@triton.jit
def _turboquant_decode_kernel(
    # Input indices: [num_tokens, num_kv_heads, head_size] as uint8
    indices_ptr,
    # Input norms: [num_tokens, num_kv_heads] as float16
    norms_ptr,
    # Rotation matrix Pi: [head_size, head_size], row-major
    pi_ptr,
    # Codebook: [num_centroids]
    codebook_ptr,
    # Output: [num_tokens, num_kv_heads, head_size]
    out_ptr,
    # Shapes
    head_size: tl.constexpr,
    # Strides
    idx_stride_token: tl.int64,
    idx_stride_head: tl.int64,
    norm_stride_token: tl.int64,
    out_stride_token: tl.int64,
    out_stride_head: tl.int64,
    # Padded dimension (power of 2)
    BLOCK_D: tl.constexpr,
    OUTPUT_BF16: tl.constexpr,
):
    """Decode one (token, head) pair: codebook lookup, unrotate, scale."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    dim_offs = tl.arange(0, BLOCK_D)
    mask = dim_offs < head_size

    # Load indices and codebook lookup
    idx_base = token_idx * idx_stride_token + head_idx * idx_stride_head
    indices = tl.load(indices_ptr + idx_base + dim_offs, mask=mask,
                      other=0).to(tl.int32)
    # Gather from codebook, then zero out padding positions
    # (padding indices are 0, so codebook[0] would pollute the dot product)
    reconstructed = tl.load(codebook_ptr + indices)
    reconstructed = tl.where(mask, reconstructed, 0.0)

    # Load norm
    norm = tl.load(
        norms_ptr + token_idx * norm_stride_token + head_idx).to(tl.float32)

    # Unrotate: x_hat[j] = sum_i reconstructed[i] * Pi[i, j]
    out_base = token_idx * out_stride_token + head_idx * out_stride_head

    for j in range(head_size):
        pi_col = tl.load(pi_ptr + dim_offs * head_size + j, mask=mask,
                         other=0.0)
        val = tl.sum(reconstructed * pi_col, axis=0) * norm

        if OUTPUT_BF16:
            tl.store(out_ptr + out_base + j, val.to(tl.bfloat16))
        else:
            tl.store(out_ptr + out_base + j, val.to(tl.float16))


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def turboquant_encode(
    x: torch.Tensor,           # [num_tokens, num_kv_heads, head_size]
    pit: torch.Tensor,         # [head_size, head_size] rotation matrix transposed
    codebook: torch.Tensor,    # [num_centroids]
    boundaries: torch.Tensor,  # [num_centroids - 1]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode K or V vectors using TurboQuant.

    Returns:
        indices: [num_tokens, num_kv_heads, head_size] uint8
        norms: [num_tokens, num_kv_heads] float16
    """
    num_tokens, num_kv_heads, head_size = x.shape
    num_boundaries = boundaries.shape[0]
    BLOCK_D = _next_power_of_2(head_size)

    indices = torch.empty(
        (num_tokens, num_kv_heads, head_size),
        dtype=torch.uint8, device=x.device)
    norms = torch.empty(
        (num_tokens, num_kv_heads),
        dtype=torch.float16, device=x.device)

    grid = (num_tokens, num_kv_heads)

    _turboquant_encode_kernel[grid](
        x_ptr=x,
        pit_ptr=pit,
        boundaries_ptr=boundaries,
        indices_ptr=indices,
        norms_ptr=norms,
        head_size=head_size,
        num_boundaries=num_boundaries,
        x_stride_token=x.stride(0),
        x_stride_head=x.stride(1),
        idx_stride_token=indices.stride(0),
        idx_stride_head=indices.stride(1),
        norm_stride_token=norms.stride(0),
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=1,
    )

    return indices, norms


def turboquant_decode(
    indices: torch.Tensor,     # [num_tokens, num_kv_heads, head_size] uint8
    norms: torch.Tensor,       # [num_tokens, num_kv_heads] float16
    pi: torch.Tensor,          # [head_size, head_size] rotation matrix
    codebook: torch.Tensor,    # [num_centroids]
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode TurboQuant indices back to K or V vectors.

    Returns:
        out: [num_tokens, num_kv_heads, head_size] in output_dtype
    """
    num_tokens, num_kv_heads, head_size = indices.shape
    BLOCK_D = _next_power_of_2(head_size)

    out = torch.empty(
        (num_tokens, num_kv_heads, head_size),
        dtype=output_dtype, device=indices.device)

    grid = (num_tokens, num_kv_heads)

    _turboquant_decode_kernel[grid](
        indices_ptr=indices,
        norms_ptr=norms,
        pi_ptr=pi,
        codebook_ptr=codebook,
        out_ptr=out,
        head_size=head_size,
        idx_stride_token=indices.stride(0),
        idx_stride_head=indices.stride(1),
        norm_stride_token=norms.stride(0),
        out_stride_token=out.stride(0),
        out_stride_head=out.stride(1),
        BLOCK_D=BLOCK_D,
        OUTPUT_BF16=(output_dtype == torch.bfloat16),
        num_warps=4,
        num_stages=1,
    )

    return out
