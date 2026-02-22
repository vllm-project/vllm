# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused bilinear position embedding interpolation Triton kernel.

Replaces `fast_pos_embed_interpolate()` in qwen3_vl.py with a single-pass
kernel that:
  1. Computes bilinear interpolation weights on-the-fly
  2. Reads 4 embedding table corners in one pass
  3. Writes the weighted sum directly in the spatial-merge-permuted layout

This eliminates ~15 intermediate tensor allocations per image/video that the
PyTorch implementation creates (meshgrids, index tensors, stacked weights, etc.)
"""

import torch

from vllm.triton_utils import tl, triton


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 256}, num_warps=4),
        triton.Config({"BLOCK_D": 512}, num_warps=8),
        triton.Config({"BLOCK_D": 1024}, num_warps=8),
    ],
    key=["hidden_dim"],
)
@triton.jit
def _fused_bilinear_pos_embed_kernel(
    # Embedding table: (num_grid_per_side^2, hidden_dim), row-major
    embed_table_ptr,
    # Output: (h * w, hidden_dim), but written in permuted order
    output_ptr,
    # Precomputed per-row data (1D arrays of length h)
    h_floor_ptr,  # int64: floor indices along h
    h_ceil_ptr,  # int64: ceil indices along h
    dh_ptr,  # float32: fractional part along h
    # Precomputed per-col data (1D arrays of length w)
    w_floor_ptr,  # int64: floor indices along w
    w_ceil_ptr,  # int64: ceil indices along w
    dw_ptr,  # float32: fractional part along w
    # Scalar parameters
    h: tl.constexpr,  # grid height
    w: tl.constexpr,  # grid width
    hidden_dim: tl.constexpr,
    num_grid_per_side: tl.constexpr,
    merge_size: tl.constexpr,
    # Block size for hidden dim tiling
    BLOCK_D: tl.constexpr,
):
    """
    Each program handles one spatial position (i, j) in the h x w output grid.
    It iterates over hidden_dim in tiles of BLOCK_D, loading 4 corners from the
    embedding table, computing the bilinear weighted sum, and writing the result
    to the output at the spatially-permuted position.

    Spatial merge permutation:
      Input layout:  (h//m, m, w//m, m, D)  [row-major in (i, j)]
      Output layout: (h//m, w//m, m, m, D)  [permuted]
      For position (i, j):
        out_idx = ((i//m)*(w//m) + j//m) * m*m + (i%m)*m + (j%m)
    """
    pid = tl.program_id(0)

    # Derive (i, j) from flat position index
    i = pid // w
    j = pid % w

    # Load precomputed floor/ceil/frac for this row and column
    hf = tl.load(h_floor_ptr + i)
    hc = tl.load(h_ceil_ptr + i)
    frac_h = tl.load(dh_ptr + i)

    wf = tl.load(w_floor_ptr + j)
    wc = tl.load(w_ceil_ptr + j)
    frac_w = tl.load(dw_ptr + j)

    # Bilinear weights (reuse w11 trick to minimize multiplications)
    # w00 = (1 - frac_h) * (1 - frac_w)
    # w01 = (1 - frac_h) * frac_w
    # w10 = frac_h * (1 - frac_w)
    # w11 = frac_h * frac_w
    w11 = frac_h * frac_w
    w10 = frac_h - w11
    w01 = frac_w - w11
    w00 = 1.0 - frac_h - w01  # = 1 - frac_h - frac_w + w11

    # Embedding table row offsets for the 4 corners
    # Index = h_idx * num_grid_per_side + w_idx
    idx00 = (hf * num_grid_per_side + wf) * hidden_dim
    idx01 = (hf * num_grid_per_side + wc) * hidden_dim
    idx10 = (hc * num_grid_per_side + wf) * hidden_dim
    idx11 = (hc * num_grid_per_side + wc) * hidden_dim

    # Compute output position with spatial merge permutation
    w_div_m = w // merge_size
    m2 = merge_size * merge_size

    i_block = i // merge_size
    j_block = j // merge_size
    i_local = i % merge_size
    j_local = j % merge_size

    out_flat = (i_block * w_div_m + j_block) * m2 + i_local * merge_size + j_local
    out_row_offset = out_flat * hidden_dim

    # Iterate over hidden_dim in tiles of BLOCK_D
    for d_start in tl.range(0, hidden_dim, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        mask = d_offsets < hidden_dim

        # Load 4 corner embeddings
        e00 = tl.load(embed_table_ptr + idx00 + d_offsets, mask=mask, other=0.0)
        e01 = tl.load(embed_table_ptr + idx01 + d_offsets, mask=mask, other=0.0)
        e10 = tl.load(embed_table_ptr + idx10 + d_offsets, mask=mask, other=0.0)
        e11 = tl.load(embed_table_ptr + idx11 + d_offsets, mask=mask, other=0.0)

        # Cast weights to match embedding dtype for the FMA
        w00_cast = w00.to(e00.dtype)
        w01_cast = w01.to(e00.dtype)
        w10_cast = w10.to(e00.dtype)
        w11_cast = w11.to(e00.dtype)

        # Bilinear weighted sum
        result = w00_cast * e00 + w01_cast * e01 + w10_cast * e10 + w11_cast * e11

        # Write to output at the permuted position
        tl.store(output_ptr + out_row_offset + d_offsets, result, mask=mask)


def fused_pos_embed_interpolate(
    grid_thw: list[list[int]],
    embed_weight: torch.Tensor,
    num_grid_per_side: int,
    merge_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Fused bilinear position embedding interpolation using Triton.

    Drop-in replacement for Qwen3_VisionTransformer.fast_pos_embed_interpolate().

    Args:
        grid_thw: List of [t, h, w] for each image/video in the batch.
        embed_weight: The pos_embed weight tensor of shape
            (num_grid_per_side^2, hidden_dim).
        num_grid_per_side: Number of grid positions per side in the embedding
            table (sqrt of num_position_embeddings).
        merge_size: spatial_merge_size parameter from the model config.
        dtype: Output dtype (e.g., torch.float16, torch.bfloat16).
        device: CUDA device.

    Returns:
        Concatenated position embeddings for all grid entries, shaped
        (total_tokens, hidden_dim) where total_tokens =
        sum(t * h * w for t, h, w in grid_thw).
    """
    hidden_dim = embed_weight.shape[1]
    outputs = []

    for t, h, w in grid_thw:
        # Precompute 1D linspace floor/ceil/frac on CPU-like path, then move
        # to GPU. These are tiny (h and w elements each).
        h_idxs = torch.linspace(
            0, num_grid_per_side - 1, h, dtype=torch.float32, device=device
        )
        w_idxs = torch.linspace(
            0, num_grid_per_side - 1, w, dtype=torch.float32, device=device
        )

        h_floor = h_idxs.to(torch.int64)
        w_floor = w_idxs.to(torch.int64)
        h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
        w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

        dh = h_idxs - h_floor.float()
        dw = w_idxs - w_floor.float()

        # Allocate output: h*w positions, each with hidden_dim values
        num_positions = h * w
        output = torch.empty(num_positions, hidden_dim, dtype=dtype, device=device)

        # Launch kernel: one program per spatial position
        grid = (num_positions,)
        _fused_bilinear_pos_embed_kernel[grid](
            embed_weight,
            output,
            h_floor,
            h_ceil,
            dh,
            w_floor,
            w_ceil,
            dw,
            h=h,
            w=w,
            hidden_dim=hidden_dim,
            num_grid_per_side=num_grid_per_side,
            merge_size=merge_size,
        )

        # Handle t-repeat: expand across temporal dimension
        # output is already in merge-permuted layout: (h*w, hidden_dim)
        # For t > 1, repeat the pattern t times
        if t > 1:
            output = output.unsqueeze(0).expand(t, -1, -1).reshape(-1, hidden_dim)

        outputs.append(output)

    return torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]


def _pytorch_reference_pos_embed_interpolate(
    grid_thw: list[list[int]],
    embed_weight: torch.Tensor,
    num_grid_per_side: int,
    merge_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Pure PyTorch reference implementation matching the original
    fast_pos_embed_interpolate(). Used for correctness testing.
    """
    hidden_dim = embed_weight.shape[1]
    outputs = []

    for t, h, w in grid_thw:
        h_idxs = torch.linspace(
            0, num_grid_per_side - 1, h, dtype=torch.float32, device=device
        )
        w_idxs = torch.linspace(
            0, num_grid_per_side - 1, w, dtype=torch.float32, device=device
        )

        h_floor = h_idxs.to(torch.long)
        w_floor = w_idxs.to(torch.long)
        h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
        w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

        dh = h_idxs - h_floor
        dw = w_idxs - w_floor

        dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
        h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
        h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

        w11 = dh_grid * dw_grid
        w10 = dh_grid - w11
        w01 = dw_grid - w11
        w00 = 1 - dh_grid - w01

        h_grid = torch.stack(
            [h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid]
        )
        w_grid = torch.stack(
            [w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid]
        )
        h_grid_idx = h_grid * num_grid_per_side

        indices = (h_grid_idx + w_grid).reshape(4, -1)
        weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
        weights = weights.to(dtype=dtype)

        embeds = embed_weight[indices]
        embeds *= weights
        combined = embeds.sum(dim=0)

        m_size = merge_size
        combined = combined.reshape(
            h // m_size, m_size, w // m_size, m_size, hidden_dim
        )
        combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
        repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
        outputs.append(repeated)

    return torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
