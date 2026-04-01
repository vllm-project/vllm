# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy tests for the fused Triton bilinear position-embedding kernel.

Compares ``triton_pos_embed_interpolate`` against the pure-PyTorch
``pos_embed_interpolate_native`` across a variety of grid shapes and dtypes.
"""

import pytest
import torch

from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.model_executor.models.qwen3_vl import (
        pos_embed_interpolate_native,
        triton_pos_embed_interpolate,
    )


DTYPES = [torch.float32, torch.bfloat16]
# Qwen3-VL default
NUM_GRID_PER_SIDE = 48
SPATIAL_MERGE_SIZE = 2
HIDDEN_DIM = 1152

# 4 square + 4 non-square grids (h, w divisible by spatial_merge_size=2)
SQUARE_GRIDS = [(1, 4, 4), (1, 16, 16), (1, 32, 32), (1, 48, 48)]
NON_SQUARE_GRIDS = [(1, 8, 16), (1, 14, 20), (1, 32, 48), (1, 60, 80)]
ALL_GRIDS = SQUARE_GRIDS + NON_SQUARE_GRIDS


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize(
    "grid_thw",
    ALL_GRIDS,
    ids=[f"{t}x{h}x{w}" for t, h, w in ALL_GRIDS],
)
def test_triton_matches_native(
    grid_thw: tuple[int, int, int],
    dtype: torch.dtype,
) -> None:
    """Triton kernel output must match the native PyTorch implementation."""
    t, h, w = grid_thw
    device = "cuda"

    # Scale to match real Qwen3-VL pos_embed weight distribution (std~0.23).
    torch.manual_seed(42)
    embed_weight = (
        torch.randn(
            NUM_GRID_PER_SIDE * NUM_GRID_PER_SIDE,
            HIDDEN_DIM,
            device=device,
            dtype=dtype,
        )
        * 0.25
    )

    native_out = pos_embed_interpolate_native(
        embed_weight, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )
    triton_out = triton_pos_embed_interpolate(
        embed_weight, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )

    assert native_out.shape == triton_out.shape, (
        f"Shape mismatch: native {native_out.shape} vs triton {triton_out.shape}"
    )

    # Small numerical differences arise from the precomputed h/w_scale
    # in the triton kernel vs torch.linspace in the native path, which can
    # cause single-ULP output differences
    # in a handful of elements.
    atol = {torch.float32: 5e-5, torch.bfloat16: 1e-2}[dtype]
    rtol = {torch.float32: 1e-5, torch.bfloat16: 1e-2}[dtype]
    torch.testing.assert_close(triton_out, native_out, atol=atol, rtol=rtol)


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_temporal_repeat(dtype: torch.dtype) -> None:
    """Verify temporal dimension t > 1 correctly repeats the spatial pattern."""
    device = "cuda"
    h, w = 16, 16
    t_single, t_multi = 1, 3

    # Scale to match real Qwen3-VL pos_embed weight distribution (std~0.23).
    torch.manual_seed(42)
    embed_weight = (
        torch.randn(
            NUM_GRID_PER_SIDE * NUM_GRID_PER_SIDE,
            HIDDEN_DIM,
            device=device,
            dtype=dtype,
        )
        * 0.25
    )

    out_single = triton_pos_embed_interpolate(
        embed_weight,
        t_single,
        h,
        w,
        NUM_GRID_PER_SIDE,
        SPATIAL_MERGE_SIZE,
        dtype,
    )
    out_multi = triton_pos_embed_interpolate(
        embed_weight,
        t_multi,
        h,
        w,
        NUM_GRID_PER_SIDE,
        SPATIAL_MERGE_SIZE,
        dtype,
    )

    expected = out_single.repeat(t_multi, 1)
    torch.testing.assert_close(out_multi, expected, atol=0, rtol=0)
