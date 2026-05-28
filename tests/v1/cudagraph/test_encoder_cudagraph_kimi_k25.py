# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight correctness tests for Kimi K2.5-VL encoder CUDA graph pieces.

These tests exercise the two rewrites that make the Kimi-K2.5 vision tower
capturable:

* ``tpool_patch_merger_indexed`` — a static-shape variant of
  ``tpool_patch_merger`` that consumes precomputed index buffers instead
  of iterating over a Python list of per-item grids.
* ``MoonViT3dPretrainedModel.prepare_encoder_metadata`` — feeds the
  encoder / merger with those buffers.

Both tests run on CPU; actual CUDA graph capture/replay is covered by the
generic tests in ``test_encoder_cudagraph.py`` and by manual server-level
validation noted in the plan.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm.model_executor.models.kimi_k25_vit import (
    _build_tpool_indices,
    tpool_patch_merger,
    tpool_patch_merger_indexed,
)


def _random_patches(
    grid_thw_list: list[tuple[int, int, int]],
    hidden: int,
    seed: int = 0,
) -> torch.Tensor:
    torch.manual_seed(seed)
    n = sum(t * h * w for t, h, w in grid_thw_list)
    return torch.randn(n, hidden, dtype=torch.float32)


@pytest.mark.parametrize(
    "grid_thw_list",
    [
        # single image
        [(1, 4, 4)],
        # two images of different sizes
        [(1, 4, 4), (1, 6, 8)],
        # video chunk (t > 1) + image
        [(4, 4, 4), (1, 2, 2)],
        # multiple video chunks
        [(4, 4, 6), (4, 6, 4)],
    ],
)
def test_indexed_tpool_matches_eager(grid_thw_list):
    """The indexed implementation must match the eager per-item loop bit-for-bit
    (up to float rounding) when no padding is involved.

    Slot 0 of the indexed output is the reserved dead slot; real output
    slots live at indices 1..max_output_slots.
    """
    kh, kw = 2, 2
    hidden = 8
    x = _random_patches(grid_thw_list, hidden)

    # Eager reference.
    grid_thws_tensor = torch.tensor(grid_thw_list, dtype=torch.int32)
    eager_list = tpool_patch_merger(x, grid_thws_tensor, merge_kernel_size=(kh, kw))
    eager_packed = torch.cat(eager_list, dim=0)  # (sum(nh*nw), kh*kw, hidden)

    # Indexed implementation with tight buffer sizes (no padding).
    total_input_patches = sum(t * h * w for t, h, w in grid_thw_list)
    total_post_tpool = sum(h * w for t, h, w in grid_thw_list)
    total_output_slots = sum((h // kh) * (w // kw) for t, h, w in grid_thw_list)
    (
        temporal_gather_idx_np,
        temporal_divisor_np,
        spatial_gather_idx_np,
    ) = _build_tpool_indices(
        grid_thw_list,
        kh=kh,
        kw=kw,
        max_total_patches=total_input_patches,
        max_post_tpool_patches=total_post_tpool,
        max_output_slots=total_output_slots,
    )
    metadata = {
        "tpool_temporal_gather_idx": torch.from_numpy(temporal_gather_idx_np),
        "tpool_temporal_divisor": torch.from_numpy(temporal_divisor_np),
        "tpool_spatial_gather_idx": torch.from_numpy(spatial_gather_idx_np),
    }
    indexed_out = tpool_patch_merger_indexed(x, metadata, kh=kh, kw=kw)

    # The indexed output does not have a leading dead slot because it's
    # indexed by ``spatial_gather_idx`` which only covers real slots.
    assert indexed_out.shape == eager_packed.shape
    torch.testing.assert_close(indexed_out, eager_packed, rtol=1e-5, atol=1e-6)


def test_indexed_tpool_dead_slot_is_zero():
    """Slot 0 of the post-tpool space is reserved as the dead slot."""
    kh, kw = 2, 2
    grid_thw_list = [(1, 4, 4)]
    total_post_tpool = 16
    total_input_patches = 16
    total_output_slots = 4

    (temporal_gather_idx_np, temporal_divisor_np, spatial_gather_idx_np) = (
        _build_tpool_indices(
            grid_thw_list,
            kh=kh,
            kw=kw,
            max_total_patches=total_input_patches,
            max_post_tpool_patches=total_post_tpool,
            max_output_slots=total_output_slots,
        )
    )
    # Dead slot stays at index 0 with divisor 1.0 (not touched by real items).
    assert temporal_divisor_np[0] == pytest.approx(1.0)
    # All real indices are strictly positive.
    assert int(temporal_gather_idx_np.min()) >= 1
    assert int(spatial_gather_idx_np.min()) >= 1


def test_indexed_tpool_temporal_pooling():
    """Temporal averaging must divide by ``t`` per post-tpool slot."""
    kh, kw = 2, 2
    hidden = 3
    # One video chunk with t=3, h=w=2 (one output slot).
    grid_thw_list = [(3, 2, 2)]
    n_patches = 3 * 2 * 2

    # Craft input so each of the 3 temporal frames contributes a distinct,
    # easy-to-verify value.  Patches are laid out as
    #   frame0: 4 patches at t_idx=0
    #   frame1: 4 patches at t_idx=1
    #   frame2: 4 patches at t_idx=2
    # Frame i has all its patches set to i. The expected mean is (0+1+2)/3 = 1.
    x = torch.zeros(n_patches, hidden, dtype=torch.float32)
    x[0:4] = 0.0
    x[4:8] = 1.0
    x[8:12] = 2.0

    (
        temporal_gather_idx_np,
        temporal_divisor_np,
        spatial_gather_idx_np,
    ) = _build_tpool_indices(
        grid_thw_list,
        kh=kh,
        kw=kw,
        max_total_patches=n_patches,
        max_post_tpool_patches=4,  # h*w for this item
        max_output_slots=1,
    )
    # Real slots live at indices 1..4 (slot 0 is the dead slot with
    # divisor 1.0).  Each of the 3 temporal frames contributes to one
    # slot, so the divisor there is 3.
    np.testing.assert_allclose(temporal_divisor_np[0], 1.0)
    np.testing.assert_allclose(temporal_divisor_np[1:5], 3.0)

    metadata = {
        "tpool_temporal_gather_idx": torch.from_numpy(temporal_gather_idx_np),
        "tpool_temporal_divisor": torch.from_numpy(temporal_divisor_np),
        "tpool_spatial_gather_idx": torch.from_numpy(spatial_gather_idx_np),
    }
    out = tpool_patch_merger_indexed(x, metadata, kh=kh, kw=kw)
    assert out.shape == (1, kh * kw, hidden)
    torch.testing.assert_close(out, torch.ones_like(out), rtol=1e-5, atol=1e-6)
