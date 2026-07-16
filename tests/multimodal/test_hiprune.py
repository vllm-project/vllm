# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HiPrune visual token selection (vllm/multimodal/hiprune.py)."""

import pytest
import torch

from vllm.multimodal.hiprune import (
    aggregate_patch_attention,
    compute_retained_tokens_count,
    compute_soft_token_grid,
    hiprune_select,
)

POOL_K = 3
PATCH_BUDGET = 2520  # Gemma4: max_soft_tokens (280) * pooling_kernel_size**2


def _make_position_ids(patch_w: int, patch_h: int) -> torch.Tensor:
    """Replicate Gemma4ImageProcessor position ids: real patches in
    row-major (x, y) order, padded to the fixed budget with (-1, -1)."""
    grid = torch.stack(
        torch.meshgrid(torch.arange(patch_w), torch.arange(patch_h), indexing="xy"),
        dim=-1,
    ).reshape(patch_w * patch_h, 2)
    pad = torch.full((PATCH_BUDGET - patch_w * patch_h, 2), -1)
    return torch.cat([grid, pad])


@pytest.mark.parametrize("num_tokens", [1, 12, 255, 260, 280])
@pytest.mark.parametrize("ratio", [0.05, 0.11, 0.14, 0.223, 0.5, 1.0])
def test_retained_count_bounds(num_tokens: int, ratio: float):
    kept = compute_retained_tokens_count(num_tokens, ratio)
    assert 1 <= kept <= num_tokens


@pytest.mark.parametrize("patch_w,patch_h", [(12, 9), (45, 39), (15, 51)])
def test_soft_token_grid(patch_w: int, patch_h: int):
    pos = _make_position_ids(patch_w, patch_h)
    valid, grid_w, grid_h, kernel_idx = compute_soft_token_grid(pos, POOL_K)

    assert valid.sum() == patch_w * patch_h
    assert (grid_w, grid_h) == (patch_w // POOL_K, patch_h // POOL_K)

    n_tokens = grid_w * grid_h
    assert kernel_idx.min() >= 0
    assert kernel_idx.max() == n_tokens - 1
    # Every soft token covers exactly k^2 patches.
    counts = torch.bincount(kernel_idx, minlength=n_tokens)
    assert (counts == POOL_K**2).all()


def test_aggregation_ignores_padding_rows():
    """Garbage attention rows at padding positions must not affect scores."""
    patch_w, patch_h = 12, 9
    n_real = patch_w * patch_h
    pos = _make_position_ids(patch_w, patch_h)
    valid, grid_w, grid_h, kernel_idx = compute_soft_token_grid(pos, POOL_K)
    n_tokens = grid_w * grid_h

    heads = 4
    attn = torch.zeros(heads, PATCH_BUDGET, PATCH_BUDGET)
    real_attn = torch.randn(heads, n_real, n_real).softmax(dim=-1)
    attn[:, :n_real, :n_real] = real_attn
    # Garbage rows for padding queries.
    attn[:, n_real:, :] = torch.rand(
        heads, PATCH_BUDGET - n_real, PATCH_BUDGET
    ).softmax(dim=-1)

    scores = aggregate_patch_attention(attn, valid, kernel_idx, n_tokens)
    assert scores.shape == (n_tokens,)
    # Still a probability distribution over soft tokens.
    assert torch.isclose(scores.sum(), torch.tensor(1.0), atol=1e-5)

    # Identical result when computed from the unpadded attention alone.
    ref = real_attn.float().mean(dim=0).mean(dim=0)
    ref_scores = (
        torch.nn.functional.one_hot(kernel_idx.long(), n_tokens).float().T @ ref
    )
    assert torch.allclose(scores, ref_scores, atol=1e-6)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("ratio", [0.11, 0.14, 0.223, 0.5])
def test_selection_invariants(seed: int, ratio: float):
    torch.manual_seed(seed)
    grid_w, grid_h = 20, 13  # a realistic Gemma4 soft-token grid
    n_tokens = grid_w * grid_h
    shallow = torch.rand(n_tokens).softmax(dim=0)
    deep = torch.rand(n_tokens).softmax(dim=0)

    anchor, buffer, register, kept = hiprune_select(
        shallow, deep, n_tokens, grid_w, ratio
    )

    # Exact budget: the count the processor promised via placeholders.
    assert kept.sum().item() == compute_retained_tokens_count(n_tokens, ratio)

    # Categories are disjoint and jointly equal the kept set.
    all_idx = torch.cat([anchor, buffer, register])
    assert all_idx.unique().numel() == all_idx.numel()
    mask_from_cats = torch.zeros(n_tokens, dtype=torch.bool)
    mask_from_cats[all_idx] = True
    assert torch.equal(mask_from_cats, kept)

    # Buffers are spatial neighbors of anchors.
    if buffer.numel() > 0:
        neighbor_sets = torch.cat(
            [anchor - 1, anchor + 1, anchor - grid_w, anchor + grid_w]
        ).clamp(0, n_tokens - 1)
        assert torch.isin(buffer, neighbor_sets).all()


def test_selection_deterministic():
    torch.manual_seed(7)
    n_tokens, grid_w = 260, 20
    shallow = torch.rand(n_tokens).softmax(dim=0)
    deep = torch.rand(n_tokens).softmax(dim=0)
    first = hiprune_select(shallow, deep, n_tokens, grid_w, 0.14)
    second = hiprune_select(shallow, deep, n_tokens, grid_w, 0.14)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_full_retention_keeps_everything():
    torch.manual_seed(3)
    n_tokens, grid_w = 255, 15
    shallow = torch.rand(n_tokens).softmax(dim=0)
    deep = torch.rand(n_tokens).softmax(dim=0)
    _, _, _, kept = hiprune_select(shallow, deep, n_tokens, grid_w, 1.0)
    assert kept.all()
