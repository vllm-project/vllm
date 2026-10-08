# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.multimodal.image_prune.rate import (
    _get_random_projections,
    compute_retained_tokens_count,
    compute_retention_mask_rate,
)


def _fake_image_embeds(
    num_tokens: int, hidden: int = 64, seed: int = 0
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(num_tokens, hidden, generator=g)


def test_retained_count_floors_at_one_token() -> None:
    assert compute_retained_tokens_count(num_tokens=100, q=0.999) == 1
    assert compute_retained_tokens_count(num_tokens=100, q=0.0) == 100
    assert compute_retained_tokens_count(num_tokens=7, q=0.0) == 7


@pytest.mark.parametrize("q", [0.25, 0.5, 0.75, 0.9])
@pytest.mark.parametrize("num_tokens", [40, 200, 1024])
def test_mask_shape_dtype_and_total(q: float, num_tokens: int) -> None:
    """Mask total must equal the placeholder-sizing helper."""
    embeds = _fake_image_embeds(num_tokens)
    mask = compute_retention_mask_rate(embeds, q=q)
    expected = compute_retained_tokens_count(num_tokens=num_tokens, q=q)
    assert mask.dtype == torch.bool
    assert mask.shape == (num_tokens,)
    assert int(mask.sum().item()) == expected


@pytest.mark.parametrize("q", [0.0, 0.5])
def test_small_image_falls_back_to_prefix_retention(q: float) -> None:
    """Below `pivot_block_size` RATE retains a deterministic prefix."""
    embeds = _fake_image_embeds(10)
    mask = compute_retention_mask_rate(embeds, q=q)
    expected = compute_retained_tokens_count(num_tokens=10, q=q)
    assert mask.tolist() == [True] * expected + [False] * (10 - expected)


def test_short_image_no_nan_scores() -> None:
    """Images with fewer than 1/skip_ratio tokens must not produce NaNs."""
    embeds = _fake_image_embeds(50)
    mask = compute_retention_mask_rate(embeds, q=0.9)
    expected = compute_retained_tokens_count(num_tokens=50, q=0.9)
    assert not torch.isnan(embeds).any()
    assert int(mask.sum().item()) == expected


def test_identical_features_degenerate_case() -> None:
    """No score spread: fall back to deterministic prefix retention."""
    embeds = torch.ones(64, 16)
    mask = compute_retention_mask_rate(embeds, q=0.5)
    expected = compute_retained_tokens_count(num_tokens=64, q=0.5)
    assert int(mask.sum().item()) == expected
    assert mask.tolist() == [True] * expected + [False] * (64 - expected)


def test_deterministic_given_same_projections() -> None:
    embeds = _fake_image_embeds(200)
    torch.manual_seed(42)
    first = compute_retention_mask_rate(embeds, q=0.5)
    _get_random_projections.cache_clear()
    torch.manual_seed(42)
    second = compute_retention_mask_rate(embeds, q=0.5)
    assert torch.equal(first, second)


def test_central_tokens_evicted_first() -> None:
    """Tokens near the feature centroid are evicted before diverse ones."""
    torch.manual_seed(0)
    hidden = 64
    diverse = torch.randn(60, hidden)
    center = torch.nn.functional.normalize(diverse.mean(0, keepdim=True), dim=-1)
    central = center.expand(40, hidden) + 0.001 * torch.randn(40, hidden)
    embeds = torch.cat([diverse, central], dim=0)
    mask = compute_retention_mask_rate(embeds, q=0.4)
    assert int((~mask[60:]).sum().item()) == 40
    assert int(mask[:60].sum().item()) == 60


def test_empty_input_safe() -> None:
    embeds = torch.zeros(0, 32)
    mask = compute_retention_mask_rate(embeds, q=0.25)
    assert mask.numel() == 0
    assert mask.dtype == torch.bool
