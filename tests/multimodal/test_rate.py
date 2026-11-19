# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.multimodal.image_prune.rate import (
    _get_diversity_scores,
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
    """Selection must not depend on global RNG state (restart-stable)."""
    embeds = _fake_image_embeds(200)
    torch.manual_seed(1)
    _get_random_projections.cache_clear()
    first = compute_retention_mask_rate(embeds, q=0.5)
    torch.manual_seed(999)
    _get_random_projections.cache_clear()
    second = compute_retention_mask_rate(embeds, q=0.5)
    assert torch.equal(first, second)


@pytest.mark.parametrize("q", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("num_tokens", [200, 784])
def test_retained_selection_is_not_positional(q: float, num_tokens: int) -> None:
    """On generic inputs RATE must score tokens, not retain a prefix.

    Guards against regressions where the pivot-selection path silently
    degenerates (e.g. all-zero scores), which turns the stable argsort
    into positional prefix retention.
    """
    embeds = _fake_image_embeds(num_tokens)
    mask = compute_retention_mask_rate(embeds, q=q)
    retained = int(mask.sum().item())
    assert retained == compute_retained_tokens_count(num_tokens=num_tokens, q=q)
    assert not mask[:retained].all(), "RATE retained a positional prefix"


def test_pivot_scores_have_spread() -> None:
    """The pivot path yields non-constant diversity scores for real inputs."""
    embeds = _fake_image_embeds(784)
    scores = _get_diversity_scores(
        embeds,
        pivot_block_size=32,
        skip_num=max(1, int(0.01 * 784)),
        num_proj=8,
    )
    assert scores.numel() == 784
    assert float(scores.std().item()) > 0.0


def test_retained_tokens_are_lowest_scored() -> None:
    """The mask must select exactly the lowest-scoring tokens."""
    embeds = _fake_image_embeds(784)
    q = 0.5
    skip_num = max(1, int(0.01 * 784))
    scores = _get_diversity_scores(
        embeds, pivot_block_size=32, skip_num=skip_num, num_proj=8
    )
    retain = compute_retained_tokens_count(num_tokens=784, q=q)
    expected = torch.zeros(784, dtype=torch.bool)
    expected[torch.argsort(scores, stable=True)[:retain]] = True
    assert torch.equal(compute_retention_mask_rate(embeds, q=q), expected)


def test_empty_input_safe() -> None:
    embeds = torch.zeros(0, 32)
    mask = compute_retention_mask_rate(embeds, q=0.25)
    assert mask.numel() == 0
    assert mask.dtype == torch.bool
