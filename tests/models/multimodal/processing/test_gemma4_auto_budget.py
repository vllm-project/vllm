# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ``_pick_auto_soft_tokens``.

Exercises the adaptive budget picker for Gemma 4's vision tower over
boundary cases, aspect ratios, and out-of-range inputs.  The resize math
is derived from transformers'
``models/gemma4/image_processing_pil_gemma4.get_aspect_ratio_preserving_size``
which sizes an image so it fits in
``max_soft_tokens * patch_size**2 * pooling_kernel_size**2`` pixels while
preserving aspect ratio.  For the default Gemma 4 config
(``patch_size=16``, ``pooling_kernel_size=3``) that is
``max_soft_tokens * 2304`` pixels per budget.
"""

import pytest

from vllm.model_executor.models.gemma4_mm import (
    _SUPPORTED_SOFT_TOKENS,
    _pick_auto_soft_tokens,
    _resolve_auto_over_images,
)

# Default Gemma 4 geometry.
PATCH = 16
POOL = 3
# Pixels of image area each ``max_soft_tokens`` slot can represent at the
# native resize target: patch_size**2 * pooling_kernel_size**2.
PX_PER_TOKEN = PATCH**2 * POOL**2  # 2304


def _target_area(budget: int) -> int:
    return budget * PX_PER_TOKEN


@pytest.mark.parametrize(
    "width,height,expected",
    [
        # Each boundary is area == budget * 2304 exactly (square).
        # At the boundary the image fits in that budget's target area;
        # one pixel over flips to the next bucket.
        (401, 401, 70),  # 160,801 <= 161,280
        (402, 402, 140),  # 161,604 >  161,280
        (568, 568, 280),  # 322,624 >  322,560
        (803, 803, 280),  # 644,809 <= 645,120
        (804, 804, 560),  # 646,416 >  645,120
        (1135, 1135, 560),  # 1,288,225 <= 1,290,240
        (1136, 1136, 1120),  # 1,290,496 >  1,290,240
        (1606, 1606, 1120),  # 2,579,236 <= 2,580,480
        (1607, 1607, 1120),  # clamp: no larger budget exists
    ],
)
def test_square_boundaries(width, height, expected):
    assert _pick_auto_soft_tokens(width, height, PATCH, POOL) == expected


@pytest.mark.parametrize(
    "width,height,expected",
    [
        # Common aspect ratios. Only the area matters; the picker is
        # aspect-ratio agnostic because the native resize preserves AR.
        (100, 75, 70),  # thumbnail
        (200, 150, 70),
        (640, 480, 140),  # VGA: 307,200 > 161,280
        (800, 600, 280),  # 480,000 > 322,560
        (1024, 768, 560),  # 786,432 > 645,120
        (1280, 720, 560),  # 921,600 > 645,120
        (1920, 1080, 1120),  # Full HD: 2,073,600 > 1,290,240
        (3840, 2160, 1120),  # 4K: 8,294,400 (clamped to largest budget)
        (4032, 3024, 1120),  # 12 MP phone photo (clamped)
    ],
)
def test_common_aspect_ratios(width, height, expected):
    assert _pick_auto_soft_tokens(width, height, PATCH, POOL) == expected


@pytest.mark.parametrize("budget", _SUPPORTED_SOFT_TOKENS)
def test_exact_target_area_fits(budget):
    """An image whose total area exactly equals the budget's target area
    must fit in that budget (factor == 1.0, no downscale)."""
    area = _target_area(budget)
    # Use a 2:1 aspect ratio so we also exercise the non-square path.
    # w * h == area, w = 2h  =>  h = sqrt(area/2), w = 2h.
    h = int((area / 2) ** 0.5)
    w = area // h
    assert _pick_auto_soft_tokens(w, h, PATCH, POOL) <= budget


def test_clamp_on_oversized_image():
    """Images larger than the largest budget's target area clamp rather
    than error.  Preserves the current fixed-budget behavior for those
    inputs."""
    huge_area = _target_area(_SUPPORTED_SOFT_TOKENS[-1]) * 10
    # Any w, h with this area — pick 1:1 for simplicity.
    side = int(huge_area**0.5) + 1
    assert _pick_auto_soft_tokens(side, side, PATCH, POOL) == _SUPPORTED_SOFT_TOKENS[-1]


def test_tiny_image_picks_smallest_budget():
    """A 1x1 image always lands in the smallest budget."""
    assert _pick_auto_soft_tokens(1, 1, PATCH, POOL) == _SUPPORTED_SOFT_TOKENS[0]


def test_monotonic_in_area():
    """Picker must be monotonically non-decreasing in image area."""
    prev = 0
    for side in range(1, 2000, 7):
        budget = _pick_auto_soft_tokens(side, side, PATCH, POOL)
        assert budget >= prev
        prev = budget


@pytest.mark.parametrize(
    "patch_size,pool_size",
    [(8, 2), (16, 3), (32, 4)],  # default + two hypothetical configs
)
def test_picker_uses_config_values(patch_size, pool_size):
    """The formula ``pixels_per_token = patch_size**2 * pool_size**2`` is
    exercised via the configurable arguments rather than hardcoded 2304."""
    px_per_tok = (patch_size**2) * (pool_size**2)
    # An image sized exactly for the smallest budget must land on it.
    smallest = _SUPPORTED_SOFT_TOKENS[0]
    area = smallest * px_per_tok
    side = int(area**0.5)
    assert _pick_auto_soft_tokens(side, side, patch_size, pool_size) <= smallest


# ---------------------------------------------------------------------------
# _resolve_auto_over_images — the multi-image request-level resolver
# ---------------------------------------------------------------------------


def test_resolve_empty_list_returns_smallest_budget():
    """With no images, return the smallest supported budget so the HF
    processor still receives a valid numeric value."""
    assert _resolve_auto_over_images([], PATCH, POOL) == _SUPPORTED_SOFT_TOKENS[0]


def test_resolve_single_image_matches_picker():
    """Single image: the resolver is equivalent to _pick_auto_soft_tokens."""
    cases = [(100, 75), (800, 600), (1920, 1080)]
    for w, h in cases:
        single = _resolve_auto_over_images([(w, h)], PATCH, POOL)
        direct = _pick_auto_soft_tokens(w, h, PATCH, POOL)
        assert single == direct


def test_resolve_multi_image_takes_max():
    """Multi-image requests must collapse to a single budget — the max
    across all images — so every image in the prompt shares one budget
    (Gemma 4 requires this; per-image budgets would mismatch the vision
    tower's output count and crash embedding merge)."""
    # Mix tiny + large: result must match the large image's budget.
    sizes = [(100, 75), (1920, 1080), (200, 150)]
    resolved = _resolve_auto_over_images(sizes, PATCH, POOL)
    expected = max(_pick_auto_soft_tokens(w, h, PATCH, POOL) for w, h in sizes)
    assert resolved == expected
    # Concrete: 1920x1080 (area 2,073,600) lands in the 1120 bucket.
    assert resolved == 1120


def test_resolve_all_small_stays_small():
    """A batch of tiny images stays in the smallest budget — no spurious
    promotion to a larger budget."""
    sizes = [(100, 75), (200, 150), (50, 50)]
    assert _resolve_auto_over_images(sizes, PATCH, POOL) == 70
