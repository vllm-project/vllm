# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MiniMax-M3 VL preprocessing (sglang-aligned size spec).

These exercise the vendored processor directly (no checkpoint / GPU needed) and
pin the ``compute_aligned_size`` / ``resolve_max_long_side_pixel`` contract
against the sglang reference algorithm
(``python/sglang/srt/multimodal/processors/minimax_m3_vl.py``).
"""

import pytest
import torch

from vllm.transformers_utils.processors.minimax_m3 import (
    MIN_SHORT_SIDE_PIXEL,
    MiniMaxM3VLImageProcessor,
    MiniMaxM3VLVideoProcessor,
    compute_aligned_size,
    resolve_max_long_side_pixel,
)

FACTOR = 28  # patch_size * merge_size
PATCH_SIZE = 14
MERGE2 = 2**2  # merge_size ** 2

# detail -> max_long_side_pixel, per the size spec.
IMAGE_DETAIL_CAPS = {"low": 672, "auto": 2016, "high": 3584}
VIDEO_DETAIL_CAPS = {"low": 504, "auto": 672, "high": 2188}


def _merged_tokens(grid_thw) -> int:
    return int(grid_thw.prod()) // MERGE2


# --------------------------------------------------------------------------- #
# compute_aligned_size: floor layers / ceil-align / long-side cap
# --------------------------------------------------------------------------- #
def test_compute_aligned_size_enlarges_short_side():
    # (200, 40) hits both floors: area 8000 < 168**2 scales to (376, 76); the
    # short-side floor then scales by 168/76, where ceil(76 * 168/76) = 169
    # (float overshoot) aligns up to 196, and 832 aligns up to 840.
    h, w = compute_aligned_size(200, 40, FACTOR, max_long_side_pixel=1008)
    assert (h, w) == (840, 196)
    assert h % FACTOR == 0 and w % FACTOR == 0


def test_compute_aligned_size_shrinks_over_cap_isotropically():
    # Long side 2048 ceil-aligns to 2072 > cap -> integer aspect-ratio
    # scale-back to (1008, 504), already factor-aligned.
    h, w = compute_aligned_size(2048, 1024, FACTOR, max_long_side_pixel=1008)
    assert (h, w) == (1008, 504)


def test_compute_aligned_size_rejects_extreme_aspect_ratio():
    with pytest.raises(ValueError, match="aspect ratio"):
        compute_aligned_size(20001, 100, FACTOR, max_long_side_pixel=1008)


# --------------------------------------------------------------------------- #
# resolve_max_long_side_pixel: detail tiers and explicit override
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("detail", "expected"),
    [("low", 672), ("auto", 2016), ("high", 3584), (None, 2016)],
)
def test_resolve_max_long_side_pixel_image(detail, expected):
    assert resolve_max_long_side_pixel(detail, None, modality="image") == expected


@pytest.mark.parametrize(
    ("detail", "expected"),
    [("low", 504), ("auto", 672), ("high", 2188), (None, 672)],
)
def test_resolve_max_long_side_pixel_video(detail, expected):
    assert resolve_max_long_side_pixel(detail, None, modality="video") == expected


@pytest.mark.parametrize("modality", ["image", "video"])
def test_resolve_max_long_side_pixel_explicit_overrides_detail(modality):
    assert resolve_max_long_side_pixel("high", 1024, modality=modality) == 1024
    assert resolve_max_long_side_pixel(None, 1024, modality=modality) == 1024


def test_processor_default_caps():
    assert MIN_SHORT_SIDE_PIXEL == 168
    assert MiniMaxM3VLImageProcessor().max_long_side_pixel == 2016
    assert MiniMaxM3VLVideoProcessor().max_long_side_pixel == 672


# --------------------------------------------------------------------------- #
# Image processor end to end: grid/patch-counter consistency and token counts
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def square_image() -> torch.Tensor:
    # 4096x4096 so detail="high" actually reaches the 3584 cap (a 2048x2048
    # image only ceil-aligns to 2072 and never upscales).
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 255, (3, 4096, 4096), dtype=torch.uint8, generator=g)


@pytest.mark.parametrize("detail", ["low", "auto", "high"])
def test_image_preprocess_grid_matches_patch_counter(detail, square_image):
    proc = MiniMaxM3VLImageProcessor()
    cap = IMAGE_DETAIL_CAPS[detail]

    out = proc.preprocess([square_image], detail=detail, return_tensors="pt")
    grid_thw = out["image_grid_thw"][0]

    patches = proc.get_number_of_image_patches(
        4096, 4096, images_kwargs={"max_long_side_pixel": cap}
    )
    assert int(grid_thw.prod()) == patches

    # Cross-check against compute_aligned_size: the square long side lands
    # exactly on the cap, giving (cap / patch_size) ** 2 patches.
    assert compute_aligned_size(4096, 4096, FACTOR, max_long_side_pixel=cap) == (
        cap,
        cap,
    )
    assert patches == (cap // PATCH_SIZE) ** 2


def test_image_tokens_increase_with_detail(square_image):
    proc = MiniMaxM3VLImageProcessor()
    counts = [
        _merged_tokens(
            proc.preprocess([square_image], detail=detail, return_tensors="pt")[
                "image_grid_thw"
            ][0]
        )
        for detail in ("low", "auto", "high")
    ]
    # (cap / patch_size) ** 2 / merge_size**2 for caps 672 / 2016 / 3584.
    assert counts == [576, 5184, 16384]
    assert counts[0] < counts[1] < counts[2]


def test_image_preprocess_explicit_cap_overrides_detail(square_image):
    proc = MiniMaxM3VLImageProcessor()
    out = proc.preprocess(
        [square_image],
        detail="high",
        max_long_side_pixel=672,
        return_tensors="pt",
    )
    # The explicit cap wins over detail="high" -> same grid as detail="low".
    assert _merged_tokens(out["image_grid_thw"][0]) == 576


# --------------------------------------------------------------------------- #
# Video processor: volumetric (frames * h * w) cap raises unconditionally
# --------------------------------------------------------------------------- #
def test_video_volume_cap_raises_by_default():
    proc = MiniMaxM3VLVideoProcessor()
    # Default cap 672: 700 frames * 672 * 672 = 316,108,800 > 301,056,000.
    # The volumetric cap raises even without any detail/max_long_side_pixel
    # override.
    video = torch.randint(0, 255, (700, 3, 672, 672), dtype=torch.uint8)
    with pytest.raises(ValueError, match="max_total_pixels"):
        proc.preprocess(videos=[video], return_tensors="pt")
