# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MiniMax-M3 VL ``max_long_side_pixel`` resize support.

These exercise the vendored processor directly (no checkpoint / GPU needed), so
they validate the long-side resize spec and the resulting prompt-token counts
deterministically.
"""

import pytest
import torch

from vllm.transformers_utils.processors.minimax_m3 import (
    IMAGE_MAX_TOTAL_PIXELS,
    MIN_SHORT_SIDE_PIXEL,
    VIDEO_MAX_TOTAL_PIXELS,
    MiniMaxM3VLImageProcessor,
    MiniMaxM3VLVideoProcessor,
    smart_resize,
)

# Long sides are multiples of patch_size*merge_size (28) so the rounding is
# exact and the expected token counts are unambiguous.
LONG_SIDES = [252, 504, 1008]
MERGE2 = 2**2  # merge_size ** 2


def _image_tokens(grid_thw) -> int:
    g = list(grid_thw)
    return int(g[0] * g[1] * g[2]) // MERGE2


# --------------------------------------------------------------------------- #
# smart_resize: the long-side spec (a) shrink / (b) enlarge / (c) hard cap
# --------------------------------------------------------------------------- #
def test_smart_resize_long_side_shrink():
    # (a) long side exceeds the cap -> shrink so the long side equals the cap.
    h, w = smart_resize(
        2048, 1024, factor=28, max_long_side_pixel=1008, max_total_pixels=10**9
    )
    assert max(h, w) == 1008
    assert (h, w) == (1008, 504)  # aspect ratio preserved


def test_smart_resize_short_side_enlarge():
    # (b) long side within the cap but short side below the floor -> enlarge so
    # the short side reaches min_short_side_pixel.
    h, w = smart_resize(
        200, 40, factor=28, max_long_side_pixel=1008, max_total_pixels=10**9
    )
    assert min(h, w) == MIN_SHORT_SIDE_PIXEL  # 112


def test_smart_resize_total_pixels_raises():
    # (c) still over the area cap after resizing -> raise instead of inferring.
    with pytest.raises(ValueError, match="max_total_pixels"):
        smart_resize(
            5000,
            5000,
            factor=28,
            max_long_side_pixel=4000,
            max_total_pixels=IMAGE_MAX_TOTAL_PIXELS,
        )


def test_smart_resize_backward_compatible_area_bound():
    # Without max_long_side_pixel the original Qwen-style area bound is used.
    assert smart_resize(2048, 2048, factor=28, max_pixels=451584) == (672, 672)


# --------------------------------------------------------------------------- #
# Image processor: monotonic prompt-token counts for 252 < 504 < 1008
# --------------------------------------------------------------------------- #
def test_image_tokens_increase_with_max_long_side_pixel():
    proc = MiniMaxM3VLImageProcessor()
    counts = []
    for long_side in LONG_SIDES:
        patches = proc.get_number_of_image_patches(
            2048, 2048, images_kwargs={"max_long_side_pixel": long_side}
        )
        counts.append(patches // MERGE2)

    assert counts == [81, 324, 1296]
    assert counts[0] < counts[1] < counts[2]


def test_image_processor_defaults_match_spec():
    proc = MiniMaxM3VLImageProcessor()
    assert proc.max_long_side_pixel is None  # opt-in
    assert proc.min_short_side_pixel == MIN_SHORT_SIDE_PIXEL
    assert proc.max_total_pixels == IMAGE_MAX_TOTAL_PIXELS


def test_image_preprocess_pipeline_monotonic():
    proc = MiniMaxM3VLImageProcessor()
    image = torch.randint(0, 255, (3, 2048, 2048), dtype=torch.uint8)
    counts = []
    for long_side in LONG_SIDES:
        out = proc.preprocess(
            [image],
            do_resize=True,
            max_long_side_pixel=long_side,
            return_tensors="pt",
        )
        counts.append(_image_tokens(out["image_grid_thw"][0]))
    assert counts == [81, 324, 1296]


# --------------------------------------------------------------------------- #
# Video processor: same monotonic behavior + volumetric (w*h*frames) cap
# --------------------------------------------------------------------------- #
def test_video_tokens_increase_with_max_long_side_pixel():
    proc = MiniMaxM3VLVideoProcessor()
    assert proc.max_total_pixels == VIDEO_MAX_TOTAL_PIXELS
    video = torch.randint(0, 255, (4, 3, 2048, 2048), dtype=torch.uint8)
    counts = []
    for long_side in LONG_SIDES:
        out = proc.preprocess(
            videos=[video],
            do_resize=True,
            max_long_side_pixel=long_side,
            return_tensors="pt",
        )
        counts.append(_image_tokens(out["video_grid_thw"][0]))
    assert counts[0] < counts[1] < counts[2]


def test_video_volumetric_cap_raises():
    proc = MiniMaxM3VLVideoProcessor()
    # 400 frames at a 1008-long-side square: 1008*1008*400 >> 301,056,000.
    video = torch.randint(0, 255, (400, 3, 2048, 2048), dtype=torch.uint8)
    with pytest.raises(ValueError, match="max_total_pixels"):
        proc.preprocess(
            videos=[video],
            do_resize=True,
            max_long_side_pixel=1008,
            return_tensors="pt",
        )
