# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Step3-VL ImagePatcher sizing order."""

from unittest.mock import patch

import pytest
from PIL import Image

from vllm.transformers_utils.processors.step3_vl import MAX_IMAGE_SIZE, ImagePatcher


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (60000, 30),
        (30, 60000),
    ],
)
def test_image_patcher_clamps_before_square_pad(width: int, height: int):
    """Extreme-aspect images must not allocate a pad canvas beyond MAX_IMAGE_SIZE."""
    patcher = ImagePatcher(enable_patch=False)
    img = Image.new("RGB", (width, height))

    created_sizes: list[tuple[int, int]] = []
    real_new = Image.new

    def tracking_new(mode, size, color=0):
        created_sizes.append(size)
        return real_new(mode, size, color)

    with patch.object(Image, "new", side_effect=tracking_new):
        out_img, patches, newlines = patcher(img)

    assert patches == []
    assert newlines == []
    assert max(out_img.size) <= MAX_IMAGE_SIZE
    assert all(max(size) <= MAX_IMAGE_SIZE for size in created_sizes)


def test_image_patcher_square_pad_still_runs_for_skinny_clamped_images():
    """After clamping, extreme-aspect images still receive square padding."""
    patcher = ImagePatcher(enable_patch=False)
    # Long edge already at MAX_IMAGE_SIZE; short edge < 32 triggers padding.
    img = Image.new("RGB", (MAX_IMAGE_SIZE, 16))
    out_img, _, _ = patcher(img)
    assert out_img.size == (MAX_IMAGE_SIZE, MAX_IMAGE_SIZE)


def test_get_num_patches_handles_extreme_aspect_ratio():
    """Token budgeting must remain finite for extreme-aspect inputs."""
    patcher = ImagePatcher()
    num_patches, num_newlines = patcher.get_num_patches(60000, 30)
    assert num_patches >= 0
    assert num_newlines >= 0
