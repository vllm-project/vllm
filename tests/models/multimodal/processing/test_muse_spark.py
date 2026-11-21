# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark Open preprocessing of the input types the parser passes through."""

import numpy as np
import pytest
import torch
from PIL import Image

from vllm.models.muse_spark.common.processor import (
    PreprocessConfig,
    _preprocess_single_video,
    preprocess_image,
)

pytestmark = pytest.mark.skip_global_cleanup


def test_preprocess_numpy_image_matches_pil():
    image = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    config = PreprocessConfig()

    expected, expected_tokens = preprocess_image(Image.fromarray(image), config)
    actual, actual_tokens = preprocess_image(image, config)

    torch.testing.assert_close(actual, expected)
    assert actual_tokens == expected_tokens


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_preprocess_pil_frame_video_matches_numpy(mode):
    pixels = np.random.default_rng(0).integers(0, 256, (8, 224, 224, 3), dtype=np.uint8)
    frames = [Image.fromarray(frame).convert(mode) for frame in pixels]
    metadata = {"fps": 2.0, "frames_indices": list(range(8)), "total_num_frames": 8}
    config = PreprocessConfig()

    rgb_frames = np.stack([np.asarray(frame.convert("RGB")) for frame in frames])
    expected = _preprocess_single_video((rgb_frames, metadata), config)
    actual = _preprocess_single_video((frames, metadata), config)

    torch.testing.assert_close(actual[0], expected[0])
    assert actual[1:] == expected[1:]
