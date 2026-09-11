# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch
from PIL import Image

from vllm.multimodal.parse import (
    AudioProcessorItems,
    ImageProcessorItems,
    MultiModalDataParser,
    VideoProcessorItems,
)

H, W = 480, 640


@pytest.mark.parametrize(
    "image",
    [
        Image.new("RGB", (W, H)),
        # HWC, e.g. from np.array(PIL.Image)
        np.zeros((H, W, 3), dtype=np.uint8),
        torch.zeros((H, W, 3), dtype=torch.uint8),
        # CHW, standard PyTorch / numpy convention
        np.zeros((3, H, W), dtype=np.uint8),
        torch.zeros((3, H, W), dtype=torch.uint8),
    ],
)
def test_image_size_hwc_chw(image):
    """Image sizes must be channel-layout agnostic.

    `get_image_size` determines the multimodal placeholder count; reading an
    HWC array (the layout `np.array(PIL.Image)` produces) as CHW yields a
    bogus size and a placeholder/embedding count mismatch at inference time.
    """
    items = ImageProcessorItems([image])

    assert items.get_image_size(0) == (W, H)


@pytest.mark.parametrize(
    "frame",
    [
        Image.new("RGB", (W, H)),
        np.zeros((H, W, 3), dtype=np.uint8),
        torch.zeros((H, W, 3), dtype=torch.uint8),
        np.zeros((3, H, W), dtype=np.uint8),
        torch.zeros((3, H, W), dtype=torch.uint8),
    ],
)
def test_frame_size_hwc_chw(frame):
    """`get_frame_size` must stay consistent with `get_image_size`."""
    items = VideoProcessorItems([[frame]])

    assert items.get_frame_size(0) == (W, H)


@pytest.mark.parametrize(
    "modality,processor_cls",
    [
        ("audio", AudioProcessorItems),
        ("image", ImageProcessorItems),
        ("video", VideoProcessorItems),
    ],
)
def test_parse_mm_data_accepts_none_cached_item(modality, processor_cls):
    mm_items = MultiModalDataParser().parse_mm_data({modality: [None]})
    items = mm_items[modality]
    assert isinstance(items, processor_cls)
    assert len(items) == 1
    assert items.get(0) is None


def test_cached_audio_items_preserve_positions_during_resampling():
    waveform = np.arange(16, dtype=np.float32)
    parser = MultiModalDataParser(
        target_sr=16000, target_channels=1, audio_resample_method="scipy"
    )
    items = parser.parse_mm_data(
        {"audio": [None, (waveform, 8000), None, (waveform, 16000)]}
    )["audio"]

    assert len(items) == 4
    assert items.get(0) is None
    assert items.get(2) is None
    assert len(items.get(1)) == 32
    np.testing.assert_array_equal(items.get(3), waveform)
