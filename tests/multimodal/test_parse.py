# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch
from PIL import Image

from vllm.multimodal.parse import (
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


def test_video_with_metadata_cpu_tensor_to_numpy():
    """CPU tensor frames keep the existing conversion to numpy."""
    frames = torch.zeros((4, H, W, 3), dtype=torch.uint8)
    video, metadata = MultiModalDataParser()._get_video_with_metadata(frames)

    assert isinstance(video, np.ndarray)
    np.testing.assert_array_equal(video, frames.numpy())
    assert metadata is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_video_with_metadata_keeps_device_tensor():
    """Device-resident frames (e.g. NVDEC-decoded) pass through as tensors,
    so a device-side HF processor can consume them without a D2H copy."""
    frames = torch.zeros((4, H, W, 3), dtype=torch.uint8, device="cuda")
    video, metadata = MultiModalDataParser()._get_video_with_metadata(frames)

    assert video is frames
    assert metadata is None
