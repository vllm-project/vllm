# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 image preprocessing and placeholder expansion."""

import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vllm.config.multimodal import MultiModalConfig
from vllm.models.inkling2.processing import (
    IMAGE_MARKER_ID,
    IMAGE_PAD_ID,
    Inkling2DummyInputsBuilder,
    Inkling2ImageProcessor,
    Inkling2MultiModalProcessor,
    Inkling2ProcessingInfo,
)
from vllm.multimodal.media import MediaWithBytes
from vllm.multimodal.processing import (
    InputProcessingContext,
    ProcessorInputs,
    TimingContext,
)


def _raw(pixels) -> torch.Tensor:
    return torch.from_numpy(np.asarray(pixels, dtype=np.float32)).bfloat16()


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def processor() -> Inkling2MultiModalProcessor:
    mm_config = MultiModalConfig()
    model_config = SimpleNamespace(
        model="inkling2",
        multimodal_config=mm_config,
        get_multimodal_config=lambda: mm_config,
    )
    info = Inkling2ProcessingInfo(InputProcessingContext(model_config, None))
    return Inkling2MultiModalProcessor(info, Inkling2DummyInputsBuilder(info))


def test_grid_covers_image_without_extra_column():
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, (39, 41, 3), dtype=np.uint8)
    out = Inkling2ImageProcessor().preprocess([Image.fromarray(pixels)])

    assert out["vision_grid_thw"].tolist() == [[1, 1, 2]]
    patches = out["vision_pixels"]
    assert patches.dtype == torch.bfloat16
    assert patches.shape == (2, 40, 40, 3)
    # Row-major cells of raw values; pixels beyond the image are -1.
    assert torch.equal(patches[0, :39], _raw(pixels[:, :40]))
    assert torch.equal(patches[1, :39, :1], _raw(pixels[:, 40:]))
    assert (patches[1, :, 1:] == -1).all()
    assert (patches[:, 39] == -1).all()


def test_transparency_composited_on_checkerboard():
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[0, 0] = (10, 20, 30, 255)
    image = Image.fromarray(rgba, "RGBA")
    expected = np.where(
        (np.arange(16)[None, :] // 8 + np.arange(16)[:, None] // 8) % 2 == 0, 204, 153
    )
    expected = np.repeat(expected[:, :, None], 3, axis=2)
    expected[0, 0] = (10, 20, 30)

    flattened = MediaWithBytes(image.convert("RGB"), _png(image))
    processor = Inkling2ImageProcessor()
    for image_input in (image, _png(image), flattened):
        patches = processor.preprocess([image_input])["vision_pixels"]
        assert torch.equal(patches[0, :16, :16], _raw(expected))


def test_placeholder_expansion(processor):
    images = [Image.new("RGB", (100, 50)), Image.new("RGBA", (40, 40))]
    prompt = [200000, IMAGE_MARKER_ID, 200010, 200000, IMAGE_MARKER_ID, 200010]
    inputs = ProcessorInputs(
        prompt=prompt, mm_data_items=processor.info.parse_mm_data({"image": images})
    )
    out = processor.apply(inputs, TimingContext(enabled=False))

    assert out["prompt_token_ids"] == (
        [200000, IMAGE_MARKER_ID]
        + [IMAGE_PAD_ID] * 6
        + [200010, 200000, IMAGE_MARKER_ID, IMAGE_PAD_ID, 200010]
    )
    ranges = out["mm_placeholders"]["image"]
    assert [(r.offset, r.length) for r in ranges] == [(1, 7), (10, 2)]
    assert [int(r.is_embed.sum()) for r in ranges] == [6, 1]
    data = out["mm_kwargs"].get_data()
    assert data["vision_pixels"].shape == (7, 40, 40, 3)
    assert data["vision_grid_thw"].tolist() == [[1, 2, 3], [1, 1, 1]]
