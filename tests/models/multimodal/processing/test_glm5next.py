# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash requires Rust preprocessing and profiles synthetic patches."""

import numpy as np
import pytest
import torch
from PIL import Image

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import ProcessorInputs, TimingContext

from ...utils import build_model_context


def test_python_multimodal_initialization_requires_rust(monkeypatch):
    """Processor registration must not fall back to the GLM-4V processor."""
    monkeypatch.setenv("VLLM_USE_RUST_FRONTEND", "0")
    ctx = build_model_context("zai-org/GLM-5.3-Flash")
    with pytest.raises(ValueError, match="requires the Rust frontend"):
        MULTIMODAL_REGISTRY.create_processor(ctx.model_config)


@pytest.mark.parametrize("modality", ["image", "video"])
def test_rust_engine_processor_rejects_python_media(monkeypatch, modality):
    """The engine's profiling processor cannot preprocess real requests."""
    monkeypatch.setenv("VLLM_USE_RUST_FRONTEND", "1")
    ctx = build_model_context("zai-org/GLM-5.3-Flash")
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    media = (
        Image.new("RGB", (28, 28))
        if modality == "image"
        else (
            np.zeros((2, 28, 28, 3), dtype=np.uint8),
            {
                "fps": 1.0,
                "total_num_frames": 2,
                "do_sample_frames": False,
                "frames_indices": [0, 1],
            },
        )
    )
    inputs = ProcessorInputs(
        prompt=[ctx.model_config.hf_config.image_token_id],
        mm_data_items=processor.info.parse_mm_data({modality: media}),
    )
    with pytest.raises(ValueError, match="requires the Rust frontend"):
        processor.apply(inputs, TimingContext(enabled=False))


@pytest.mark.parametrize("modality,count", [("image", 2), ("video", 1)])
def test_rust_engine_profiles_synthetic_patches(monkeypatch, modality, count):
    """Synthetic patch rows and placeholder lengths agree without HF media IO."""
    monkeypatch.setenv("VLLM_USE_RUST_FRONTEND", "1")
    ctx = build_model_context(
        "zai-org/GLM-5.3-Flash",
        model_config_kwargs={"max_model_len": 256},
        limit_mm_per_prompt={modality: count},
        mm_processor_kwargs={"max_pixels": 1568 * 100},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    inputs = processor.get_dummy_mm_inputs({modality: count})
    items = inputs["mm_kwargs"][modality]
    assert len(items) == count
    vision_config = ctx.model_config.hf_config.vision_config
    pixel_key = "pixel_values" if modality == "image" else "pixel_values_videos"
    for item, placeholder in zip(items, inputs["mm_placeholders"][modality]):
        assert item is not None
        data = item.get_data()
        grid = data[f"{modality}_grid_thw"]
        pixels = data[pixel_key]
        assert isinstance(grid, torch.Tensor)
        assert isinstance(pixels, torch.Tensor)
        num_patches = int(grid.prod())
        assert pixels.shape == (
            num_patches,
            vision_config.in_channels
            * vision_config.temporal_patch_size
            * vision_config.patch_size**2,
        )
        assert pixels.dtype == torch.float32
        assert torch.count_nonzero(pixels) == 0
        assert placeholder.length == num_patches // vision_config.spatial_merge_size**2
        assert placeholder.offset + placeholder.length <= len(
            inputs["prompt_token_ids"]
        )


def _image_info(**kwargs):
    ctx = build_model_context(
        "zai-org/GLM-5.3-Flash",
        limit_mm_per_prompt={"image": 1},
        **kwargs,
    )
    return MULTIMODAL_REGISTRY.get_processing_info(ctx.model_config)


def test_image_encoder_cache_covers_full_token_budget():
    """The most-features probe must reach the processor's token ceiling.

    The inherited square probe refits to 89x89 = 7921 tokens under the
    max_image_tokens=8000 budget, so the encoder cache came up short and
    ordinary non-square images in 7922-8000 tokens were refused with
    HTTP 400 (#59539).
    """
    info = _image_info()
    assert info.get_max_image_tokens() == 8000

    # The shapes from the issue, with the token counts the processor
    # actually produces; every one must fit the cache.
    for width, height, expected_tokens in [
        (4032, 3024, 7931),  # phone photo, 4:3
        (3840, 2160, 7973),  # 4K frame, 16:9
        (3508, 2480, 7950),  # A4 at 300 dpi
        (2600, 2400, 7998),  # 13:12
        (3000, 3000, 7921),  # square worst case before the fix
    ]:
        num_tokens = info.get_num_image_tokens(image_width=width, image_height=height)
        assert num_tokens == expected_tokens
        assert num_tokens <= info.get_max_image_tokens()

    # The profiling dummy covers the real worst case, not the square one.
    size = info.get_image_size_with_most_features()
    assert size.width * size.height == 2240 * 2800


def test_image_encoder_cache_follows_max_pixels_override():
    info = _image_info(mm_processor_kwargs={"max_pixels": 1568 * 100})
    assert info.get_max_image_tokens() == 100
    size = info.get_image_size_with_most_features()
    assert (
        info.get_num_image_tokens(image_width=size.width, image_height=size.height)
        == 100
    )
