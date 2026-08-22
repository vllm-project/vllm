# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Mistral3's multimodal preprocessing kwargs."""

import pytest
import torch
from PIL import Image
from transformers import BatchFeature

from vllm.model_executor.models.lightonocr import LightOnOCRProcessingInfo
from vllm.model_executor.models.mistral3 import Mistral3HFEncoderInfo
from vllm.model_executor.models.pixtral import PixtralHFEncoderInfo
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItems

from ...utils import build_model_context

# This repo ships both params.json (Mistral) and config.json (HF). Auto config
# selects PixtralForConditionalGeneration; force HF to exercise Mistral3.
_MODEL_CONFIG_KWARGS = {"config_format": "hf"}
_MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
_LIGHTON_MODEL_ID = "lightonai/LightOnOCR-1B-1025"


def _processed_pixel_values(
    hf_processor,
    images: list[Image.Image],
    mm_processor_kwargs: dict[str, object],
) -> list[torch.Tensor]:
    """Resize via the HF image processor and un-pad to per-image H×W."""
    image_processor = hf_processor.image_processor
    hf_out = image_processor(images=images, return_tensors="pt", **mm_processor_kwargs)
    pixel_values = hf_out["pixel_values"]
    image_sizes = hf_out["image_sizes"]
    return [p[:, :h, :w] for p, (h, w) in zip(pixel_values, image_sizes)]


def _placeholder_count_from_prompt_updates(
    processor,
    images: list[Image.Image],
    pixel_values: list[torch.Tensor],
    mm_processor_kwargs: dict[str, object],
) -> int:
    hf_inputs = BatchFeature({"pixel_values": pixel_values})
    fields_config = processor._get_mm_fields_config(hf_inputs, mm_processor_kwargs)
    out_mm_kwargs = MultiModalKwargsItems.from_hf_inputs(hf_inputs, fields_config)
    # Prompt updates use raw PIL sizes and must predict the processed grid.
    mm_items = processor.info.parse_mm_data({"image": images})
    updates = processor._get_prompt_updates(
        mm_items, mm_processor_kwargs, out_mm_kwargs
    )
    image_token_id = processor.info.get_hf_config().image_token_index

    total = 0
    for item_idx in range(len(images)):
        details = updates[0].resolve(item_idx).content
        total += details.full.count(image_token_id)
    return total


def _expected_placeholder_tokens_per_image(
    hf_processor,
    pixel_values: torch.Tensor,
) -> int:
    """Count projected tokens from the actual HF-processed H×W."""
    image_h, image_w = pixel_values.shape[-2:]
    patch_size = hf_processor.image_processor.patch_size
    if isinstance(patch_size, dict):
        patch_h = patch_size["height"]
        patch_w = patch_size["width"]
    else:
        patch_h = patch_w = int(patch_size)

    spatial_merge_size = getattr(hf_processor, "spatial_merge_size", 1)
    assert image_h % patch_h == 0
    assert image_w % patch_w == 0

    return (image_h // (patch_h * spatial_merge_size)) * (
        image_w // (patch_w * spatial_merge_size)
    )


@pytest.mark.parametrize("model_id", [_MODEL_ID])
@pytest.mark.parametrize(
    ("mm_processor_kwargs", "image_size", "expected_toks_per_img"),
    [
        ({}, (448, 448), 256),
        ({"size": {"longest_edge": 1008}}, (1540, 1540), 1296),
        ({"size": {"longest_edge": 1288}}, (1536, 1187), 1656),
        ({"size": {"longest_edge": 1008}}, (29, 29), 1),
        ({"size": {"longest_edge": 1000}}, (1540, 1700), 1152),
    ],
)
@pytest.mark.parametrize("num_imgs", [1, 2])
@pytest.mark.parametrize("kwargs_on_init", [True, False])
def test_processor_size_override(
    model_id: str,
    mm_processor_kwargs: dict[str, object],
    image_size: tuple[int, int],
    expected_toks_per_img: int,
    num_imgs: int,
    kwargs_on_init: bool,
):
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": num_imgs},
        model_config_kwargs=_MODEL_CONFIG_KWARGS,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor_mm_kwargs = {} if kwargs_on_init else mm_processor_kwargs
    hf_processor = processor.info.get_hf_processor(**hf_processor_mm_kwargs)

    dummy_image = Image.new("RGB", image_size, color=(127, 127, 127))
    images = [dummy_image] * num_imgs
    merged_mm_kwargs = processor.info.ctx.get_merged_mm_kwargs(hf_processor_mm_kwargs)
    pixel_values = _processed_pixel_values(hf_processor, images, merged_mm_kwargs)

    image_token_count = _placeholder_count_from_prompt_updates(
        processor, images, pixel_values, hf_processor_mm_kwargs
    )
    expected_from_pixel_values = _expected_placeholder_tokens_per_image(
        hf_processor, pixel_values[0]
    )
    assert expected_from_pixel_values == expected_toks_per_img
    assert image_token_count == expected_from_pixel_values * num_imgs


def test_lightonocr_keeps_vision_config_image_size():
    ctx = build_model_context(
        _LIGHTON_MODEL_ID,
        mm_processor_kwargs={"size": {"longest_edge": 1008}},
        model_config_kwargs=_MODEL_CONFIG_KWARGS,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    assert isinstance(processor.info, LightOnOCRProcessingInfo)
    encoder_info = processor.info.get_vision_encoder_info()
    assert isinstance(encoder_info, PixtralHFEncoderInfo)
    assert not isinstance(encoder_info, Mistral3HFEncoderInfo)
    assert encoder_info.get_image_size() == (
        processor.info.get_hf_config().vision_config.image_size
    )
