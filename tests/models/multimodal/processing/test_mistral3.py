# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Mistral3's multimodal preprocessing kwargs."""

import pytest
import torch
from PIL import Image
from transformers import BatchFeature

from vllm.model_executor.models.mistral3 import Mistral3HFEncoderInfo
from vllm.model_executor.models.pixtral import PixtralHFEncoderInfo
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItems

from ...utils import build_model_context

# This repo ships both params.json (Mistral) and config.json (HF). Auto config
# selects PixtralForConditionalGeneration; force HF to exercise Mistral3.
_MODEL_CONFIG_KWARGS = {"config_format": "hf"}
_MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"


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
    # Maps each HF tensor name to a modality (here: pixel_values → image).
    fields_config = processor._get_mm_fields_config(hf_inputs, mm_processor_kwargs)
    # Splits the HF batch into the per-image kwargs _get_prompt_updates reads.
    out_mm_kwargs = MultiModalKwargsItems.from_hf_inputs(hf_inputs, fields_config)
    # Raw PIL items; their sizes must not drive the placeholder count.
    mm_items = processor.info.parse_mm_data({"image": images})
    # Method under test: placeholder grid from processed pixel_values, not raw PIL.
    updates = processor._get_prompt_updates(
        mm_items, mm_processor_kwargs, out_mm_kwargs
    )
    image_token_id = processor.info.get_hf_config().image_token_index
    total = 0
    for item_idx in range(len(images)):
        # Expand this image's PromptReplacement into token ids.
        details = updates[0].resolve(item_idx).content
        total += details.full.count(image_token_id)
    return total


def _expected_placeholder_tokens_per_image(
    hf_processor,
    pixel_values: torch.Tensor,
) -> int:
    """Ground truth: processed H×W divided by patch_size * spatial_merge_size."""
    image_h, image_w = pixel_values.shape[-2:]
    patch_size = hf_processor.image_processor.patch_size
    if isinstance(patch_size, dict):
        patch_h = patch_size["height"]
        patch_w = patch_size["width"]
    else:
        patch_h = patch_w = int(patch_size)
    # spatial_merge_size is on PixtralProcessor / Mistral3Config, not the
    # image processor.
    spatial_merge_size = getattr(hf_processor, "spatial_merge_size", 1)
    return (image_h // (patch_h * spatial_merge_size)) * (
        image_w // (patch_w * spatial_merge_size)
    )


def _tokens_from_raw_size_and_vision_config(
    hf_config,
    image_size: tuple[int, int],
) -> int:
    """Token count if the raw image is scaled to vision_config.image_size."""
    raw_w, raw_h = image_size
    return PixtralHFEncoderInfo(hf_config).get_num_image_tokens(
        image_width=raw_w,
        image_height=raw_h,
    )


@pytest.mark.parametrize("model_id", [_MODEL_ID])
def test_mistral3_encoder_info_uses_processed_dims(model_id: str):
    """Mistral3HFEncoderInfo must not ratio-scale already-processed dims."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
        model_config_kwargs=_MODEL_CONFIG_KWARGS,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_config = processor.info.get_hf_config()
    encoder_info = processor.info.get_vision_encoder_info()
    assert isinstance(encoder_info, Mistral3HFEncoderInfo)

    image_size = encoder_info.get_image_size()
    patch_size = encoder_info.get_patch_size()
    # Canvas larger than vision_config.image_size, aligned to the patch stride.
    processed = ((image_size // patch_size) + 4) * patch_size

    ncols, nrows = encoder_info.get_patch_grid_size(
        image_width=processed, image_height=processed
    )
    assert (ncols, nrows) == (processed // patch_size, processed // patch_size)

    parent = PixtralHFEncoderInfo(hf_config)
    parent_ncols, parent_nrows = parent.get_patch_grid_size(
        image_width=processed, image_height=processed
    )
    # PixtralHFEncoderInfo still treats image_size as the max edge.
    assert (parent_ncols, parent_nrows) != (ncols, nrows)
    assert parent_ncols * parent_nrows < ncols * nrows


@pytest.mark.parametrize("model_id", [_MODEL_ID])
@pytest.mark.parametrize(
    "mm_processor_kwargs",
    [
        {},
        {"size": {"longest_edge": 1008}},
        {"size": {"longest_edge": 1288}},
    ],
)
@pytest.mark.parametrize(
    "image_size",
    [
        (1540, 1540),
        (1536, 1187),
        (2200, 1700),
    ],
)
@pytest.mark.parametrize("num_imgs", [1, 2])
@pytest.mark.parametrize("kwargs_on_init", [True, False])
def test_processor_placeholder_matches_processed_pixel_values(
    model_id: str,
    mm_processor_kwargs: dict[str, object],
    image_size: tuple[int, int],
    num_imgs: int,
    kwargs_on_init: bool,
):
    """Placeholder count must follow the HF-processed pixel_values geometry."""
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
    # Resize once; reused as both the prompt-update input and the expected count.
    pixel_values = _processed_pixel_values(hf_processor, images, hf_processor_mm_kwargs)

    img_tok_count = _placeholder_count_from_prompt_updates(
        processor, images, pixel_values, hf_processor_mm_kwargs
    )
    expected_per_image = _expected_placeholder_tokens_per_image(
        hf_processor, pixel_values[0]
    )
    assert img_tok_count == expected_per_image * num_imgs

    # Scaling the raw photo to vision_config.image_size ignores longest_edge.
    vision_config_count = _tokens_from_raw_size_and_vision_config(
        processor.info.get_hf_config(), image_size
    )
    if vision_config_count != expected_per_image:
        assert img_tok_count != vision_config_count * num_imgs
