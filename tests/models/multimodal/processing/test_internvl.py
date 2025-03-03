# SPDX-License-Identifier: Apache-2.0
"""Tests for InternVL's multimodal preprocessing kwargs."""
from collections.abc import Mapping
from typing import Optional

import pytest
from PIL import Image
from transformers import PretrainedConfig

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import rescale_image_size
from vllm.multimodal.processing import BaseMultiModalProcessor
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config

from ....conftest import _ImageAssets
from ...utils import build_model_context


def _get_expected_num_patches(
    config: PretrainedConfig,
    image: Image.Image,
    num_imgs: int,
    min_num: int,
    max_num: int,
):
    from vllm.model_executor.models.internvl import (
        calculate_internvl_targets, get_internvl_target_ratios)

    width, height = image.size

    blocks, _, _ = calculate_internvl_targets(
        orig_width=width,
        orig_height=height,
        target_ratios=get_internvl_target_ratios(
            min_num,
            max_num,
        ),
        image_size=config.vision_config.image_size,
        use_thumbnail=False,
    )
    expected_num_patches = blocks

    if config.use_thumbnail and expected_num_patches > 1:
        expected_num_patches += 1

    return expected_num_patches


def _run_check(
    processor: BaseMultiModalProcessor,
    images: list[Image.Image],
    min_num: int,
    max_num: int,
    mm_processor_kwargs: Mapping[str, object],
):
    tokenizer = processor.info.get_tokenizer()
    config = processor.info.get_hf_config()

    mm_data = {"image": images}

    total_expected_num_patches = sum(
        _get_expected_num_patches(config, image, len(images), min_num, max_num)
        for image in images)

    processed_inputs = processor.apply("<image>" * len(images), mm_data,
                                       mm_processor_kwargs)

    # Ensure we have the right number of placeholders per num_crops size
    image_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    pixel_shape = processed_inputs["mm_kwargs"]["pixel_values_flat"].shape

    assert img_tok_count == 256 * total_expected_num_patches
    assert pixel_shape[0] == total_expected_num_patches


@pytest.mark.parametrize("model_id", ["OpenGVLab/InternVL2-2B"])
@pytest.mark.parametrize(
    "size_factors",
    [
        # Single-scale
        [1.0],
        # Single-scale, batched
        [1.0, 1.0, 1.0],
        # Multi-scale
        [0.25, 0.5, 1.0],
        [4.0, 2.0, 1.0],
    ],
)
@pytest.mark.parametrize(
    ("min_dynamic_patch", "max_dynamic_patch"),
    [(1, 1), (1, 2), (1, 4), (1, 8), (2, 4), (4, 8)],
)
@pytest.mark.parametrize("dynamic_image_size", [True, False])
@pytest.mark.parametrize("kwargs_on_init", [True, False])
def test_processor_override(
    model_id: str,
    image_assets: _ImageAssets,
    size_factors: list[int],
    min_dynamic_patch: int,
    max_dynamic_patch: int,
    dynamic_image_size: Optional[bool],
    kwargs_on_init: bool,
):
    mm_processor_kwargs = {
        "min_dynamic_patch": min_dynamic_patch,
        "max_dynamic_patch": max_dynamic_patch,
        "dynamic_image_size": dynamic_image_size,
    }

    ctx = build_model_context(
        model_name=model_id,
        tokenizer_name=model_id,
        trust_remote_code=True,
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": len(size_factors)},
    )
    tokenizer = cached_tokenizer_from_config(ctx.model_config)
    processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config,
        tokenizer=tokenizer,
    )
    hf_processor_mm_kwargs = {} if kwargs_on_init else mm_processor_kwargs

    min_num = min_dynamic_patch if dynamic_image_size else 1
    max_num = max_dynamic_patch if dynamic_image_size else 1

    _run_check(
        processor,
        [
            rescale_image_size(image_assets[0].pil_image, f)
            for f in size_factors
        ],
        min_num,
        max_num,
        hf_processor_mm_kwargs,
    )
