# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for H2OVL's multimodal preprocessing kwargs."""

from collections.abc import Mapping

import pytest
import torch
from PIL import Image
from transformers import PreTrainedConfig

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import rescale_image_size
from vllm.multimodal.processing import BaseMultiModalProcessor
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs

from ....conftest import ImageTestAssets
from ...registry import HF_EXAMPLE_MODELS
from ...utils import build_model_context


def _get_expected_num_patches(
    config: PreTrainedConfig,
    image: Image.Image,
    num_imgs: int,
    min_num: int,
    max_num: int,
):
    from vllm.transformers_utils.processors.h2ovl import (
        calculate_h2ovl_targets,
        get_h2ovl_target_ratios,
    )

    width, height = image.size

    # Calculate the expected number of blocks
    if num_imgs == 1 and config.use_msac:
        # First pass
        blocks1, _, _, aspect_ratio = calculate_h2ovl_targets(
            orig_width=width,
            orig_height=height,
            target_ratios=get_h2ovl_target_ratios(
                min_num=1,
                max_num=max_num,
                prior_aspect_ratio=None,
            ),
            image_size=config.vision_config.image_size,
            use_thumbnail=False,  # Thumbnail is handled separately
        )

        # Second pass
        blocks2, _, _, _ = calculate_h2ovl_targets(
            orig_width=width,
            orig_height=height,
            target_ratios=get_h2ovl_target_ratios(
                min_num=3,
                max_num=max_num,
                prior_aspect_ratio=aspect_ratio,
            ),
            image_size=config.vision_config.image_size,
            use_thumbnail=False,
        )

        # Add thumbnail if use_thumbnail is True and total_blocks > 1
        if config.use_thumbnail:
            blocks1 += 1 if blocks1 > 1 else 0
            blocks2 += 1 if blocks2 > 1 else 0

        # Total blocks is the sum of blocks from both passes minus
        # overlapping
        total_blocks = blocks1 + blocks2 - 1

        return total_blocks

    blocks, _, _, _ = calculate_h2ovl_targets(
        orig_width=width,
        orig_height=height,
        target_ratios=get_h2ovl_target_ratios(
            min_num,
            max_num,
            prior_aspect_ratio=None,
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

    prompt = "<image>" * len(images)
    mm_data = {"image": images}

    total_expected_num_patches = sum(
        _get_expected_num_patches(config, image, len(images), min_num, max_num)
        for image in images
    )

    processed_inputs = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=mm_processor_kwargs,
    )

    # Ensure we have the right number of placeholders per num_crops size
    image_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    pixel_values = processed_inputs["mm_kwargs"].get_data()["pixel_values_flat"]
    assert isinstance(pixel_values, torch.Tensor)
    pixel_shape = pixel_values.shape

    assert img_tok_count == 256 * total_expected_num_patches
    assert pixel_shape[0] == total_expected_num_patches


@pytest.mark.parametrize(
    "model_id",
    [
        "h2oai/h2ovl-mississippi-800m",
        "h2oai/h2ovl-mississippi-2b",
    ],
)
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
    image_assets: ImageTestAssets,
    size_factors: list[int],
    min_dynamic_patch: int,
    max_dynamic_patch: int,
    dynamic_image_size: bool | None,
    kwargs_on_init: bool,
):
    mm_processor_kwargs = {
        "min_dynamic_patch": min_dynamic_patch,
        "max_dynamic_patch": max_dynamic_patch,
        "dynamic_image_size": dynamic_image_size,
    }

    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": len(size_factors)},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor_mm_kwargs = {} if kwargs_on_init else mm_processor_kwargs

    min_num = min_dynamic_patch if dynamic_image_size else 1
    max_num = max_dynamic_patch if dynamic_image_size else 1

    _run_check(
        processor,
        [rescale_image_size(image_assets[0].pil_image, f) for f in size_factors],
        min_num,
        max_num,
        hf_processor_mm_kwargs,
    )


@pytest.mark.parametrize("use_uuid", [False, True])
@pytest.mark.parametrize(
    "use_embeds, use_msac", [(False, True), (False, False), (True, True)]
)
def test_image_cache_identity_across_image_counts(
    use_uuid: bool, use_embeds: bool, use_msac: bool
):
    config = ModelConfig(
        model="h2oai/h2ovl-mississippi-800m",
        trust_remote_code=HF_EXAMPLE_MODELS.find_hf_info(
            "h2oai/h2ovl-mississippi-800m"
        ).trust_remote_code,
        hf_overrides={"use_msac": use_msac},
        limit_mm_per_prompt={"image": 3},
        enable_mm_embeds=True,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(config)
    tokenizer = processor.info.get_tokenizer()
    images: list[torch.Tensor] | list[Image.Image]
    if use_embeds:
        images = [torch.ones(8, config.get_hidden_size())] * 3
    else:
        images = [
            Image.new("RGB", (1200, 800), color) for color in ("red", "green", "blue")
        ]
    hashes, lengths = [], []
    for count in (1, 2, 3):
        inputs = ProcessorInputs(
            prompt=tokenizer.encode("<image>\n" * count + "Describe."),
            mm_data_items=processor.info.parse_mm_data({"image": images[:count]}),
            mm_uuid_items={"image": [f"image-{i}" for i in range(count)]}
            if use_uuid
            else None,
        )
        if use_embeds:
            hashes.append(processor._get_mm_hashes(inputs)["image"][0])
            continue
        result = processor.apply(inputs, TimingContext(enabled=False))
        hashes.append(result["mm_hashes"]["image"][0])
        lengths.append(result["mm_placeholders"]["image"][0].length)
    assert hashes[1] == hashes[2]
    if use_embeds:
        assert hashes[0] == hashes[1]
        if use_uuid:
            assert hashes[0] == "image-0"
    elif use_msac:
        assert lengths[0] > lengths[1] == lengths[2]
        assert hashes[0] != hashes[1]
    else:
        assert lengths[0] == lengths[1] == lengths[2]
        assert hashes[0] == hashes[1]
