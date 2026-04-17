# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for QianfanOCR's multimodal preprocessing kwargs.

QianfanOCR is architecturally identical to InternVLChatModel
(InternViT vision encoder + pixel-shuffle MLP + Qwen3 LLM), so the
processing logic is fully shared.  These tests verify that the
QianfanOCRForConditionalGeneration registration wires up the processor
correctly and that dynamic-patch tiling produces the expected token counts.
"""

from collections.abc import Mapping

import pytest
from PIL import Image
from transformers import PretrainedConfig

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import rescale_image_size
from vllm.multimodal.processing import BaseMultiModalProcessor

from ....conftest import ImageTestAssets
from ...utils import build_model_context

MODEL_PATH = "bairongz/QianfanOCR"


def _get_expected_num_patches(
    config: PretrainedConfig,
    image: Image.Image,
    min_num: int,
    max_num: int,
) -> int:
    from vllm.transformers_utils.processors.internvl import (
        calculate_internvl_targets,
        get_internvl_target_ratios,
    )

    width, height = image.size

    blocks, _, _ = calculate_internvl_targets(
        orig_width=width,
        orig_height=height,
        target_ratios=get_internvl_target_ratios(min_num, max_num),
        image_size=config.vision_config.image_size,
        use_thumbnail=False,
    )

    if config.use_thumbnail and blocks > 1:
        blocks += 1

    return blocks


def _run_check(
    processor: BaseMultiModalProcessor,
    images: list[Image.Image],
    min_num: int,
    max_num: int,
    mm_processor_kwargs: Mapping[str, object],
) -> None:
    tokenizer = processor.info.get_tokenizer()
    config = processor.info.get_hf_config()

    # num_image_token = (force_image_size / patch_size)^2 * downsample_ratio^2
    # = (448 / 14)^2 * 0.5^2 = 1024 * 0.25 = 256
    image_size = config.force_image_size or config.vision_config.image_size
    patch_size = config.vision_config.patch_size
    num_image_token = int((image_size // patch_size) ** 2 * config.downsample_ratio**2)

    total_expected_patches = sum(
        _get_expected_num_patches(config, img, min_num, max_num) for img in images
    )

    processed_inputs = processor(
        "<image>" * len(images),
        mm_items=processor.info.parse_mm_data({"image": images}),
        hf_processor_mm_kwargs=mm_processor_kwargs,
    )

    image_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    pixel_shape = processed_inputs["mm_kwargs"].get_data()["pixel_values_flat"].shape

    assert img_tok_count == num_image_token * total_expected_patches
    assert pixel_shape[0] == total_expected_patches


@pytest.mark.parametrize("model_id", [MODEL_PATH])
@pytest.mark.parametrize(
    "size_factors",
    [
        # Single image
        [1.0],
        # Multiple images, same scale
        [1.0, 1.0, 1.0],
        # Multiple images, mixed scales
        [0.25, 0.5, 1.0],
        [4.0, 2.0, 1.0],
    ],
)
@pytest.mark.parametrize(
    ("min_dynamic_patch", "max_dynamic_patch"),
    [(1, 1), (1, 4), (1, 12), (2, 6), (4, 12)],
)
@pytest.mark.parametrize("dynamic_image_size", [True, False])
@pytest.mark.parametrize("kwargs_on_init", [True, False])
def test_processor_override(
    model_id: str,
    image_assets: ImageTestAssets,
    size_factors: list[float],
    min_dynamic_patch: int,
    max_dynamic_patch: int,
    dynamic_image_size: bool,
    kwargs_on_init: bool,
) -> None:
    """Verify that dynamic-patch kwargs produce the correct token counts."""
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


@pytest.mark.parametrize("model_id", [MODEL_PATH])
def test_config_loaded_correctly(model_id: str) -> None:
    """Smoke-test: QianfanOCRConfig fields are accessible as expected."""
    ctx = build_model_context(model_id, limit_mm_per_prompt={"image": 1})
    config = ctx.model_config.hf_config

    assert config.model_type == "qianfan_ocr"
    assert config.vision_config.model_type == "qianfan_ocr_vision"
    assert config.vision_config.image_size == 448
    assert config.vision_config.patch_size == 14
    assert config.downsample_ratio == 0.5
    assert config.text_config.model_type == "qwen3"


@pytest.mark.parametrize("model_id", [MODEL_PATH])
def test_processor_registered(model_id: str) -> None:
    """Smoke-test: the multimodal processor is created without error."""
    ctx = build_model_context(model_id, limit_mm_per_prompt={"image": 1})
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    assert processor is not None
