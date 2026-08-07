# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for InternVL's multimodal preprocessing kwargs."""

from collections.abc import Mapping
from types import SimpleNamespace

import pytest
from PIL import Image
from transformers import PretrainedConfig

from vllm.model_executor.models.internvl import InternVLProcessingInfo
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import rescale_image_size
from vllm.multimodal.processing import BaseMultiModalProcessor

from ....conftest import ImageTestAssets
from ...utils import build_model_context


class _FakeInternVLProcessingContext:
    def __init__(self, mm_processor_kwargs: Mapping[str, object] | None = None) -> None:
        self._mm_processor_kwargs = dict(mm_processor_kwargs or {})
        self._config = SimpleNamespace(
            vision_config=SimpleNamespace(image_size=448),
            min_dynamic_patch=1,
            max_dynamic_patch=12,
            dynamic_image_size=True,
            use_thumbnail=True,
        )

    def get_hf_config(self):
        return self._config

    def get_merged_mm_kwargs(self, kwargs: Mapping[str, object]) -> dict[str, object]:
        return self._mm_processor_kwargs | dict(kwargs)


def _get_expected_num_patches(
    config: PretrainedConfig,
    image: Image.Image,
    num_imgs: int,
    min_num: int,
    max_num: int,
):
    from vllm.transformers_utils.processors.internvl import (
        calculate_internvl_targets,
        get_internvl_target_ratios,
    )

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
    pixel_shape = processed_inputs["mm_kwargs"].get_data()["pixel_values_flat"].shape

    assert img_tok_count == 256 * total_expected_num_patches
    assert pixel_shape[0] == total_expected_num_patches


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("operator_kwargs", "request_max_dynamic_patch", "ceiling"),
    [
        ({}, 13, 12),
        ({"max_dynamic_patch": 4}, 5, 4),
    ],
)
def test_request_max_dynamic_patch_cannot_raise_trusted_ceiling(
    operator_kwargs: Mapping[str, object],
    request_max_dynamic_patch: int,
    ceiling: int,
) -> None:
    info = InternVLProcessingInfo(_FakeInternVLProcessingContext(operator_kwargs))

    with pytest.raises(
        ValueError,
        match=(
            rf"max_dynamic_patch={request_max_dynamic_patch} "
            rf"cannot exceed.*{ceiling}"
        ),
    ):
        info.get_image_processor(max_dynamic_patch=request_max_dynamic_patch)


@pytest.mark.cpu_test
def test_request_max_dynamic_patch_can_lower_trusted_ceiling() -> None:
    info = InternVLProcessingInfo(
        _FakeInternVLProcessingContext({"max_dynamic_patch": 4})
    )

    image_processor = info.get_image_processor(max_dynamic_patch=2)

    assert image_processor.max_dynamic_patch == 2


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
