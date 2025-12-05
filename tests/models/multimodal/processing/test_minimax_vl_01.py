# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from PIL import Image

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import ImageSize
from vllm.multimodal.processing import BaseMultiModalProcessor

from ....conftest import ImageTestAssets
from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["MiniMaxAI/MiniMax-VL-01"])
@pytest.mark.parametrize("num_imgs", [1, 2])
def test_processor_override(
    image_assets: ImageTestAssets,
    model_id: str,
    num_imgs: int,
):
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.renderer_config)
    prompt = "<image>" * num_imgs
    image = Image.new("RGB", size=(364, 364))
    mm_data = {"image": [image] * num_imgs}

    processed_inputs = processor.apply(prompt, mm_data, {})
    image_placeholders = processed_inputs["mm_placeholders"]["image"]

    assert len(image_placeholders) == num_imgs


def _validate_image_prompt_replacements_one(
    processor: BaseMultiModalProcessor,
    num_imgs: int,
    failed_size_excs: list[tuple[ImageSize, Exception]],
    image_size: ImageSize,
) -> None:
    prompt = "<image>" * num_imgs
    image = Image.new("RGB", size=image_size)
    mm_data = {"image": [image] * num_imgs}

    try:
        processed_inputs = processor.apply(prompt, mm_data, {})

        image_placeholders = processed_inputs["mm_placeholders"]["image"]
        assert len(image_placeholders) == num_imgs

    except Exception as exc:
        failed_size_excs.append((image_size, exc))


def _test_image_prompt_replacements(
    processor,
    *,
    num_imgs: int,
    image_sizes: list[ImageSize],
) -> None:
    failed_size_excs = list[tuple[ImageSize, Exception]]()

    for size in image_sizes:
        _validate_image_prompt_replacements_one(
            processor, num_imgs, failed_size_excs, size
        )

    if failed_size_excs:
        msg = "Found failing image sizes:" + "\n========\n".join(
            f"[{size}]\n{exc}" for size, exc in failed_size_excs
        )
        raise AssertionError(msg)


@pytest.mark.parametrize("model_id", ["MiniMaxAI/MiniMax-VL-01"])
@pytest.mark.parametrize("num_imgs", [1, 2])
def test_processor_prompt_replacements_regression(model_id, num_imgs):
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.renderer_config)

    image_ratios = [
        (171, 152),
        (184, 161),
        (198, 176),
        (333, 296),
        (369, 328),
        (488, 183),
        (2560, 1669),
    ]
    image_sizes = [
        size for w, h in image_ratios for size in [ImageSize(w, h), ImageSize(h, w)]
    ]

    _test_image_prompt_replacements(
        processor,
        num_imgs=num_imgs,
        image_sizes=image_sizes,
    )
