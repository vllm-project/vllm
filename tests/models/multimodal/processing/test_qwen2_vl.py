# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ....conftest import ImageTestAssets
from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["Qwen/Qwen2-VL-2B-Instruct"])
@pytest.mark.parametrize(
    ("mm_processor_kwargs", "expected_toks_per_img", "expected_pixels_shape"),
    [
        ({}, 1426, (5704, 1176)),
        ({"min_pixels": 64**2, "max_pixels": 512**2}, 330, (1320, 1176)),
    ],
)
@pytest.mark.parametrize("num_imgs", [1, 2])
@pytest.mark.parametrize("kwargs_on_init", [True, False])
def test_processor_override(
    image_assets: ImageTestAssets,
    model_id: str,
    mm_processor_kwargs: dict[str, object],
    expected_toks_per_img: int,
    expected_pixels_shape: tuple[int, int],
    num_imgs: int,
    kwargs_on_init: bool,
):
    """Ensure Qwen2VLMultiModalProcessor handles min/max pixels properly."""
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokenizer = processor.info.get_tokenizer()
    hf_processor_mm_kwargs = {} if kwargs_on_init else mm_processor_kwargs

    # Build the image str / prompt based on the number of images we pass
    prompt = "<|vision_start|><|image_pad|><|vision_end|>" * num_imgs
    mm_data = {"image": [image_assets[0].pil_image] * num_imgs}

    processed_inputs = processor.apply(prompt, mm_data, hf_processor_mm_kwargs)

    # Ensure we have the right number of placeholders per num_crops size
    hf_processor = processor.info.get_hf_processor(**hf_processor_mm_kwargs)
    image_token_id = tokenizer.convert_tokens_to_ids(hf_processor.image_token)
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    pixel_shape = processed_inputs["mm_kwargs"].get_data()["pixel_values"].shape

    assert img_tok_count == expected_toks_per_img * num_imgs
    assert pixel_shape[0] == expected_pixels_shape[0] * num_imgs
    assert pixel_shape[1] == expected_pixels_shape[1]


@pytest.mark.parametrize("model_id", ["Qwen/Qwen2-VL-2B-Instruct"])
@pytest.mark.parametrize("max_pixels", [1280 * 28 * 28, 1283 * 28 * 28])
def test_get_image_size_with_most_features(
    image_assets: ImageTestAssets,
    model_id: str,
    max_pixels: int,
):
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs={"max_pixels": max_pixels},
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    hf_processor_mm_kwargs: dict[str, object] = {}
    hf_processor = processor.info.get_hf_processor(**hf_processor_mm_kwargs)
    merge_size = processor.info.get_hf_config().vision_config.spatial_merge_size

    max_image_size = processor.info.get_image_size_with_most_features()
    max_tokens = processor.info.get_num_image_tokens(
        image_width=max_image_size.width,
        image_height=max_image_size.height,
        image_processor=hf_processor.image_processor,
    )

    prompt = "<|vision_start|><|image_pad|><|vision_end|>"
    for asset in image_assets:
        mm_data = {"image": [asset.pil_image]}
        processed_inputs = processor.apply(prompt, mm_data, hf_processor_mm_kwargs)
        grid_thw = processed_inputs["mm_kwargs"].get_data()["image_grid_thw"].tolist()
        t, h, w = grid_thw[0]
        tokens = (t * h * w) // (merge_size**2)
        assert tokens < max_tokens
