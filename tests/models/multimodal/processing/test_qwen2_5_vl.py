# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ....conftest import ImageTestAssets
from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["Qwen/Qwen2.5-VL-3B-Instruct"])
# yapf: disable
@pytest.mark.parametrize(
    ("resize_shape"), [
        ((112, 112)),
        ((114, 114)),
        ((256, 221)),
        ((1024, 1080)),
        ((784, 1120)),
    ])
# yapf: enable
@pytest.mark.parametrize("num_imgs", [1, 2])
def test_processor_force_alignment_resize(
    image_assets: ImageTestAssets,
    model_id: str,
    resize_shape: tuple[int, int],
    num_imgs: int,
):
    """Ensure images are resized by factor 112."""

    w, h = resize_shape
    factor = 112
    h_bar = round(h / factor) * factor
    w_bar = round(w / factor) * factor
    expected_pixels_shape_zero = (w_bar // 14) * (h_bar // 14)
    expected_pixels_shape_one = 1176
    expected_toks_per_img = expected_pixels_shape_zero // 4
    mm_processor_kwargs: dict[str, object] = {}

    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokenizer = processor.info.get_tokenizer()

    # Build the image str / prompt based on the number of images we pass
    prompt = "<|vision_start|><|image_pad|><|vision_end|>" * num_imgs
    mm_data = {
        "image": [image_assets[0].pil_image.resize(resize_shape)] * num_imgs
    }

    processed_inputs = processor.apply(prompt, mm_data, mm_processor_kwargs)

    hf_processor = processor.info.get_hf_processor(**mm_processor_kwargs)

    # Ensure we have the right number of placeholders per num_crops size
    image_token_id = tokenizer.convert_tokens_to_ids(hf_processor.image_token)
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    pixel_shape = processed_inputs["mm_kwargs"]["pixel_values"].shape

    assert img_tok_count == expected_toks_per_img * num_imgs
    assert pixel_shape[0] == expected_pixels_shape_zero * num_imgs
    assert pixel_shape[1] == expected_pixels_shape_one
    assert pixel_shape[0] % 64 == 0


@pytest.mark.parametrize("model_id", ["Qwen/Qwen2.5-VL-3B-Instruct"])
# yapf: disable
@pytest.mark.parametrize(
    ("resize_shape"), [
        ((110, 112)),
        ((32, 32)),
    ])
# yapf: enable
@pytest.mark.parametrize("num_imgs", [1])
def test_processor_force_alignment_resize_to_min_value(
    image_assets: ImageTestAssets,
    model_id: str,
    resize_shape: tuple[int, int],
    num_imgs: int,
):
    """Ensure processor resizes small images to 112 x 112"""
    expected_pixels_shape_zero = (112 // 14) * (112 // 14)
    expected_pixels_shape_one = 1176
    expected_toks_per_img = expected_pixels_shape_zero // 4

    mm_processor_kwargs: dict[str, object] = {}

    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokenizer = processor.info.get_tokenizer()

    prompt = "<|vision_start|><|image_pad|><|vision_end|>" * num_imgs
    mm_data = {
        "image": [image_assets[0].pil_image.resize(resize_shape)] * num_imgs
    }

    processed_inputs = processor.apply(prompt, mm_data, mm_processor_kwargs)

    hf_processor = processor.info.get_hf_processor(**mm_processor_kwargs)

    # Ensure we have the right number of placeholders per num_crops size
    image_token_id = tokenizer.convert_tokens_to_ids(hf_processor.image_token)
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    pixel_shape = processed_inputs["mm_kwargs"]["pixel_values"].shape

    assert img_tok_count == expected_toks_per_img * num_imgs
    assert pixel_shape[0] == expected_pixels_shape_zero * num_imgs
    assert pixel_shape[1] == expected_pixels_shape_one
    assert pixel_shape[0] % 64 == 0
