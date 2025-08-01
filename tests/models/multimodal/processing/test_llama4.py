# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Llama4's multimodal preprocessing kwargs."""

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.tokenizer import encode_tokens

from ....conftest import ImageTestAssets
from ...utils import build_model_context


@pytest.mark.parametrize("model_id",
                         ["meta-llama/Llama-4-Scout-17B-16E-Instruct"])
@pytest.mark.parametrize("mm_processor_kwargs", [{}])
@pytest.mark.parametrize("num_imgs", [1, 5])
@pytest.mark.parametrize("disable_mm_preprocessor_cache", [True, False])
@pytest.mark.parametrize("tokenized_prompt", [True, False])
def test_processor_override(
    image_assets: ImageTestAssets,
    model_id: str,
    mm_processor_kwargs: dict,
    num_imgs: int,
    disable_mm_preprocessor_cache: bool,
    tokenized_prompt: bool,
):
    """Ensure llama4 processor works properly."""
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"image": num_imgs},
        disable_mm_preprocessor_cache=disable_mm_preprocessor_cache,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    config = processor.info.get_hf_config()
    tokenizer = processor.info.get_tokenizer()
    hf_processor = processor.info.get_hf_processor()
    vocab = tokenizer.get_vocab()

    prompt = "<|begin_of_text|><|header_start|>user<|header_end|>" \
        + "<|image|>" * num_imgs \
        + "<|eot|><|header_start|>assistant<|header_end|>"
    mm_data = {
        "image": [
            image_assets[(i % len(image_assets))].pil_image
            for i in range(num_imgs)
        ]
    }
    if tokenized_prompt:
        prompt = encode_tokens(tokenizer, prompt)

    processed_inputs = processor.apply(prompt, mm_data, mm_processor_kwargs)
    mm_kwargs = processed_inputs["mm_kwargs"]

    # place holder replacements
    prompt_token_ids = processed_inputs["prompt_token_ids"]
    assert prompt_token_ids.count(config.boi_token_index) == num_imgs
    assert prompt_token_ids.count(config.eoi_token_index) == num_imgs
    assert prompt_token_ids.count(vocab[hf_processor.image_token]) == num_imgs
    aspect_ratios = mm_kwargs["aspect_ratios"]
    num_x_separators = num_y_separators = 0
    for tiles_y, tiles_x in aspect_ratios:
        if tiles_x * tiles_y > 1:
            num_x_separators += (tiles_x - 1) * tiles_y
            num_y_separators += tiles_y
    assert prompt_token_ids.count(vocab[hf_processor.tile_token]) \
        == num_x_separators
    assert prompt_token_ids.count(vocab[hf_processor.tile_global_token]) \
        ==  num_y_separators

    # image token offsets
    img_locs = processed_inputs["mm_placeholders"].get("image", [])
    assert len(img_locs) == num_imgs
    assert [img_loc.offset for img_loc in img_locs] == \
        [i for i, v in enumerate(prompt_token_ids) \
        if v == config.boi_token_index]

    # patch sizes and masks
    num_patches_per_chunk = processor.info.get_patch_per_chunk(
        config.vision_config)
    assert prompt_token_ids.count(config.image_token_index) \
        == mm_kwargs["patches_per_image"].sum() * num_patches_per_chunk
    assert mm_kwargs["pixel_values"].shape[0] \
        == mm_kwargs["patches_per_image"].sum()
