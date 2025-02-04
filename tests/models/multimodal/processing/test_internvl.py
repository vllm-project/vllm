# SPDX-License-Identifier: Apache-2.0
"""Tests for InternVL's multimodal preprocessing kwargs."""
from typing import Optional

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.utils import cached_get_tokenizer

from ....conftest import _ImageAssets
from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["OpenGVLab/InternVL2-2B"])
@pytest.mark.parametrize("max_dynamic_patch", [1, 4])
@pytest.mark.parametrize("dynamic_image_size", [True, False, None])
@pytest.mark.parametrize("num_imgs", [1, 2])
def test_processor_override(
    model_id: str,
    image_assets: _ImageAssets,
    max_dynamic_patch: int,
    dynamic_image_size: Optional[bool],
    num_imgs: int,
):
    ctx = build_model_context(
        model_name=model_id,
        tokenizer_name=model_id,
        trust_remote_code=True,
        mm_processor_kwargs=None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    tokenizer = cached_get_tokenizer(
        ctx.model_config.tokenizer,
        trust_remote_code=ctx.model_config.trust_remote_code,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config,
        tokenizer=tokenizer,
    )

    mm_processor_kwargs = {
        "max_dynamic_patch": max_dynamic_patch,
    }
    if dynamic_image_size is not None:
        mm_processor_kwargs["dynamic_image_size"] = dynamic_image_size

    # Build the image str / prompt based on the number of images we pass
    prompt = "<image>" * num_imgs
    image = image_assets[0].pil_image.resize((448 * 2, 448 * 2))
    mm_data = {"image": [image] * num_imgs}

    expected_num_patches = max_dynamic_patch + 1 if max_dynamic_patch > 1 else 1
    if dynamic_image_size is False:
        expected_num_patches = 1

    processed_inputs = processor.apply(prompt, mm_data, mm_processor_kwargs)

    # Ensure we have the right number of placeholders per num_crops size
    image_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    pixel_shape = processed_inputs["mm_kwargs"]["pixel_values_flat"].shape

    assert img_tok_count == 256 * expected_num_patches * num_imgs
    assert pixel_shape[0] == expected_num_patches * num_imgs
