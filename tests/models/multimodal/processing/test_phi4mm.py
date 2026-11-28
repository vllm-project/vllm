# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for phi4mm's multimodal preprocessing kwargs."""

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ....conftest import ImageTestAssets
from ...registry import HF_EXAMPLE_MODELS
from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["microsoft/Phi-4-multimodal-instruct"])
@pytest.mark.parametrize(
    ("mm_processor_kwargs", "expected_toks_per_img"),
    [
        ({"dynamic_hd": 4}, 1329),
        ({"dynamic_hd": 16}, 4433),
        # the default num_crops of phi-4-multimodal is 36
        ({}, 9585),
    ],
)
@pytest.mark.parametrize("num_imgs", [1, 2])
@pytest.mark.parametrize("kwargs_on_init", [True, False])
def test_processor_override(
    image_assets: ImageTestAssets,
    model_id: str,
    mm_processor_kwargs: dict[str, int],
    expected_toks_per_img: int,
    num_imgs: int,
    kwargs_on_init: bool,
):
    """Ensure Phi4MMMultiModalProcessor handles dynamic_hd properly."""
    HF_EXAMPLE_MODELS.get_hf_info(
        "Phi4MultimodalForCausalLM"
    ).check_transformers_version(
        on_fail="skip",
        check_max_version=False,
        check_version_reason="vllm",
    )

    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor_mm_kwargs = {} if kwargs_on_init else mm_processor_kwargs

    img_str = "<|image|>\n" * num_imgs
    prompt = f"<|user|>\n{img_str}<|end|>\n<|assistant|>\n"

    image_size = ctx.get_hf_config().vision_config.crop_size
    dummy_image_size = (image_size * 7, image_size * 7)
    dummy_image = image_assets[0].pil_image.resize(dummy_image_size)
    mm_data = {"image": [dummy_image] * num_imgs}

    processed_inputs = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    )

    # Ensure we have the right number of placeholders per num_crops size
    image_token_id = ctx.get_hf_config().image_token_id
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    assert img_tok_count == expected_toks_per_img * num_imgs
