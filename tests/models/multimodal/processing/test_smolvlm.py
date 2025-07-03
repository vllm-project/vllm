# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for smolvlm's multimodal preprocessing kwargs."""
import pytest
from transformers import SmolVLMConfig

from vllm.multimodal import MULTIMODAL_REGISTRY

from ....conftest import ImageTestAssets
from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["HuggingFaceTB/SmolVLM2-2.2B-Instruct"])
# yapf: disable
@pytest.mark.parametrize(
    ("mm_processor_kwargs", "expected_toks_per_img"),
    [
        ({"max_image_size": {"longest_edge": 384}}, 1377),
        ({"max_image_size": {"longest_edge": 768}}, 405),
    ])
# yapf: enable
@pytest.mark.parametrize("num_imgs", [1, 2])
@pytest.mark.parametrize("kwargs_on_init", [True, False])
def test_processor_override(
    image_assets: ImageTestAssets,
    model_id: str,
    mm_processor_kwargs: dict[str, object],
    expected_toks_per_img: int,
    num_imgs: int,
    kwargs_on_init: bool,
):
    """Ensure Idefics3MultiModalProcessor handles num_crops properly."""
    # Same as the previous test - don't initialize mm_processor_kwargs
    # in this test and assume that the kwargs will be correctly expanded by
    # the partial when calling the custom input processor.
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor_mm_kwargs = {} if kwargs_on_init else mm_processor_kwargs

    # Build the image str / prompt based on the number of images we pass
    placeholders = "<image>" if num_imgs == 1 else "\n".join(
        f"Image-{i}: <image>\n" for i in range(1, num_imgs + 1))
    prompt = f"<|im_start|>User:{placeholders}\n<end_of_utterance>\nAssistant:"  # noqa: E501

    # Build mm_data
    image_size = ctx.get_hf_config(SmolVLMConfig).vision_config.image_size
    dummy_image_size = (image_size * 4, image_size * 4)
    dummy_image = image_assets[0].pil_image.resize(dummy_image_size)
    mm_data = {"image": [dummy_image] * num_imgs}

    processed_inputs = processor.apply(prompt, mm_data, hf_processor_mm_kwargs)

    # Ensure the placeholders format are correct
    hf_processor = processor.info.get_hf_processor(**hf_processor_mm_kwargs)
    hf_processed_inputs = hf_processor(text=prompt, images=mm_data["image"])
    assert processed_inputs["prompt_token_ids"] == hf_processed_inputs[
        "input_ids"][0]

    # Ensure we have the right number of placeholders per num_crops size
    image_token_id = ctx.get_hf_config().image_token_id
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    assert img_tok_count == expected_toks_per_img * num_imgs
