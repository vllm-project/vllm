# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for phi3v's multimodal preprocessing kwargs."""

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ....conftest import ImageTestAssets
from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["microsoft/Phi-3.5-vision-instruct"])
@pytest.mark.parametrize(
    ("mm_processor_kwargs", "expected_toks_per_img"),
    [
        ({"num_crops": 4}, 757),
        ({"num_crops": 16}, 1921),
        # the default num_crops of phi-3.5-vision is 4
        ({}, 757),
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
    """Ensure Phi3VMultiModalProcessor handles num_crops properly."""
    # Avoid initializing CUDA early
    from vllm.model_executor.models.phi3v import _IMAGE_TOKEN_ID

    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor_mm_kwargs = {} if kwargs_on_init else mm_processor_kwargs

    # Build the image str / prompt based on the number of images we pass
    img_str = "".join([f"<|image_{idx}|>\n" for idx in range(1, num_imgs + 1)])
    prompt = f"<|user|>\n{img_str}<|end|>\n<|assistant|>\n"
    mm_data = {"image": [image_assets[0].pil_image] * num_imgs}

    processed_inputs = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    )

    # Ensure we have the right number of placeholders per num_crops size
    img_tok_count = processed_inputs["prompt_token_ids"].count(_IMAGE_TOKEN_ID)
    assert img_tok_count == expected_toks_per_img * num_imgs


@pytest.mark.parametrize("model_id", ["microsoft/Phi-3.5-vision-instruct"])
def test_out_of_order_image_tags(model_id: str):
    """`<|image_N|>` binds to the N-th image regardless of its prompt position."""
    from PIL import Image

    ctx = build_model_context(model_id, limit_mm_per_prompt={"image": 2})
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor = processor.info.get_hf_processor()

    images = [Image.new("RGB", (336, 336)), Image.new("RGB", (1344, 672))]
    num_tokens = [
        processor.info.get_num_image_tokens(
            image_width=image.width,
            image_height=image.height,
            processor=hf_processor,
        )
        for image in images
    ]
    assert num_tokens[0] != num_tokens[1]

    prompt = "<|user|>\n<|image_2|>\n<|image_1|>\n<|end|>\n<|assistant|>\n"
    processed_inputs = processor(
        prompt,
        mm_items=processor.info.parse_mm_data({"image": images}),
        hf_processor_mm_kwargs={},
    )

    image_1, image_2 = processed_inputs["mm_placeholders"]["image"]
    assert image_2.offset < image_1.offset
    assert [image_1.length, image_2.length] == num_tokens
