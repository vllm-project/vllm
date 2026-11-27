# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for LFM2-VL's multimodal preprocessing kwargs."""

import pytest

from vllm.model_executor.models.lfm2_vl import Lfm2VLProcessingInfo
from vllm.multimodal import MULTIMODAL_REGISTRY

from ....conftest import ImageTestAssets
from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["LiquidAI/LFM2-VL-450M"])
@pytest.mark.parametrize(
    ("mm_processor_kwargs", "expected_toks_per_img"),
    [
        ({}, 1770),
        ({"do_image_splitting": False}, 234),
        ({"use_thumbnail": False}, 1536),
        ({"min_tiles": 2, "max_tiles": 4}, 1258),
    ],
)
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
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    assert isinstance(processor.info, Lfm2VLProcessingInfo)
    hf_processor_mm_kwargs = {} if kwargs_on_init else mm_processor_kwargs

    prompt = (
        "<|startoftext|><|im_start|>user\n"
        + "<image>" * num_imgs
        + "Describe.<|im_end|>\n<|im_start|>assistant\n"
    )
    image = image_assets[0].pil_image.resize((1024, 768))
    mm_data = {"image": [image] * num_imgs}

    processed_inputs = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    )

    hf_processor = processor.info.get_hf_processor(**hf_processor_mm_kwargs)
    hf_processed_inputs = hf_processor(
        text=prompt,
        images=mm_data["image"],
        **processor.info.ctx.get_merged_mm_kwargs(hf_processor_mm_kwargs),
    )
    assert processed_inputs["prompt_token_ids"] == hf_processed_inputs["input_ids"][0]

    image_token_id = ctx.get_hf_config().image_token_id
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    assert img_tok_count == expected_toks_per_img * num_imgs
