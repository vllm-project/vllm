# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.processors.deepseek_ocr import calculate_aspect_ratios

from ...utils import build_model_context


@pytest.mark.usefixtures("default_vllm_config")
def test_profiling_tokens_cover_all_crop_grids():
    """Every supported crop grid fits the profiled image token budget."""
    ctx = build_model_context(
        "baidu/Unlimited-OCR",
        limit_mm_per_prompt={"image": 1},
        model_config_kwargs={"max_model_len": 8192},
    )
    mm_processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor = mm_processor.info.get_hf_processor()
    dummy_inputs = mm_processor.get_dummy_mm_inputs({"image": 1})
    budget = dummy_inputs["mm_placeholders"]["image"][0].get_num_embeds()

    for width_tiles, height_tiles in calculate_aspect_ratios(
        max_num=hf_processor.max_crops
    ):
        num_tokens = mm_processor.info.get_num_image_tokens(
            image_width=width_tiles * hf_processor.image_size,
            image_height=height_tiles * hf_processor.image_size,
        )
        assert num_tokens <= budget, (width_tiles, height_tiles, num_tokens, budget)
