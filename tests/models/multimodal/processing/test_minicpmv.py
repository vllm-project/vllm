# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for MiniCPMV's multimodal preprocessing."""

import numpy as np
import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import ImageSize

from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["openbmb/MiniCPM-V-4"])
def test_get_hf_processor_for_same_model_different_kwargs(model_id: str):
    """Calls with different kwargs must not reuse stale processor instances."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    info = processor.info
    processor_1 = info.get_hf_processor(max_slice_nums=1)
    processor_2 = info.get_hf_processor(max_slice_nums=2)
    assert processor_1.image_processor.max_slice_nums == 1
    assert processor_2.image_processor.max_slice_nums == 2


@pytest.mark.parametrize(
    "model_ids",
    [
        ("openbmb/MiniCPM-Llama3-V-2_5", "openbmb/MiniCPM-V-4"),
        ("openbmb/MiniCPM-Llama3-V-2_5", "openbmb/MiniCPM-o-2_6"),
    ],
)
def test_image_processor_for_different_models(model_ids):
    first_model_id, second_model_id = model_ids

    first_ctx = build_model_context(
        first_model_id,
        limit_mm_per_prompt={"image": 1},
    )
    first_processor = MULTIMODAL_REGISTRY.create_processor(first_ctx.model_config)
    first_image_processor = first_processor.info.get_image_processor()

    second_ctx = build_model_context(
        second_model_id,
        limit_mm_per_prompt={"image": 1},
    )
    second_processor = MULTIMODAL_REGISTRY.create_processor(second_ctx.model_config)
    second_image_processor = second_processor.info.get_image_processor()

    second_processor.info.get_sliced_grid(ImageSize(width=128, height=128))

    assert type(first_image_processor) is not type(second_image_processor)
    assert (
        type(first_image_processor).__module__
        != type(second_image_processor).__module__
    )


@pytest.mark.parametrize("model_id", ["openbmb/MiniCPM-V-4"])
def test_prompt_has_dif_BPE_boundaries_in_context(model_id: str):
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokenizer = ctx.get_tokenizer()

    messages = [
        {"role": "user", "content": "(<image>./</image>)\nWhat is in this image?"}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image = np.zeros((768, 1024, 3), dtype=np.uint8)

    mm_items = processor.info.parse_mm_data({"image": [image]})
    processed = processor(
        prompt,
        mm_items=mm_items,
        hf_processor_mm_kwargs={},
    )
    image_placeholders = processed["mm_placeholders"].get("image", [])
    assert len(image_placeholders) == 1
    assert image_placeholders[0].length > 0
