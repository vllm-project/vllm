# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for MiniCPMV's multimodal preprocessing."""

import numpy as np
import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

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
    "model_ids", [("openbmb/MiniCPM-Llama3-V-2_5", "openbmb/MiniCPM-V-4")]
)
def test_image_processor_for_dif_model(model_ids):
    model_id_25, model_id_4 = model_ids

    ctx_25 = build_model_context(model_id_25, limit_mm_per_prompt={"image": 1})
    processor_25 = MULTIMODAL_REGISTRY.create_processor(ctx_25.model_config)
    image_processor_25 = processor_25.info.get_image_processor()

    ctx_4 = build_model_context(model_id_4, limit_mm_per_prompt={"image": 1})
    processor_4 = MULTIMODAL_REGISTRY.create_processor(ctx_4.model_config)
    image_processor_4 = processor_4.info.get_image_processor()

    assert type(image_processor_25) is not type(image_processor_4)
    assert type(image_processor_25).__module__ != type(image_processor_4).__module__


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
