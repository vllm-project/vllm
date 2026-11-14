# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import get_mm_max_toks_per_item

from ...utils import build_model_context


@pytest.mark.parametrize(
    ("model_id", "max_model_len"),
    [
        ("zai-org/GLM-4.1V-9B-Thinking", 16384),
        ("AIDC-AI/Ovis2.5-2B", 32768),
        ("OpenGVLab/InternVL3-1B", 16384),
        ("internlm/Intern-S1", 16384),
    ],
)
def test_video_budget_matches_profiled_video(model_id: str, max_model_len: int):
    ctx = build_model_context(
        model_id,
        model_config_kwargs={"max_model_len": max_model_len},
        limit_mm_per_prompt={"image": 1, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    budget = get_mm_max_toks_per_item(processor, {"image": 1, "video": 1}, None)
    dummy = processor.get_dummy_mm_inputs({"video": 1})
    num_embeds = sum(p.get_num_embeds() for p in dummy["mm_placeholders"]["video"])

    assert budget["video"] == num_embeds


@pytest.mark.parametrize(
    ("model_id", "max_model_len"),
    [("baidu/ERNIE-4.5-VL-28B-A3B-PT", 16384)],
)
def test_video_budget_ignores_image_count(model_id: str, max_model_len: int):
    ctx = build_model_context(
        model_id,
        model_config_kwargs={"max_model_len": max_model_len},
        limit_mm_per_prompt={"image": 1, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    with_image = get_mm_max_toks_per_item(processor, {"image": 1, "video": 1}, None)
    video_only = get_mm_max_toks_per_item(processor, {"video": 1}, None)

    assert with_image["video"] == video_only["video"]
