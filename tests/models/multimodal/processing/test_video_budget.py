# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import get_mm_max_toks_per_item
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs

from ...utils import build_model_context


@pytest.mark.parametrize(
    ("model_id", "max_model_len"),
    [
        ("zai-org/GLM-4.1V-9B-Thinking", 16384),
        ("AIDC-AI/Ovis2.5-2B", 32768),
        ("OpenGVLab/InternVL3-1B", 16384),
        ("internlm/Intern-S1", 16384),
        ("baidu/ERNIE-4.5-VL-28B-A3B-PT", 32768),
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


ERNIE = "baidu/ERNIE-4.5-VL-28B-A3B-PT"


def _video_embeds(outputs) -> int:
    return sum(p.get_num_embeds() for p in outputs["mm_placeholders"]["video"])


@pytest.mark.parametrize(
    ("hint_width", "hint_height", "width", "height", "mm_processor_kwargs"),
    [
        (1288, 728, 1288, 728, None),
        (728, 1288, 728, 1288, None),
        (1288, None, 1288, 728, None),
        (None, 1288, 728, 1288, None),
        (1920, None, 1288, 728, None),
        (1920, 1080, 1288, 728, None),
        (896, None, 896, 448, {"video_min_pixels": 401408, "video_max_pixels": 401408}),
    ],
)
def test_ernie45_vl_dummy_video_follows_size_hint(
    hint_width: int | None,
    hint_height: int | None,
    width: int,
    height: int,
    mm_processor_kwargs: dict | None,
):
    hint = {"count": 1, "num_frames": 2, "width": hint_width, "height": hint_height}
    ctx = build_model_context(
        ERNIE,
        model_config_kwargs={"max_model_len": 32768},
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={
            "image": 1,
            "video": {k: v for k, v in hint.items() if v is not None},
        },
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    frames = np.zeros((2, height, width, 3), dtype=np.uint8)
    metadata = {
        "fps": 2.0,
        "duration": 1.0,
        "total_num_frames": 2,
        "frames_indices": [0, 1],
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    inputs = ProcessorInputs(
        prompt=processor.info.get_tokenizer().encode("<|video@placeholder|>"),
        mm_data_items=processor.info.parse_mm_data({"video": [(frames, metadata)]}),
    )
    real = processor.apply(inputs, TimingContext(enabled=False))
    dummy = processor.get_dummy_mm_inputs({"video": 1})

    assert _video_embeds(dummy) == _video_embeds(real)


def test_ernie45_vl_video_budget_ignores_image_pixel_limit():
    ctx = build_model_context(
        ERNIE,
        model_config_kwargs={"max_model_len": 2048},
        mm_processor_kwargs={"max_pixels": 153664},
        limit_mm_per_prompt={"image": 1, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    budget = get_mm_max_toks_per_item(processor, {"image": 1, "video": 1}, None)
    dummy = processor.get_dummy_mm_inputs({"video": 1})

    assert budget["video"] == _video_embeds(dummy)
