# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import numpy as np
import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context

MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
VIDEO_PROMPT = "<|vision_bos|><|VIDEO|><|vision_eos|>"


def _create_video_processor(
    max_model_len: int,
    mm_processor_kwargs: dict[str, Any] | None = None,
):
    ctx = build_model_context(
        MODEL_ID,
        model_config_kwargs={"max_model_len": max_model_len},
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"audio": 0, "image": 0, "video": 1},
    )
    return MULTIMODAL_REGISTRY.create_processor(ctx.model_config)


def _num_video_embeds(processed) -> int:
    return sum(p.get_num_embeds() for p in processed["mm_placeholders"]["video"])


@pytest.mark.parametrize("max_model_len", [20000, 24000])
def test_video_budget_reaches_max_model_len(max_model_len: int) -> None:
    processor = _create_video_processor(max_model_len)
    budget = processor.info.get_mm_max_tokens_per_item(max_model_len, {"video": 1})
    assert budget["video"] == max_model_len


def test_fps_sampled_video_fits_video_budget() -> None:
    max_model_len = 24000
    processor = _create_video_processor(max_model_len)
    video = np.zeros((60, 504, 896, 3), dtype=np.uint8)
    processed = processor(
        VIDEO_PROMPT,
        mm_items=processor.info.parse_mm_data({"video": [video]}),
        hf_processor_mm_kwargs={
            "videos_kwargs": {"fps": 2.0, "do_sample_frames": False}
        },
    )
    budget = processor.info.get_mm_max_tokens_per_item(max_model_len, {"video": 1})
    assert len(processed["prompt_token_ids"]) <= max_model_len
    assert _num_video_embeds(processed) == 17280
    assert _num_video_embeds(processed) <= budget["video"]


@pytest.mark.parametrize("max_model_len", [4096, 16384, 24000])
def test_dummy_video_fills_video_budget(max_model_len: int) -> None:
    processor = _create_video_processor(max_model_len)
    budget = processor.info.get_mm_max_tokens_per_item(max_model_len, {"video": 1})
    num_embeds = _num_video_embeds(processor.get_dummy_mm_inputs({"video": 1}))
    assert budget["video"] - 768 < num_embeds <= budget["video"]


@pytest.mark.parametrize(
    "mm_processor_kwargs",
    [
        None,
        {"max_pixels": 256 * 28 * 28},
        {
            "videos_kwargs": {
                "size": {
                    "shortest_edge": 128 * 28 * 28,
                    "longest_edge": 256 * 28 * 28,
                }
            }
        },
    ],
)
def test_video_canvas_matches_processed_frame_budget(
    mm_processor_kwargs: dict[str, Any] | None,
) -> None:
    processor = _create_video_processor(16384, mm_processor_kwargs)
    canvas = processor.info.get_video_size_with_most_features()
    video = np.zeros((2, canvas.height * 2, canvas.width * 2, 3), dtype=np.uint8)
    processed = processor(
        VIDEO_PROMPT,
        mm_items=processor.info.parse_mm_data({"video": [video]}),
        hf_processor_mm_kwargs={"videos_kwargs": {"do_sample_frames": False}},
    )
    assert _num_video_embeds(processed) == canvas.width * canvas.height // (28 * 28)


def test_image_budget_is_unchanged() -> None:
    ctx = build_model_context(
        MODEL_ID,
        limit_mm_per_prompt={"audio": 0, "image": 1, "video": 0},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    budget = processor.info.get_mm_max_tokens_per_item(4096, {"image": 1})
    assert budget is not None
    assert budget["image"] == 16384
