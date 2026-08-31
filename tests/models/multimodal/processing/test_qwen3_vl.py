# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for Qwen3-VL processor.

Covers the fix for num_frames-based timestamp calculation
(issue vllm-project/vllm#35909).
"""

from typing import Any

import numpy as np
import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def _build_video_mm_data(
    num_frames: int,
    width: int = 128,
    height: int = 128,
    original_fps: float = 30.0,
) -> dict[str, Any]:
    """Create synthetic video data with metadata indicating that
    HF processor should re-sample frames (do_sample_frames=True).

    ``total_num_frames`` is set equal to the ndarray frame count so
    that HF's ``sample_frames`` indices stay within bounds of the
    actual tensor that is passed."""
    video = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    metadata = {
        "fps": original_fps,
        "duration": num_frames / original_fps,
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
        "video_backend": "opencv",
        "do_sample_frames": True,
    }
    return {"video": [(video, metadata)]}


@pytest.mark.parametrize("model_id", [MODEL_ID])
@pytest.mark.parametrize(
    "num_frames",
    [8, 16],
)
def test_processor_num_frames_timestamp(
    model_id: str,
    num_frames: int,
) -> None:
    """Regression test: using ``num_frames`` (without ``fps``) must not
    cause a timestamp / token-count mismatch.

    Before the fix, ``_get_video_second_idx`` ignored the explicit
    ``num_frames`` and fell back to an fps-based calculation, which
    produced a different number of timestamp entries and ultimately led
    to shape mismatches in downstream token construction.

    We deliberately choose ``num_frames`` values (8, 16) that differ
    from what the default fps-based path would compute (which clamps
    to ``min_frames=4`` for a short video at 30 fps), so this test
    would fail without the fix.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    prompt = "<|vision_start|><|video_pad|><|vision_end|>"
    mm_data = _build_video_mm_data(num_frames=num_frames)

    # Process with explicit num_frames (no fps) -- this is the path
    # that was broken before the fix.
    hf_mm_kwargs: dict[str, Any] = {"num_frames": num_frames}
    processed = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=hf_mm_kwargs,
    )

    # Basic sanity: the processor must produce video tokens.
    token_ids = processed["prompt_token_ids"]
    assert len(token_ids) > 0, "Processor produced empty token list"

    # Verify that video placeholders were actually inserted.
    assert "mm_placeholders" in processed
    video_phs = processed["mm_placeholders"].get("video", [])
    assert len(video_phs) == 1, (
        f"Expected exactly 1 video placeholder, got {len(video_phs)}"
    )


@pytest.mark.parametrize("model_id", [MODEL_ID])
@pytest.mark.parametrize("num_videos", [2, 4])
def test_processor_multi_video(
    model_id: str,
    num_videos: int,
) -> None:
    """Verify that multi-video processing produces correct placeholders.

    This exercises the token-level replacement path in
    ``_apply_hf_processor_main`` which avoids the quadratic text-level
    prompt expansion.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 0, "video": num_videos},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    prompt = "<|vision_start|><|video_pad|><|vision_end|>" * num_videos
    mm_data = {"video": [_build_video_mm_data(num_frames=8)["video"][0]] * num_videos}

    processed = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={"num_frames": 8},
    )

    token_ids = processed["prompt_token_ids"]
    assert len(token_ids) > 0

    video_phs = processed["mm_placeholders"].get("video", [])
    assert len(video_phs) == num_videos, (
        f"Expected {num_videos} video placeholders, got {len(video_phs)}"
    )

    # All placeholders should have the same length (same video params)
    # and must not overlap.
    lengths = {ph.length for ph in video_phs}
    assert len(lengths) == 1, f"Placeholder lengths differ: {lengths}"
    for i in range(1, len(video_phs)):
        prev_end = video_phs[i - 1].offset + video_phs[i - 1].length
        assert video_phs[i].offset >= prev_end, (
            f"Placeholder {i} overlaps with placeholder {i - 1}"
        )


# Qwen3-VL / Qwen3.8 "Long Video Understanding" pixel budget from the
# model card. Used to check --mm-processor-kwargs scoping (#52834).
_LONG_VIDEO_SIZE = {"longest_edge": 469762048, "shortest_edge": 4096}


def _probe_mm_token_budgets(
    model_id: str, mm_processor_kwargs: dict[str, Any] | None
) -> tuple[int, int]:
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"image": 1, "video": 1},
    )
    info = MULTIMODAL_REGISTRY.create_processor(ctx.model_config).info
    video = info.get_max_video_tokens(
        seq_len=500000, mm_counts={"video": 1, "image": 1}
    )
    return video, info.get_max_image_tokens()


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("model_id", [MODEL_ID])
def test_processor_kwargs_videos_kwargs_does_not_leak_into_image_budget(
    model_id: str,
) -> None:
    """``videos_kwargs.size`` must raise only the video token budget.

    A flat ``size`` override still applies to both modalities (the previous
    shared-namespace behavior). Regression for #52834.
    """
    stock_video, stock_image = _probe_mm_token_budgets(model_id, None)
    scoped_video, scoped_image = _probe_mm_token_budgets(
        model_id, {"videos_kwargs": {"size": _LONG_VIDEO_SIZE}}
    )
    flat_video, flat_image = _probe_mm_token_budgets(
        model_id, {"size": _LONG_VIDEO_SIZE}
    )

    assert scoped_video == flat_video
    assert scoped_video > stock_video
    assert scoped_image == stock_image
    assert flat_image > stock_image


@pytest.mark.parametrize("model_id", [MODEL_ID])
@pytest.mark.parametrize(
    "hf_mm_kwargs",
    [{"num_frames": [8, 16]}, {"fps": [2.0, 4.0]}],
)
def test_processor_multi_video_list_kwargs(
    model_id: str,
    hf_mm_kwargs: dict[str, Any],
) -> None:
    """Regression test: a multi-video request with list-valued per-video
    ``mm_processor_kwargs`` (one ``fps``/``num_frames`` per video) must not
    crash.

    Before the fix, ``_apply_hf_processor_main`` copied the whole kwargs to every
    video without slicing, so ``_get_video_second_idx`` received the list
    where a scalar was expected and raised ``TypeError``.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 0, "video": 2},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    prompt = (
        "<|vision_start|><|video_pad|><|vision_end|>"
        "<|vision_start|><|video_pad|><|vision_end|>"
    )
    mm_data = {
        "video": [
            _build_video_mm_data(num_frames=16)["video"][0],
            _build_video_mm_data(num_frames=32)["video"][0],
        ]
    }

    processed = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=hf_mm_kwargs,
    )

    video_phs = processed["mm_placeholders"].get("video", [])
    assert len(video_phs) == 2, (
        f"Expected exactly 2 video placeholders, got {len(video_phs)}"
    )


@pytest.mark.parametrize("model_id", [MODEL_ID])
def test_dummy_video_spreads_budget_when_frame_cap_enabled(model_id: str) -> None:
    """Regression test for memory profiling with ``cap_pixels_per_frame``.

    With the HF per-frame pixel cap enabled (transformers#48071) and a
    raised video budget, a 2-frame profiling dummy would be processed at
    only 2 * cap pixels, underestimating the largest possible video (a
    fully sampled one still fills the whole ``longest_edge`` budget). The
    dummy builder must spread the budget over enough frames that the cap
    is not binding.
    """
    # 16 capped frames' worth of budget on top of the default per-frame
    # ceiling (max_video_tokens=768 at patch 16, merge 2 -> 786,432
    # pixels per frame).
    per_frame_cap = 768 * (16 * 2) ** 2
    budget = per_frame_cap * 16
    size = {"longest_edge": budget, "shortest_edge": 4096}

    capped_ctx = build_model_context(
        model_id,
        mm_processor_kwargs={"size": size, "cap_pixels_per_frame": True},
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    capped = MULTIMODAL_REGISTRY.create_processor(capped_ctx.model_config)
    capped_dummy = capped.dummy_inputs.get_dummy_mm_data(1024, {"video": 1}, {})
    capped_frames = capped_dummy["video"][0][0].shape[0]
    assert capped_frames == 16, (
        f"Expected the dummy to spread the budget over 16 frames, got {capped_frames}"
    )

    uncapped_ctx = build_model_context(
        model_id,
        mm_processor_kwargs={"size": size},
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    uncapped = MULTIMODAL_REGISTRY.create_processor(uncapped_ctx.model_config)
    uncapped_dummy = uncapped.dummy_inputs.get_dummy_mm_data(1024, {"video": 1}, {})
    uncapped_frames = uncapped_dummy["video"][0][0].shape[0]
    assert uncapped_frames == 2, (
        f"Expected the uncapped dummy to keep 2 frames, got {uncapped_frames}"
    )
