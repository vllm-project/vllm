# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MiMo-V2 Omni video timestamp construction.

Regression tests for https://github.com/vllm-project/vllm/issues/65523:
the processor used to label sampled frames with ``arange(T) / 1 fps``, so
every clip's injected timestamps read 00:00..00:30 regardless of the real
frame times. The timestamps must come from the video loader metadata
(``frames_indices / fps``).
"""

from typing import Any

import numpy as np
import pytest

from vllm.config.multimodal import MultiModalDummyOptions, VideoDummyOptions
from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context


def _build_video_mm_data(
    num_frames: int,
    fps: float,
    frames_indices: list[int] | None,
) -> dict[str, Any]:
    """A (video, metadata) item as produced by the video loader.

    ``frames_indices=None`` omits the field entirely (fallback case).
    """
    video = np.zeros((num_frames, 56, 56, 3), dtype=np.uint8)
    total = num_frames if frames_indices is None else frames_indices[-1] + 1
    metadata: dict[str, Any] = {
        "fps": fps,
        "duration": total / fps,
        "total_num_frames": total,
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    if frames_indices is not None:
        metadata["frames_indices"] = frames_indices
    return {"video": [(video, metadata)]}


def _get_temporal_stride(processor) -> int:
    p = processor.info.get_hf_processor().mimo_processor
    return p.temporal_patch_size * p.temporal_compression_ratio


@pytest.mark.parametrize("model_id", ["XiaomiMiMo/MiMo-V2.5-Omni"])
def test_video_timestamps_use_frame_media_time(model_id: str) -> None:
    ctx = build_model_context(model_id, limit_mm_per_prompt={"image": 0, "video": 1})
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokenizer = processor.info.get_tokenizer()

    stride = _get_temporal_stride(processor)

    # A 120 s @ 25 fps clip uniformly sampled to 16 frames (8 s apart).
    fps = 25.0
    num_frames = 16
    frames_indices = [i * 200 for i in range(num_frames)]
    mm_data = _build_video_mm_data(num_frames, fps, frames_indices)

    prompt = "<|vision_start|><|video_pad|><|vision_end|>"
    processed = processor(prompt, mm_items=processor.info.parse_mm_data(mm_data))

    item = processed["mm_kwargs"]["video"][0]
    assert item is not None
    data = item.get_data()

    # The real sampling interval is 200/25 = 8 s between frames, so
    # second_per_grid_ts is 16 s for a temporal patch of 2 frames - not 2 s
    # as produced by the old hardcoded 1 fps timestamps.
    temporal_patch_size = (
        processor.info.get_hf_processor().mimo_processor.temporal_patch_size
    )
    spg = float(data["second_per_grid_ts"])
    assert spg == pytest.approx(temporal_patch_size * 200 / fps)
    assert float(data["video_start_times"]) == pytest.approx(frames_indices[0] / fps)

    # The injected timestamp labels carry the real media time. With the old
    # 1 fps timestamps every label stayed within 00:00-00:15 for this clip.
    text = tokenizer.decode(processed["prompt_token_ids"])
    n_labels = num_frames // stride
    for j in range(1, n_labels):
        seconds = int(j * stride * 200 / fps)
        label = f"{seconds // 60:02d}:{seconds % 60:02d}"
        assert label in text, f"missing timestamp label {label} in prompt"


@pytest.mark.parametrize("model_id", ["XiaomiMiMo/MiMo-V2.5-Omni"])
def test_video_timestamps_fall_back_to_1fps_without_frame_indices(
    model_id: str,
) -> None:
    ctx = build_model_context(model_id, limit_mm_per_prompt={"image": 0, "video": 1})
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    temporal_patch_size = (
        processor.info.get_hf_processor().mimo_processor.temporal_patch_size
    )

    num_frames = 16
    mm_data = _build_video_mm_data(num_frames, fps=25.0, frames_indices=None)

    prompt = "<|vision_start|><|video_pad|><|vision_end|>"
    processed = processor(prompt, mm_items=processor.info.parse_mm_data(mm_data))

    item = processed["mm_kwargs"]["video"][0]
    assert item is not None
    spg = float(item.get_data()["second_per_grid_ts"])
    assert spg == pytest.approx(temporal_patch_size / 1.0)


@pytest.mark.parametrize("model_id", ["XiaomiMiMo/MiMo-V2.5-Omni"])
def test_dummy_videos_keep_1fps_timestamps(model_id: str) -> None:
    """Profiling dummies keep 1 fps metadata so memory profiling is unchanged."""
    ctx = build_model_context(model_id, limit_mm_per_prompt={"image": 0, "video": 1})
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    temporal_patch_size = (
        processor.info.get_hf_processor().mimo_processor.temporal_patch_size
    )

    dummy_inputs = processor.get_dummy_inputs(
        seq_len=8192,
        mm_counts={"video": 1},
        mm_options=MultiModalDummyOptions(
            {"video": VideoDummyOptions(num_frames=16, width=56, height=56)}
        ),
    )
    processed = processor(
        dummy_inputs.prompt,
        mm_items=dummy_inputs.mm_data_items,
    )

    item = processed["mm_kwargs"]["video"][0]
    assert item is not None
    spg = float(item.get_data()["second_per_grid_ts"])
    assert spg == pytest.approx(temporal_patch_size / 1.0)
