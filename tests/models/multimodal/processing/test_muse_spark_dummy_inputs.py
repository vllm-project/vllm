# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse-Spark-Open dummy video inputs must reflect the model's frame budget.

Regression test: the profiling dummy hardcoded `num_frames=4` while the
model samples up to `num_frames_per_video` (600 by default) frames per
video. Encoder profiling therefore measured a 150x smaller activation than a
real maximum-length video request, and a single 600-frame video OOMed every
TP rank at default `gpu-memory-utilization`. The dummy now carries the
model's per-video frame cap, matching the token budget that
`get_mm_max_tokens_per_item` already declares for video regardless of
`seq_len` (which is clamped to `max_num_batched_tokens` under chunked
prefill and must not shrink the per-item dummy).

Tests use small caps to stay light; the default cap is asserted separately.
"""

from dataclasses import replace
from unittest.mock import MagicMock

from vllm.models.muse_spark.common.processor import (
    MuseSparkOmniDummyInputsBuilder,
    MuseSparkOmniProcessingInfo,
    PreprocessConfig,
)
from vllm.multimodal.processing.dummy_inputs import VideoDummyOptions


def _builder(config: PreprocessConfig) -> MuseSparkOmniDummyInputsBuilder:
    info = MagicMock(spec=MuseSparkOmniProcessingInfo)
    info.preprocess_config = config
    return MuseSparkOmniDummyInputsBuilder(info)


def test_muse_spark_dummy_video_uses_model_frame_cap():
    """The dummy's frames follow the config's cap, even at a small seq_len."""
    capped = replace(PreprocessConfig(), num_frames_per_video=8)
    data = _builder(capped).get_dummy_mm_data(
        seq_len=8192, mm_counts={"video": 2}, mm_options={}
    )

    videos = data["video"]
    assert isinstance(videos, list)
    assert len(videos) == 2
    for video, meta in videos:
        assert video.shape[0] == 8
        assert meta["total_num_frames"] == 8
        assert meta["frames_indices"] == list(range(8))
        assert meta["fps"] == 1.0


def test_muse_spark_dummy_video_override_can_reduce():
    """The VideoDummyOptions override still reduces the dummy (min clamp)."""
    capped = replace(PreprocessConfig(), num_frames_per_video=8)
    data = _builder(capped).get_dummy_mm_data(
        seq_len=1048576,
        mm_counts={"video": 1},
        mm_options={"video": VideoDummyOptions(num_frames=4)},
    )

    videos = data["video"]
    assert isinstance(videos, list)
    video, _ = videos[0]
    assert video.shape[0] == 4


def test_muse_spark_default_frame_cap_is_600():
    """The model's default per-video frame cap (the documented worst case)."""
    assert PreprocessConfig().num_frames_per_video == 600
