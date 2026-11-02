# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.multimodal.video import (
    DynamicVideoBackend,
    VideoSourceMetadata,
    VideoTargetMetadata,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.mark.parametrize(
    "source",
    [
        VideoSourceMetadata(total_frames_num=0, original_fps=30, duration=1),
        VideoSourceMetadata(total_frames_num=10, original_fps=0, duration=1),
        VideoSourceMetadata(total_frames_num=10, original_fps=30, duration=0),
    ],
)
def test_dynamic_video_sampling_rejects_invalid_source_metadata(
    source: VideoSourceMetadata,
):
    target = VideoTargetMetadata(num_frames=-1, fps=1, max_duration=10)

    with pytest.raises(ValueError, match="video .*metadata"):
        DynamicVideoBackend.compute_frames_index_to_sample(source, target)


@pytest.mark.parametrize(
    "target",
    [
        VideoTargetMetadata(num_frames=-1, fps=0, max_duration=10),
        VideoTargetMetadata(num_frames=-1, fps=1, max_duration=0),
    ],
)
def test_dynamic_video_sampling_rejects_invalid_target_metadata(
    target: VideoTargetMetadata,
):
    source = VideoSourceMetadata(total_frames_num=10, original_fps=30, duration=1)

    with pytest.raises(ValueError, match="video .*metadata"):
        DynamicVideoBackend.compute_frames_index_to_sample(source, target)


def test_dynamic_video_sampling_keeps_one_frame_for_short_video():
    source = VideoSourceMetadata(total_frames_num=10, original_fps=30, duration=0.1)
    target = VideoTargetMetadata(num_frames=-1, fps=1, max_duration=10)

    assert DynamicVideoBackend.compute_frames_index_to_sample(source, target) == [0]
