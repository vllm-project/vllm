# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Video placeholder accounting for GLM-5.3-Flash.

``Glm4vProcessingInfo._construct_video_placeholder`` emits one frame of
placeholders per timestamp, so the timestamps returned by
``_get_video_second_idx_glm46v`` decide how many placeholders the prompt gets
while ``video_grid_thw`` decides how many rows the vision tower produces. If
the two disagree they collide in ``_merge_multimodal_embeddings``, which raises
inside a worker and takes the engine down with it.

The checks below are arithmetic: no checkpoint, no GPU.
"""

from unittest.mock import Mock

import pytest

from vllm.models.glm5next.nvidia.multimodal import Glm5NextProcessingInfo
from vllm.transformers_utils.processors.glm5next import (
    Glm5NextVideoProcessor,
    _pixel_budget,
    glm_sample_frame_indices,
    smart_resize,
)


def _video_processor() -> Mock:
    """The video processor's class defaults, without loading a checkpoint."""
    processor = Mock(spec=Glm5NextVideoProcessor)
    for attr in (
        "fps_interval",
        "max_frame_count_dynamic",
        "max_image_tokens",
        "merge_size",
        "min_image_tokens",
        "patch_expand_factor",
        "patch_size",
        "temporal_patch_size",
    ):
        setattr(processor, attr, getattr(Glm5NextVideoProcessor, attr))
    return processor


def _processing_info(processor: Mock) -> Mock:
    info = Mock(spec=Glm5NextProcessingInfo)
    info.get_video_processor.return_value = processor
    return info


def _pixel_path_grid(
    processor: Mock,
    num_frames: int,
    height: int,
    width: int,
) -> tuple[int, int, int]:
    """The ``video_grid_thw`` ``Glm5NextVideoProcessor._preprocess`` builds."""
    min_pixels, max_pixels = _pixel_budget(
        processor.min_image_tokens,
        processor.max_image_tokens,
        processor.patch_size,
        processor.merge_size,
        processor.temporal_patch_size,
    )
    factor = processor.patch_size * processor.merge_size * processor.patch_expand_factor
    resized_height, resized_width = smart_resize(
        t=num_frames,
        h=height,
        w=width,
        t_factor=processor.temporal_patch_size,
        h_factor=factor,
        w_factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    padded_frames = num_frames + (-num_frames % processor.temporal_patch_size)
    return (
        padded_frames // processor.temporal_patch_size,
        resized_height // processor.patch_size,
        resized_width // processor.patch_size,
    )


@pytest.mark.parametrize(
    ("total_num_frames", "fps", "duration", "height", "width", "expected_grid"),
    [
        # 4 s at 8 fps: the GLM-4.6V sampler asks for 3x as many timestamps.
        (32, 8.0, 4.0, 480, 640, (4, 36, 46)),
        # 1080p, 20 s: same factor of 3 at a full-size canvas.
        (600, 30.0, 20.0, 1080, 1920, (20, 78, 138)),
        # 1080p, 60 s: the one duration window where the two samplers agree
        # anyway -- a regression guard, the count must not move.
        (1800, 30.0, 60.0, 1080, 1920, (60, 78, 138)),
        # Past 300 s the GLM-4.6V sampler asks for half as many instead.
        (9030, 30.0, 301.0, 720, 1280, (301, 42, 74)),
    ],
)
def test_video_placeholders_match_encoder_rows(
    total_num_frames: int,
    fps: float,
    duration: float,
    height: int,
    width: int,
    expected_grid: tuple[int, int, int],
):
    processor = _video_processor()
    info = _processing_info(processor)

    frame_indices = glm_sample_frame_indices(
        total_num_frames,
        fps,
        duration,
        target_fps=processor.fps_interval,
        max_frame_count=processor.max_frame_count_dynamic,
        temporal_patch_size=processor.temporal_patch_size,
    )
    grid_t, grid_h, grid_w = _pixel_path_grid(
        processor, len(frame_indices), height, width
    )
    assert (grid_t, grid_h, grid_w) == expected_grid

    timestamps = Glm5NextProcessingInfo._get_video_second_idx_glm46v(
        info,
        {
            "total_num_frames": total_num_frames,
            "fps": fps,
            "duration": duration,
            "do_sample_frames": True,
        },
        total_num_frames,
    )

    merge_length = processor.merge_size**2
    tokens_per_frame = grid_h * grid_w // merge_length
    encoder_rows = grid_t * grid_h * grid_w // merge_length

    assert len(timestamps) == grid_t
    assert len(timestamps) * tokens_per_frame == encoder_rows
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == 0
    assert timestamps[-1] <= duration


def test_video_placeholders_match_encoder_rows_when_presampled():
    """The loader may pre-sample and hand the frames over as they are."""
    processor = _video_processor()
    info = _processing_info(processor)

    num_frames = 32
    grid_t, _, _ = _pixel_path_grid(processor, num_frames, 480, 640)

    timestamps = Glm5NextProcessingInfo._get_video_second_idx_glm46v(
        info,
        {
            "total_num_frames": 256,
            "fps": 8.0,
            "duration": 32.0,
            "do_sample_frames": False,
            "frames_indices": list(range(0, 256, 256 // num_frames)),
        },
        num_frames,
    )

    assert len(timestamps) == grid_t


def test_video_shorter_than_one_sampling_interval_is_rejected():
    """A clip the sampler cannot pick a single frame from is a bad request."""
    processor = _video_processor()
    info = _processing_info(processor)

    with pytest.raises(ValueError, match="selected no frames"):
        Glm5NextProcessingInfo._get_video_second_idx_glm46v(
            info,
            {
                "total_num_frames": 2,
                "fps": 8.0,
                "duration": 0.25,
                "do_sample_frames": True,
            },
            2,
        )
