# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Video placeholder accounting for GLM-5.3-Flash.

``Glm4vProcessingInfo._construct_video_placeholder`` emits one frame of
placeholders per timestamp, so the timestamps returned by
``_get_video_second_idx_glm46v`` decide how many placeholders the prompt gets
while ``video_grid_thw`` decides how many rows the vision tower produces. If
the two disagree they collide in ``_merge_multimodal_embeddings``, which raises
inside a worker and takes the engine down with it.

The checks below are arithmetic: no weights, no GPU.
"""

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.processors.glm5next import (
    Glm5NextVideoProcessor,
    _pixel_budget,
    glm_sample_frame_indices,
    smart_resize,
)

from ...utils import build_model_context


@pytest.fixture(scope="module")
def processor():
    ctx = build_model_context(
        "zai-org/GLM-5.3-Flash",
        limit_mm_per_prompt={"video": 1},
    )
    return MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config,
        tokenizer=ctx.tokenizer,
    )


def _pixel_path_grid(
    video_processor: Glm5NextVideoProcessor,
    num_frames: int,
    height: int,
    width: int,
) -> tuple[int, int, int]:
    """The ``video_grid_thw`` ``Glm5NextVideoProcessor._preprocess`` builds."""
    min_pixels, max_pixels = _pixel_budget(
        video_processor.min_image_tokens,
        video_processor.max_image_tokens,
        video_processor.patch_size,
        video_processor.merge_size,
        video_processor.temporal_patch_size,
    )
    factor = (
        video_processor.patch_size
        * video_processor.merge_size
        * video_processor.patch_expand_factor
    )
    resized_height, resized_width = smart_resize(
        t=num_frames,
        h=height,
        w=width,
        t_factor=video_processor.temporal_patch_size,
        h_factor=factor,
        w_factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    padded_frames = num_frames + (-num_frames % video_processor.temporal_patch_size)
    return (
        padded_frames // video_processor.temporal_patch_size,
        resized_height // video_processor.patch_size,
        resized_width // video_processor.patch_size,
    )


@pytest.mark.parametrize(
    ("total_num_frames", "fps", "duration", "height", "width", "expected_grid"),
    [
        # 4 s at 8 fps: the GLM-4.6V sampler asks for 3x as many timestamps.
        (32, 8.0, 4.0, 480, 640, (4, 36, 46)),
        # 1080p, 20 s: same factor of 3 at a full-size canvas.
        (600, 30.0, 20.0, 1080, 1920, (20, 58, 102)),
        # 1080p, 60 s: the one duration window where the two samplers agree
        # anyway -- a regression guard, the count must not move.
        (1800, 30.0, 60.0, 1080, 1920, (60, 34, 58)),
        # Past 300 s the GLM-4.6V sampler asks for half as many instead.
        (9030, 30.0, 301.0, 720, 1280, (301, 14, 26)),
    ],
)
def test_video_placeholders_match_encoder_rows(
    processor,
    total_num_frames: int,
    fps: float,
    duration: float,
    height: int,
    width: int,
    expected_grid: tuple[int, int, int],
):
    info = processor.info
    video_processor = info.get_video_processor()

    frame_indices = glm_sample_frame_indices(
        total_num_frames,
        fps,
        duration,
        target_fps=video_processor.fps_interval,
        max_frame_count=video_processor.max_frame_count_dynamic,
        temporal_patch_size=video_processor.temporal_patch_size,
    )
    grid_t, grid_h, grid_w = _pixel_path_grid(
        video_processor, len(frame_indices), height, width
    )
    assert (grid_t, grid_h, grid_w) == expected_grid

    timestamps = info._get_video_second_idx_glm46v(
        {
            "total_num_frames": total_num_frames,
            "fps": fps,
            "duration": duration,
            "do_sample_frames": True,
        },
        total_num_frames,
    )

    merge_length = video_processor.merge_size**2
    tokens_per_frame = grid_h * grid_w // merge_length
    encoder_rows = grid_t * grid_h * grid_w // merge_length

    assert len(timestamps) == grid_t
    assert len(timestamps) * tokens_per_frame == encoder_rows
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == 0
    assert timestamps[-1] <= duration


def test_video_placeholders_match_encoder_rows_when_presampled(processor):
    """The loader may pre-sample and hand the frames over as they are."""
    info = processor.info
    video_processor = info.get_video_processor()

    num_frames = 32
    grid_t, _, _ = _pixel_path_grid(video_processor, num_frames, 480, 640)

    timestamps = info._get_video_second_idx_glm46v(
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


def test_video_shorter_than_one_sampling_interval_is_rejected(processor):
    """A clip the sampler cannot pick a single frame from is a bad request."""
    with pytest.raises(ValueError, match="selected no frames"):
        processor.info._get_video_second_idx_glm46v(
            {
                "total_num_frames": 2,
                "fps": 8.0,
                "duration": 0.25,
                "do_sample_frames": True,
            },
            2,
        )
