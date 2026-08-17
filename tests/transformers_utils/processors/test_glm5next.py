# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the vLLM-native GLM-5-Next multimodal processor.

Expected ``smart_resize`` outputs are hand-computed from the training-side
reference (``Glmga`` image processor / ``Glm5Next`` video processor in
transformers): sides shorter than the spatial factor are first upscaled to it,
and the shrink branch clamps to at least one factor so slim sides never
collapse to 0 under the pixel budget.
"""

import pytest
import torch
from PIL import Image
from transformers.image_utils import PILImageResampling, SizeDict
from transformers.video_utils import VideoMetadata

from vllm.transformers_utils.processors.glm5next import (
    Glm5NextImageProcessorFast,
    Glm5NextVideoProcessor,
    smart_resize,
)

PATCH_SIZE = 14
MERGE_SIZE = 2
PATCH_EXPAND_FACTOR = 2
FACTOR = PATCH_SIZE * MERGE_SIZE * PATCH_EXPAND_FACTOR  # 56
MIN_PIXELS = 112 * 112
MAX_PIXELS = 1_254_400  # serving cap applied in Glm5NextProcessor.from_pretrained


def resize(height: int, width: int, t: int = 2) -> tuple[int, int]:
    return smart_resize(
        t,
        height,
        width,
        t_factor=2,
        h_factor=FACTOR,
        w_factor=FACTOR,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )


@pytest.mark.parametrize(
    ("height", "width", "t", "expected"),
    [
        (300, 400, 2, (280, 392)),  # plain factor snap, no budget pressure
        (50, 80, 2, (56, 112)),  # both sides below factor get upscaled
        (57, 3, 2, (1064, 56)),  # slim side upscaled, aspect ratio kept
        (2160, 3840, 2, (560, 1008)),  # shrink under the serving cap
        (112, 11200, 2, (56, 7896)),  # slim side clamps to one factor, not 0
        (300, 400, 16, (224, 280)),  # frame count multiplies pixel pressure
        (112, 11200, 16, (56, 2800)),  # slim side stays positive for videos
    ],
)
def test_smart_resize_reference_values(height, width, t, expected):
    assert resize(height, width, t) == expected


def test_smart_resize_stays_snapped_and_positive():
    heights = (1, 27, 57, 111, 113, 300, 720, 2160)
    widths = (1, 27, 57, 111, 113, 400, 1280, 11200)
    for h in heights:
        for w in widths:
            if max(h, w) / min(h, w) > 200:
                continue
            for t in (2, 4, 16, 64):
                resized_h, resized_w = resize(h, w, t)
                assert resized_h > 0 and resized_w > 0
                assert resized_h % FACTOR == 0
                assert resized_w % FACTOR == 0


@pytest.mark.parametrize(
    ("height", "width", "t", "match"),
    [
        (100, 100, 1, "must be >= temporal_factor"),
        (0, 100, 2, "h or w is 0"),
        (100, 0, 2, "h or w is 0"),
        (100, 30000, 2, "aspect ratio"),
    ],
)
def test_smart_resize_rejects_degenerate_inputs(height, width, t, match):
    with pytest.raises(ValueError, match=match):
        resize(height, width, t)


@pytest.fixture(scope="module")
def image_processor():
    return Glm5NextImageProcessorFast(
        size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
    )


@pytest.mark.parametrize(
    ("height", "width"),
    [(50, 80), (300, 400), (2160, 3840), (112, 11200)],
)
def test_image_grid_matches_smart_resize(image_processor, height, width):
    out = image_processor(Image.new("RGB", (width, height)), return_tensors="pt")
    resized_h, resized_w = resize(height, width)
    grid = out["image_grid_thw"][0].tolist()
    assert grid == [1, resized_h // PATCH_SIZE, resized_w // PATCH_SIZE]
    assert out["pixel_values"].shape == (
        grid[1] * grid[2],
        3 * 2 * PATCH_SIZE * PATCH_SIZE,
    )


@pytest.mark.parametrize(
    ("height", "width"),
    [(50, 80), (300, 400), (2160, 3840), (112, 11200)],
)
def test_image_patch_count_matches_preprocess(image_processor, height, width):
    out = image_processor(Image.new("RGB", (width, height)), return_tensors="pt")
    grid = out["image_grid_thw"][0].tolist()
    expected = grid[1] * grid[2]
    assert image_processor.get_number_of_image_patches(height, width) == expected


@pytest.fixture(scope="module")
def video_processor():
    return Glm5NextVideoProcessor(
        size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
    )


def run_video_preprocess(video_processor, frames):
    return video_processor._preprocess(
        [frames],
        size=SizeDict(shortest_edge=MIN_PIXELS, longest_edge=MAX_PIXELS),
        resample=PILImageResampling.BICUBIC,
        image_mean=(0.48145466, 0.4578275, 0.40821073),
        image_std=(0.26862954, 0.26130258, 0.27577711),
        patch_size=PATCH_SIZE,
        temporal_patch_size=2,
        merge_size=MERGE_SIZE,
        patch_expand_factor=PATCH_EXPAND_FACTOR,
        return_tensors="pt",
    )


def test_video_preprocess_pads_odd_frame_count(video_processor):
    out = run_video_preprocess(video_processor, torch.rand(7, 3, 300, 400))
    grid = out["video_grid_thw"][0].tolist()
    assert grid == [4, 20, 28]  # 7 frames padded to 8 -> grid_t = 4
    assert out["pixel_values_videos"].shape == (
        4 * 20 * 28,
        3 * 2 * PATCH_SIZE * PATCH_SIZE,
    )


def test_video_preprocess_keeps_min_side_under_frame_budget(video_processor):
    out = run_video_preprocess(video_processor, torch.rand(16, 3, 112, 11200))
    assert out["video_grid_thw"][0].tolist() == [8, 4, 200]  # height 56, not 0


@pytest.mark.parametrize(
    ("total_frames", "fps", "duration", "expected_len"),
    [
        (900, 30.0, 30.0, 180),  # <= 30s -> 3 fps * temporal_patch_size 2
        (3000, 30.0, 100.0, 200),  # <= 300s -> 1 fps * temporal_patch_size 2
        (72000, 30.0, 2400.0, 640),  # > 300s -> 0.5 fps, capped at max_frames
    ],
)
def test_sample_frames_dynamic_fps(
    video_processor, total_frames, fps, duration, expected_len
):
    indices = video_processor.sample_frames(
        VideoMetadata(total_num_frames=total_frames, fps=fps, duration=duration)
    )
    assert len(indices) == expected_len
    assert len(indices) % 2 == 0
    assert indices[0] == 0
    assert indices[-1] < total_frames
