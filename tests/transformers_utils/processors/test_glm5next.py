# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the vLLM-native GLM-5.3-Flash multimodal processor."""

import math

import pytest
import torch
from PIL import Image
from transformers.image_utils import PILImageResampling
from transformers.video_utils import VideoMetadata

from vllm.transformers_utils.processors.glm5next import (
    Glm5NextImageProcessor,
    Glm5NextVideoProcessor,
    _get_pad_content_size,
    _resize_or_pad,
    glm_sample_frame_indices,
    smart_resize,
)

PATCH_SIZE = 14
MERGE_SIZE = 2
PATCH_EXPAND_FACTOR = 1  # checkpoint processor_config.json
FACTOR = PATCH_SIZE * MERGE_SIZE * PATCH_EXPAND_FACTOR  # 28
PIXELS_PER_TOKEN = 2 * (PATCH_SIZE * MERGE_SIZE) ** 2  # 1568
MIN_TOKENS = 16
MAX_TOKENS = 800  # serving-cap equivalent budget, keeps test canvases small
MIN_PIXELS = MIN_TOKENS * PIXELS_PER_TOKEN
MAX_PIXELS = MAX_TOKENS * PIXELS_PER_TOKEN


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
        (300, 400, 2, (308, 420)),  # ceil alignment, no budget pressure
        (50, 80, 2, (112, 168)),  # below min budget: proportional upscale
        (57, 3, 2, (504, 28)),  # slim side ceils to one factor
        (2160, 3840, 2, (588, 1064)),  # shrink: max canvas under the cap
        (112, 11200, 2, (84, 7420)),  # extreme aspect stays proportional
        (300, 400, 16, (252, 308)),  # frame count eats into the budget
        (112, 11200, 16, (28, 2800)),  # slim side stays positive, budget holds
    ],
)
def test_smart_resize_reference_values(height, width, t, expected):
    got = smart_resize(
        t,
        height,
        width,
        t_factor=2,
        h_factor=FACTOR,
        w_factor=FACTOR,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    assert got == expected
    # The aligned canvas never exceeds the budget it was fitted against.
    t_bar = max(2, round(t / 2) * 2)
    assert t_bar * got[0] * got[1] <= MAX_PIXELS


def test_smart_resize_stays_snapped_and_positive():
    heights = (1, 27, 57, 111, 113, 300, 720, 2160)
    widths = (1, 27, 57, 111, 113, 400, 1280, 11200)
    for h in heights:
        for w in widths:
            for t in (2, 4, 16, 64):
                resized_h, resized_w = resize(h, w, t)
                assert resized_h > 0 and resized_w > 0
                assert resized_h % FACTOR == 0
                assert resized_w % FACTOR == 0
                t_bar = max(2, round(t / 2) * 2)
                assert t_bar * resized_h * resized_w <= MAX_PIXELS


@pytest.mark.parametrize(
    ("height", "width", "match"),
    [
        (0, 100, "must be positive"),
        (100, 0, "must be positive"),
    ],
)
def test_smart_resize_rejects_degenerate_inputs(height, width, match):
    with pytest.raises(ValueError, match=match):
        resize(height, width, 2)


def test_smart_resize_rejects_inverted_budget():
    with pytest.raises(ValueError, match="min_pixels must be less than or equal"):
        smart_resize(
            2,
            100,
            100,
            t_factor=2,
            h_factor=FACTOR,
            w_factor=FACTOR,
            min_pixels=MAX_PIXELS,
            max_pixels=MIN_PIXELS,
        )


def test_get_pad_content_size():
    # Oversized content shrinks proportionally, never upscales by default.
    assert _get_pad_content_size(300, 400, 308, 420) == (300, 400)
    assert _get_pad_content_size(600, 800, 308, 420) == (308, 410)
    # allow_upscale enlarges small content toward the canvas.
    assert _get_pad_content_size(28, 28, 112, 112, allow_upscale=True) == (112, 112)


def test_resize_or_pad_pads_right_and_bottom():
    stacked = torch.rand(1, 3, 300, 400)

    def identity_resize(x, size, resample=None):
        assert (size.height, size.width) == (300, 400)
        return x

    padded = _resize_or_pad(stacked, 308, 420, "pad", None, identity_resize)
    assert padded.shape == (1, 3, 308, 420)
    torch.testing.assert_close(padded[..., :300, :400], stacked)
    assert padded[..., 300:, :].eq(0).all()
    assert padded[..., :, 400:].eq(0).all()

    def force_resize(x, size, resample=None):
        return torch.zeros(x.shape[0], x.shape[1], size.height, size.width)

    assert _resize_or_pad(stacked, 308, 420, "resize", None, force_resize).shape == (
        1,
        3,
        308,
        420,
    )
    with pytest.raises(ValueError, match="resize_mode"):
        _resize_or_pad(stacked, 308, 420, "crop", None, identity_resize)


@pytest.fixture(scope="module")
def image_processor():
    return Glm5NextImageProcessor(
        min_image_tokens=MIN_TOKENS, max_image_tokens=MAX_TOKENS
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
        min_image_tokens=MIN_TOKENS, max_image_tokens=MAX_TOKENS
    )


def run_video_preprocess(video_processor, frames, **overrides):
    kwargs = dict(
        resample=PILImageResampling.BICUBIC,
        image_mean=(0.48145466, 0.4578275, 0.40821073),
        image_std=(0.26862954, 0.26130258, 0.27577711),
        patch_size=PATCH_SIZE,
        temporal_patch_size=2,
        merge_size=MERGE_SIZE,
        patch_expand_factor=PATCH_EXPAND_FACTOR,
        return_tensors="pt",
    )
    kwargs.update(overrides)
    return video_processor._preprocess([frames], **kwargs)


def test_video_preprocess_pads_odd_frame_count(video_processor):
    out = run_video_preprocess(video_processor, torch.rand(7, 3, 300, 400))
    grid = out["video_grid_thw"][0].tolist()
    # 7 frames padded to 8 -> grid_t 4; canvas ceil-aligned (308, 420).
    assert grid == [4, 22, 30]
    assert out["pixel_values_videos"].shape == (
        4 * 22 * 30,
        3 * 2 * PATCH_SIZE * PATCH_SIZE,
    )


def test_video_preprocess_keeps_min_side_under_frame_budget(video_processor):
    out = run_video_preprocess(video_processor, torch.rand(16, 3, 112, 11200))
    # Budget-fitted canvas (28, 2800): 16 * 28 * 2800 == the pixel cap.
    assert out["video_grid_thw"][0].tolist() == [8, 2, 200]  # height 28, not 0


def test_video_preprocess_pads_content_not_distort(video_processor):
    frames = torch.zeros(4, 3, 300, 400)
    frames[..., 50, 60] = 1.0  # marker inside the content area
    # Bypass rescale/normalize so zero padding stays exactly zero.
    out = run_video_preprocess(
        video_processor, frames, do_rescale=False, do_normalize=False
    )
    # Pad mode keeps the 300x400 content aspect on the (308, 420) canvas
    # with zero padding on the right/bottom.
    grid = out["video_grid_thw"][0].tolist()
    assert grid == [2, 22, 30]
    patches = out["pixel_values_videos"].view(
        grid[0],
        grid[1] // MERGE_SIZE,
        grid[2] // MERGE_SIZE,
        MERGE_SIZE,
        MERGE_SIZE,
        3 * 2 * PATCH_SIZE * PATCH_SIZE,
    )
    frame = patches[0].permute(0, 3, 1, 4, 2).reshape(grid[1], grid[2], -1)
    # Patch columns fully right of the 400px content are pure padding.
    assert frame[:, math.ceil(400 / PATCH_SIZE) :].abs().max() == 0


@pytest.mark.parametrize(
    ("total_frames", "fps", "duration", "expected_len", "first", "last"),
    [
        # Dense source: the tp-scaled greedy overshoots extract_t, so the
        # fixup spreads picks uniformly across the whole video (linspace).
        (900, 30.0, 30.0, 60, 0, 899),
        (3000, 30.0, 100.0, 200, 0, 2999),
        # extract_t capped at 2048 frames.
        (72000, 30.0, 2400.0, 2048, 0, 71999),
        # Low container fps: the greedy picks every frame.
        (48, 2.0, 24.0, 48, 0, 47),
        # Duration derived from the frame count when metadata lacks it.
        (300, 25.0, 0, 26, 0, 299),
    ],
)
def test_glm_sample_frame_indices_behaviour(
    total_frames, fps, duration, expected_len, first, last
):
    indices = glm_sample_frame_indices(total_frames, fps, duration)
    assert len(indices) == expected_len
    assert indices[0] == first
    assert indices[-1] == last
    assert len(indices) % 2 == 0
    assert indices == sorted(indices)
    assert all(0 <= idx < total_frames for idx in indices)


def test_glm_sample_frame_indices_short_clip_floor_spread():
    # extract_t (20) > total (7): evenly spaced timestamps, deduplicated,
    # then pair-padded to an even count.
    assert glm_sample_frame_indices(7, 30.0, 10.0) == [0, 1, 2, 3, 4, 5, 6, 6]


@pytest.mark.parametrize(
    ("kwargs", "expected_len"),
    [
        ({"target_fps": 0.5}, 16),
        ({"max_frame_count": 16}, 16),
        ({"target_fps": 8}, 240),  # 30s * 8 = 240, under the 2048 cap
    ],
)
def test_glm_sample_frame_indices_request_overrides(kwargs, expected_len):
    indices = glm_sample_frame_indices(900, 30.0, 30.0, **kwargs)
    assert len(indices) == expected_len
    assert len(indices) % 2 == 0


def test_sample_frames_fps_interval(video_processor):
    indices = video_processor.sample_frames(
        VideoMetadata(total_num_frames=900, fps=30.0, duration=30.0)
    )
    assert len(indices) == 60
    assert indices[0] == 0
    assert indices[-1] == 899  # uniform spread reaches the final frame

    # Request overrides reach the sampler through both kwarg spellings. The
    # 15-frame spread is odd, so pair-padding duplicates the last frame.
    assert (
        len(
            video_processor.sample_frames(
                VideoMetadata(total_num_frames=900, fps=30.0, duration=30.0),
                fps=0.5,
            )
        )
        == 16
    )
    assert (
        len(
            video_processor.sample_frames(
                VideoMetadata(total_num_frames=900, fps=30.0, duration=30.0),
                target_fps=0.5,
            )
        )
        == 16
    )
    assert (
        len(
            video_processor.sample_frames(
                VideoMetadata(total_num_frames=900, fps=30.0, duration=30.0),
                max_frames=16,
            )
        )
        == 16
    )


def test_defaults_mirror_checkpoint_config():
    """Bare instantiation matches the checkpoint's ``processor_config.json``
    token-budget style defaults; ``size`` carries no budget."""
    image_processor = Glm5NextImageProcessor()
    assert image_processor.patch_expand_factor == 1
    assert image_processor.min_image_tokens == 16
    assert image_processor.max_image_tokens == 8000
    assert image_processor.resize_mode == "pad"

    video_processor = Glm5NextVideoProcessor()
    assert video_processor.patch_expand_factor == 1
    assert video_processor.min_image_tokens == 16
    assert video_processor.max_image_tokens == 240000
    assert video_processor.fps_interval == 2.0
    assert video_processor.max_frame_count_dynamic == 2048


def test_token_budgets_drive_geometry():
    """The token bounds fully determine the pixel budget and the grid."""
    token_proc = Glm5NextImageProcessor(min_image_tokens=64, max_image_tokens=512)
    img = Image.new("RGB", (400, 300))
    grid = token_proc(img, return_tensors="pt")["image_grid_thw"][0].tolist()
    # Budgets 100,352..802,816 px: the 300x400 canvas ceils to (308, 420)
    # without hitting either bound -> 22x30 patches.
    assert grid == [1, 22, 30]
    assert token_proc.get_number_of_image_patches(300, 400) == 22 * 30


def test_missing_token_budgets_rejected():
    proc = Glm5NextImageProcessor(min_image_tokens=None)
    with pytest.raises(ValueError, match="min_image_tokens"):
        proc(Image.new("RGB", (400, 300)), return_tensors="pt")


def test_video_config_fields_land():
    """fps_interval / max_frame_count_dynamic from the dedicated config
    shape sampling without any request overrides."""
    proc = Glm5NextVideoProcessor(
        fps_interval=4,
        max_frame_count_dynamic=32,
    )
    indices = proc.sample_frames(
        VideoMetadata(total_num_frames=900, fps=30.0, duration=30.0)
    )
    # 30s * 4 = 120 candidates, capped at 32 -> uniform spread of 32.
    assert len(indices) == 32
    assert indices[0] == 0
    assert indices[-1] == 899

    # Request overrides still win over the config values.
    assert (
        len(
            proc.sample_frames(
                VideoMetadata(total_num_frames=900, fps=30.0, duration=30.0),
                fps=0.5,
            )
        )
        == 16
    )
