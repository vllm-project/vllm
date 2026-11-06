# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal, NamedTuple

from vllm import envs

PYNVVIDEOCODEC_VIDEO_BACKEND: Literal["pynvvideocodec"] = "pynvvideocodec"
PYNVVIDEOCODEC_DEFAULT_HW_DECODERS = 2


class VideoTargetMetadata(NamedTuple):
    """Metadata describing the requested video sample."""

    num_frames: int
    fps: float
    max_duration: float


class VideoSourceMetadata(NamedTuple):
    """Metadata describing the encoded video source."""

    total_frames_num: int
    original_fps: float
    duration: float


def check_frame_pixel_limit(width: int, height: int) -> None:
    """Reject video frames exceeding ``VLLM_MAX_IMAGE_PIXELS``."""
    max_pixels = envs.VLLM_MAX_IMAGE_PIXELS
    if max_pixels > 0 and width * height > max_pixels:
        raise ValueError(
            f"Video frame dimensions {width}x{height} "
            f"({width * height} pixels) exceed the maximum of "
            f"{max_pixels} pixels. Set VLLM_MAX_IMAGE_PIXELS to "
            f"increase this limit."
        )


def check_video_decode_frame_limit(frames_to_walk: int) -> None:
    """Reject videos that would walk more than ``VLLM_MAX_VIDEO_DECODE_FRAMES``.

    Set the env var to ``0`` to disable.
    """
    max_frames = envs.VLLM_MAX_VIDEO_DECODE_FRAMES
    if max_frames > 0 and frames_to_walk > max_frames:
        raise ValueError(
            f"Video decode would walk {frames_to_walk} frames, which "
            f"exceeds the maximum of {max_frames} frames. Set "
            f"VLLM_MAX_VIDEO_DECODE_FRAMES to increase this limit."
        )
