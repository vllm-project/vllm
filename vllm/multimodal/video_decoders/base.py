# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal, NamedTuple

from vllm import envs

PYNVVIDEOCODEC_VIDEO_BACKEND: Literal["pynvvideocodec"] = "pynvvideocodec"


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
