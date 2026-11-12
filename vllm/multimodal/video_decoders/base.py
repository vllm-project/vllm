# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
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
    """Reject video frames exceeding `VLLM_MAX_IMAGE_PIXELS`."""
    max_pixels = envs.VLLM_MAX_IMAGE_PIXELS
    if max_pixels > 0 and width * height > max_pixels:
        raise ValueError(
            f"Video frame dimensions {width}x{height} "
            f"({width * height} pixels) exceed the maximum of "
            f"{max_pixels} pixels. Set VLLM_MAX_IMAGE_PIXELS to "
            f"increase this limit."
        )


class SampledFramesCharge:
    """Charges the decoded size of sampled video frames to a request budget.

    `VideoMediaIO` installs one with `charging_sampled_frames` while a loader
    runs. Decoders call `charge_sampled_frames` once they know the frame
    size and the sampled indices, so an oversized sample is rejected before
    any frame is decoded.
    """

    def __init__(self, charge: Callable[[int], None]) -> None:
        self._charge = charge
        self.charged_bytes = 0

    def __call__(self, num_bytes: int) -> None:
        self._charge(num_bytes)
        self.charged_bytes += num_bytes


_sampled_frames_charge: ContextVar[SampledFramesCharge | None] = ContextVar(
    "_sampled_frames_charge", default=None
)


@contextmanager
def charging_sampled_frames(charge: SampledFramesCharge | None) -> Iterator[None]:
    """Route `charge_sampled_frames` calls made in this context to `charge`."""
    token = _sampled_frames_charge.set(charge)
    try:
        yield
    finally:
        _sampled_frames_charge.reset(token)


def charge_sampled_frames(num_frames: int, width: int, height: int) -> None:
    """Charge `num_frames` RGB frames of `width` x `height` before decoding.

    Raises `ValueError` when the frames would exceed the per-request
    decoded-media budget (`VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES`).
    """
    charge = _sampled_frames_charge.get()
    if charge is not None:
        charge(max(0, num_frames) * max(0, width) * max(0, height) * 3)
