# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal

from .base import VideoSourceMetadata, VideoTargetMetadata, check_frame_pixel_limit
from .deepstream import DeepStreamVideoBackendMixin, decode_deepstream
from .opencv import OpenCVVideoBackendMixin, decode_opencv
from .pyav import PyAVVideoBackendMixin, decode_pyav
from .pynvvideocodec import (
    PYNVVIDEOCODEC_CUDA_CONTEXT_BYTES,
    PYNVVIDEOCODEC_DECODER_CACHE_SIZE,
    PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES,
    PYNVVIDEOCODEC_MAX_RETAINED_DECODERS,
    PYNVVIDEOCODEC_VIDEO_BACKEND,
    PyNvVideoCodecDecoderSlot,
    PyNvVideoCodecVideoBackendMixin,
    decode_pynvvideocodec,
)
from .torchcodec import TorchCodecVideoBackendMixin, decode_torchcodec

VideoDecoderBackend = Literal[
    "opencv", "pyav", "torchcodec", "pynvvideocodec", "deepstream"
]

_BACKEND_OPTION_DEFAULTS: dict[str, dict[str, Any]] = {
    "opencv": {},
    "pyav": {},
    "torchcodec": {
        "num_ffmpeg_threads": 0,
        "seek_mode": "exact",
    },
    "pynvvideocodec": {},
    "deepstream": {
        "pool_size": None,
        "timeout_sec": 120.0,
    },
}


def resolve_video_backend_kwargs(
    backend: str,
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split frame-sampling kwargs from options owned by a decoder backend."""
    try:
        defaults = _BACKEND_OPTION_DEFAULTS[backend]
    except KeyError:
        valid = ", ".join(repr(name) for name in _BACKEND_OPTION_DEFAULTS)
        raise ValueError(
            f"Unknown video codec backend {backend!r}; valid options: {valid}."
        ) from None

    sampling_kwargs = dict(kwargs)
    backend_kwargs = dict(defaults)
    backend_option_names = {
        name for options in _BACKEND_OPTION_DEFAULTS.values() for name in options
    }
    misplaced = (sampling_kwargs.keys() & backend_option_names) - defaults.keys()
    if misplaced:
        names = ", ".join(sorted(misplaced))
        raise ValueError(f"{names} is not supported by the {backend!r} backend")

    for name in defaults:
        if name in sampling_kwargs:
            backend_kwargs[name] = sampling_kwargs.pop(name)

    return sampling_kwargs, backend_kwargs


def decode_video(
    backend: str,
    loader_cls,
    data: bytes,
    target: VideoTargetMetadata,
    sampling_kwargs: dict[str, Any],
    backend_kwargs: dict[str, Any],
    *,
    frame_recovery: bool,
):
    """Decode a sampled video using the selected codec implementation."""
    if backend == "opencv":
        return decode_opencv(
            loader_cls,
            data,
            target,
            sampling_kwargs,
            frame_recovery=frame_recovery,
        )

    if backend == PYNVVIDEOCODEC_VIDEO_BACKEND and frame_recovery:
        raise ValueError(
            "frame_recovery is not supported for "
            f"`{PYNVVIDEOCODEC_VIDEO_BACKEND}` backend"
        )
    assert not frame_recovery, "frame_recovery is only available for `opencv` backend"
    decoders = {
        "pyav": decode_pyav,
        "torchcodec": decode_torchcodec,
        "pynvvideocodec": decode_pynvvideocodec,
        "deepstream": decode_deepstream,
    }
    return decoders[backend](
        loader_cls,
        data,
        target,
        sampling_kwargs,
        **backend_kwargs,
    )


__all__ = [
    "DeepStreamVideoBackendMixin",
    "OpenCVVideoBackendMixin",
    "PYNVVIDEOCODEC_CUDA_CONTEXT_BYTES",
    "PYNVVIDEOCODEC_DECODER_CACHE_SIZE",
    "PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES",
    "PYNVVIDEOCODEC_MAX_RETAINED_DECODERS",
    "PYNVVIDEOCODEC_VIDEO_BACKEND",
    "PyAVVideoBackendMixin",
    "PyNvVideoCodecDecoderSlot",
    "PyNvVideoCodecVideoBackendMixin",
    "TorchCodecVideoBackendMixin",
    "VideoDecoderBackend",
    "VideoSourceMetadata",
    "VideoTargetMetadata",
    "check_frame_pixel_limit",
    "decode_video",
    "resolve_video_backend_kwargs",
]
