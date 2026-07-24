# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module
from typing import Any, Literal

from .base import (
    PYNVVIDEOCODEC_VIDEO_BACKEND,
    VideoSourceMetadata,
    VideoTargetMetadata,
    check_frame_pixel_limit,
)

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
    PYNVVIDEOCODEC_VIDEO_BACKEND: {},
    "deepstream": {
        "pool_size": None,
        "timeout_sec": 120.0,
    },
}


def _get_backend_option_defaults(backend: str) -> dict[str, Any]:
    try:
        return _BACKEND_OPTION_DEFAULTS[backend]
    except KeyError:
        valid = ", ".join(repr(name) for name in _BACKEND_OPTION_DEFAULTS)
        raise ValueError(
            f"Unknown video codec backend {backend!r}; valid options: {valid}."
        ) from None


def resolve_video_backend_kwargs(
    backend: str,
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split frame-sampling kwargs from options owned by a decoder backend."""
    defaults = _get_backend_option_defaults(backend)
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
    """Decode a sampled video, importing only the selected backend."""
    _get_backend_option_defaults(backend)
    if frame_recovery and backend != "opencv":
        error = (
            ValueError if backend == PYNVVIDEOCODEC_VIDEO_BACKEND else AssertionError
        )
        raise error(f"frame_recovery is not supported by the {backend!r} backend")

    decoder_kwargs = dict(backend_kwargs)
    if backend == "opencv":
        decoder_kwargs["frame_recovery"] = frame_recovery
    module = import_module(f".{backend}", __name__)
    decoder = getattr(module, f"decode_{backend}")
    return decoder(
        loader_cls,
        data,
        target,
        sampling_kwargs,
        **decoder_kwargs,
    )


__all__ = [
    "PYNVVIDEOCODEC_VIDEO_BACKEND",
    "VideoDecoderBackend",
    "VideoSourceMetadata",
    "VideoTargetMetadata",
    "check_frame_pixel_limit",
    "decode_video",
    "resolve_video_backend_kwargs",
]
