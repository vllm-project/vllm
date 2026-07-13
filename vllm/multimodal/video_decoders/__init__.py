# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy.typing as npt

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

VideoDecodeResult = tuple[
    npt.NDArray,
    VideoSourceMetadata,
    list[int],
    list[int],
]
VideoDecoder = Callable[..., VideoDecodeResult]


@dataclass(frozen=True)
class VideoDecoderSpec:
    decoder: VideoDecoder
    option_defaults: dict[str, Any] = field(default_factory=dict)
    supports_frame_recovery: bool = False
    frame_recovery_error: type[Exception] = AssertionError

    def validate_frame_recovery(self, enabled: bool, backend: str) -> None:
        if enabled and not self.supports_frame_recovery:
            raise self.frame_recovery_error(
                f"frame_recovery is not supported by the {backend!r} backend"
            )

    def decode(
        self,
        backend: str,
        loader_cls,
        data: bytes,
        target: VideoTargetMetadata,
        sampling_kwargs: dict[str, Any],
        backend_kwargs: dict[str, Any],
        *,
        frame_recovery: bool,
    ) -> VideoDecodeResult:
        self.validate_frame_recovery(frame_recovery, backend)
        decoder_kwargs = dict(backend_kwargs)
        if self.supports_frame_recovery:
            decoder_kwargs["frame_recovery"] = frame_recovery
        return self.decoder(
            loader_cls,
            data,
            target,
            sampling_kwargs,
            **decoder_kwargs,
        )


_VIDEO_DECODER_SPECS: dict[str, VideoDecoderSpec] = {
    "opencv": VideoDecoderSpec(
        decoder=decode_opencv,
        supports_frame_recovery=True,
    ),
    "pyav": VideoDecoderSpec(decoder=decode_pyav),
    "torchcodec": VideoDecoderSpec(
        decoder=decode_torchcodec,
        option_defaults={
            "num_ffmpeg_threads": 0,
            "seek_mode": "exact",
        },
    ),
    PYNVVIDEOCODEC_VIDEO_BACKEND: VideoDecoderSpec(
        decoder=decode_pynvvideocodec,
        frame_recovery_error=ValueError,
    ),
    "deepstream": VideoDecoderSpec(
        decoder=decode_deepstream,
        option_defaults={
            "pool_size": None,
            "timeout_sec": 120.0,
        },
    ),
}


def _get_video_decoder_spec(backend: str) -> VideoDecoderSpec:
    try:
        return _VIDEO_DECODER_SPECS[backend]
    except KeyError:
        valid = ", ".join(repr(name) for name in _VIDEO_DECODER_SPECS)
        raise ValueError(
            f"Unknown video codec backend {backend!r}; valid options: {valid}."
        ) from None


def resolve_video_backend_kwargs(
    backend: str,
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split frame-sampling kwargs from options owned by a decoder backend."""
    spec = _get_video_decoder_spec(backend)
    defaults = spec.option_defaults

    sampling_kwargs = dict(kwargs)
    backend_kwargs = dict(defaults)
    backend_option_names = {
        name
        for decoder_spec in _VIDEO_DECODER_SPECS.values()
        for name in decoder_spec.option_defaults
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
    return _get_video_decoder_spec(backend).decode(
        backend,
        loader_cls,
        data,
        target,
        sampling_kwargs,
        backend_kwargs,
        frame_recovery=frame_recovery,
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
