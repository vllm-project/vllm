# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

import numpy as np
import numpy.typing as npt

from vllm.utils.import_utils import PlaceholderModule, check_torchcodec_available

from .base import (
    VideoSourceMetadata,
    VideoTargetMetadata,
    check_frame_pixel_limit,
)

try:
    from torchcodec.decoders import VideoDecoder
except (ImportError, RuntimeError):
    VideoDecoder = PlaceholderModule("torchcodec").placeholder_attr(  # type: ignore[assignment]
        "decoders.VideoDecoder"
    )


def decode_torchcodec(
    loader_cls,
    data: bytes,
    target: VideoTargetMetadata,
    sampling_kwargs: dict,
    *,
    num_ffmpeg_threads: int = 0,
    seek_mode: Literal["exact", "approximate"] = "exact",
) -> tuple[npt.NDArray, VideoSourceMetadata, list[int], list[int]]:
    check_torchcodec_available()
    decoder = loader_cls.make_torchcodec_decoder(
        data,
        num_ffmpeg_threads=num_ffmpeg_threads,
        seek_mode=seek_mode,
    )
    check_frame_pixel_limit(
        decoder.metadata.width or 0,
        decoder.metadata.height or 0,
    )
    source = loader_cls._prepare_source(loader_cls.get_torchcodec_metadata(decoder))
    frame_idx = loader_cls.compute_frames_index_to_sample(
        source=source, target=target, **sampling_kwargs
    )
    frames, valid = loader_cls.decode_torchcodec_frames(decoder, frame_idx)
    return frames, source, frame_idx, valid


class TorchCodecVideoBackendMixin:
    """TorchCodec (FFmpeg-backed, PyTorch-native) codec utilities.

    Builds a :class:`~torchcodec.decoders.VideoDecoder` over the in-memory
    bytes and extracts the sampled indices with a single batched
    ``get_frames_at`` call, while releasing the GIL during decode.
    """

    @staticmethod
    def make_torchcodec_decoder(
        data: bytes,
        *,
        num_ffmpeg_threads: int = 0,
        seek_mode: Literal["exact", "approximate"] = "exact",
    ) -> "VideoDecoder":
        # NHWC matches the (num_frames, H, W, 3) uint8 RGB layout the rest
        # of the pipeline expects, avoiding a transpose.
        return VideoDecoder(
            data,
            dimension_order="NHWC",
            num_ffmpeg_threads=num_ffmpeg_threads,
            seek_mode=seek_mode,
        )

    @staticmethod
    def get_torchcodec_metadata(decoder: "VideoDecoder") -> VideoSourceMetadata:
        md = decoder.metadata
        total_frames = md.num_frames or 0
        fps = float(md.average_fps) if md.average_fps else 0.0
        duration = float(md.duration_seconds) if md.duration_seconds else 0.0
        if total_frames == 0 and duration > 0 and fps > 0:
            total_frames = int(duration * fps)
        return VideoSourceMetadata(total_frames, fps, duration)

    @staticmethod
    def decode_torchcodec_frames(
        decoder: "VideoDecoder",
        frame_indices: list[int],
    ) -> tuple[npt.NDArray, list[int]]:
        """Decode the requested indices in one batched, index-exact call."""
        if not frame_indices:
            return np.empty((0,), dtype=np.uint8), []
        # Note: torchcodec releases the GIL for the entire call
        batch = decoder.get_frames_at(frame_indices)
        return batch.data.numpy(), list(frame_indices)
