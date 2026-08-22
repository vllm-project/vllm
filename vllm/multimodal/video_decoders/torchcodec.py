# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

import numpy as np
import numpy.typing as npt
import torch

from vllm.logger import init_logger
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

logger = init_logger(__name__)


def decode_torchcodec(
    loader_cls,
    data: bytes,
    target: VideoTargetMetadata,
    sampling_kwargs: dict,
    *,
    num_ffmpeg_threads: int = 0,
    seek_mode: Literal["exact", "approximate"] = "exact",
    device: str = "cpu",
) -> tuple[npt.NDArray | torch.Tensor, VideoSourceMetadata, list[int], list[int]]:
    check_torchcodec_available()
    decoder = TorchCodecVideoBackendMixin.make_torchcodec_decoder(
        data,
        num_ffmpeg_threads=num_ffmpeg_threads,
        seek_mode=seek_mode,
        device=device,
    )
    check_frame_pixel_limit(
        decoder.metadata.width or 0,
        decoder.metadata.height or 0,
    )
    source = loader_cls._prepare_source(
        TorchCodecVideoBackendMixin.get_torchcodec_metadata(decoder)
    )
    frame_idx = loader_cls.compute_frames_index_to_sample(
        source=source, target=target, **sampling_kwargs
    )
    frames, valid = TorchCodecVideoBackendMixin.decode_torchcodec_frames(
        decoder, frame_idx, device=device
    )
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
        device: str = "cpu",
    ) -> "VideoDecoder":
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                f"torchcodec video decoding on device {device!r} requires "
                "CUDA, but CUDA is not available."
            )
        if torch_device.type not in ("cpu", "cuda"):
            raise ValueError(
                f"torchcodec video decoding only supports 'cpu' and 'cuda' "
                f"devices, got {device!r}."
            )
        # NHWC matches the (num_frames, H, W, 3) uint8 RGB layout the rest
        # of the pipeline expects, avoiding a transpose.
        return VideoDecoder(
            data,
            dimension_order="NHWC",
            num_ffmpeg_threads=num_ffmpeg_threads,
            seek_mode=seek_mode,
            device=device,
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
        *,
        device: str = "cpu",
    ) -> tuple[npt.NDArray | torch.Tensor, list[int]]:
        """Decode the requested indices in one batched, index-exact call.

        With a non-CPU ``device`` the frames stay on the GPU as a torch
        tensor (NVDEC hardware decoding), so a device-side HF processor can
        consume them without a host round-trip.
        """
        if not frame_indices:
            return np.empty((0,), dtype=np.uint8), []
        # Note: torchcodec releases the GIL for the entire call
        batch = decoder.get_frames_at(frame_indices)
        frames = batch.data
        if frames.device.type == "cpu":
            if torch.device(device).type != "cpu":
                # The codec or resolution is not supported by NVDEC, and
                # torchcodec silently fell back to CPU decoding.
                logger.warning_once(
                    "torchcodec could not use NVDEC for this video and "
                    "decoded on CPU instead; check codec support and "
                    "libnvcuvid."
                )
            return frames.numpy(), list(frame_indices)
        return frames, list(frame_indices)
