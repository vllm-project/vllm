# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from vllm.utils.import_utils import PlaceholderModule, check_torchcodec_available

from .base import (
    VideoSourceMetadata,
    VideoTargetMetadata,
    check_frame_pixel_limit,
)

if TYPE_CHECKING:
    import torch

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
    device: "str | torch.device | None" = None,
) -> tuple[npt.NDArray, VideoSourceMetadata, list[int], list[int]]:
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
    budget: AbstractContextManager[Any] = nullcontext()
    if _is_cuda_device(device):
        # GPU decode runs in the API server process. Account for the sampled
        # frames against the frontend budget like the other GPU backends do.
        from vllm.multimodal.gpu_ipc_memory import get_mm_gpu_ipc_pool

        pool = get_mm_gpu_ipc_pool()
        raw_frame_bytes = (
            len(frame_idx)
            * (decoder.metadata.height or 0)
            * (decoder.metadata.width or 0)
            * 3
        )
        if pool is not None and raw_frame_bytes > 0:
            budget = pool.acquire(raw_frame_bytes)
    with budget:
        frames, valid = TorchCodecVideoBackendMixin.decode_torchcodec_frames(
            decoder, frame_idx
        )
    return frames, source, frame_idx, valid


def _is_cuda_device(device: "str | torch.device | None") -> bool:
    return device is not None and str(device).startswith("cuda")


class TorchCodecVideoBackendMixin:
    """TorchCodec (FFmpeg-backed, PyTorch-native) codec utilities.

    Builds a :class:`~torchcodec.decoders.VideoDecoder` over the in-memory
    bytes and extracts the sampled indices with a single batched
    ``get_frames_at`` call, while releasing the GIL during decode.

    ``device`` selects TorchCodec's decoder device. ``None`` keeps the CPU
    (FFmpeg) default; ``"cuda"`` decodes with NVDEC when the installed
    TorchCodec build has CUDA support. Sampled frames are always returned as
    host NumPy arrays, which is what the multimodal processors consume.
    """

    @staticmethod
    def make_torchcodec_decoder(
        data: bytes,
        *,
        num_ffmpeg_threads: int = 0,
        seek_mode: Literal["exact", "approximate"] = "exact",
        device: "str | torch.device | None" = None,
    ) -> "VideoDecoder":
        # NHWC matches the (num_frames, H, W, 3) uint8 RGB layout the rest
        # of the pipeline expects, avoiding a transpose.
        decoder_kwargs: dict[str, Any] = {
            "dimension_order": "NHWC",
            "num_ffmpeg_threads": num_ffmpeg_threads,
            "seek_mode": seek_mode,
        }
        # Forward ``device`` only when one was selected so the TorchCodec
        # default (CPU) and its own validation errors stay untouched.
        if device is not None:
            decoder_kwargs["device"] = device
        return VideoDecoder(data, **decoder_kwargs)

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
        # Multimodal processors consume host NumPy frames. A CUDA decoder
        # returns device tensors, which cannot be converted directly, so copy
        # only the sampled frames at this boundary. ``cpu()`` is a no-op for
        # the default CPU decoder.
        return batch.data.cpu().numpy(), list(frame_indices)
