# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
import threading
from abc import abstractmethod
from contextlib import contextmanager, suppress
from typing import ClassVar, Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from vllm.logger import init_logger
from vllm.utils.mem_constants import MiB_bytes

from .base import VideoSourceMetadata, VideoTargetMetadata, check_frame_pixel_limit

logger = init_logger(__name__)


def decode_pynvvideocodec(
    loader_cls,
    data: bytes,
    target: VideoTargetMetadata,
    sampling_kwargs: dict,
) -> tuple[npt.NDArray, VideoSourceMetadata, list[int], list[int]]:
    return loader_cls.decode_frames_pynvvideocodec(
        data,
        target,
        **sampling_kwargs,
    )


class PyNvVideoCodecSourceMetadata(NamedTuple):
    """Metadata needed before GPU video decode."""

    source: VideoSourceMetadata
    width: int
    height: int


PYNVVIDEOCODEC_VIDEO_BACKEND: Literal["pynvvideocodec"] = "pynvvideocodec"
# Fixed upper bound reserved for persistent PyNvVideoCodec decoder surfaces.
PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES = 128 * MiB_bytes
PYNVVIDEOCODEC_DECODER_CACHE_SIZE = 2
PYNVVIDEOCODEC_MAX_RETAINED_DECODERS = 1
# Per-API-server CUDA context and driver allocation, measured with
# PyNvVideoCodec 2.0.4 on H100.
PYNVVIDEOCODEC_CUDA_CONTEXT_BYTES = int(1.8 * 1024 * MiB_bytes)


class PyNvVideoCodecDecoderSlot:
    """A retained PyNv decoder slot and its CUDA stream.

    The decoder is reused across requests: ``reconfigure_decoder`` repoints the
    existing decoder at each new source instead of paying a fresh
    ``SimpleDecoder`` construction per request. Construction (CUVID parser +
    decoder + surface-pool allocation) is the dominant per-request cost, so
    reconfiguring is far cheaper. A single decoder serves both metadata
    (``len``/``get_stream_metadata``) and frame decode -- no separate
    metadata decoder.
    """

    def __init__(self, stream) -> None:
        self.stream = stream
        self.decoder = None
        self.source_path: str | None = None

    def _construct(self, file_path: str, nvc, device_index: int) -> None:
        self.decoder = nvc.SimpleDecoder(
            file_path,
            output_color_type=nvc.OutputColorType.RGB,
            use_device_memory=True,
            need_scanned_stream_metadata=True,
            gpu_id=device_index,
            cuda_stream=self.stream.cuda_stream,
            decoder_cache_size=PYNVVIDEOCODEC_DECODER_CACHE_SIZE,
        )
        self.source_path = file_path

    def get_decoder(self, file_path: str, nvc, device_index: int):
        if self.decoder is None:
            self._construct(file_path, nvc, device_index)
        elif self.source_path != file_path:
            try:
                self.decoder.reconfigure_decoder(file_path)
                self.source_path = file_path
            except Exception:
                # reconfigure unsupported/unsafe for this source -> rebuild.
                self._construct(file_path, nvc, device_index)
        return self.decoder


class PyNvVideoCodecVideoBackendMixin:
    """PyNvVideoCodec utilities for GPU-backed frame decode."""

    _decoder_slots: ClassVar[list[PyNvVideoCodecDecoderSlot]] = []
    _active_decoder_slots: ClassVar[int] = 0
    _decoder_slot_cond: ClassVar[threading.Condition] = threading.Condition()
    _DEVICE_INDEX: ClassVar[int] = 0

    @classmethod
    @abstractmethod
    def compute_frames_index_to_sample(
        cls,
        source: VideoSourceMetadata,
        target: VideoTargetMetadata,
        **kwargs,
    ) -> list[int]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _prepare_source(cls, source: VideoSourceMetadata) -> VideoSourceMetadata:
        raise NotImplementedError

    @classmethod
    def _create_decoder_slot(cls) -> PyNvVideoCodecDecoderSlot:
        import torch

        return PyNvVideoCodecDecoderSlot(torch.cuda.Stream(device=cls._DEVICE_INDEX))

    @staticmethod
    @contextmanager
    def _torch_stream_context(stream):
        import torch

        torch.accelerator.set_device_index(stream.device.index)
        previous_stream = torch.accelerator.current_stream()
        torch.accelerator.set_stream(stream)
        try:
            yield
        finally:
            torch.accelerator.set_stream(previous_stream)

    @classmethod
    @contextmanager
    def _borrow_decoder_slot(cls):
        create_slot = False
        with cls._decoder_slot_cond:
            while True:
                if cls._decoder_slots:
                    slot = cls._decoder_slots.pop()
                    break
                if cls._active_decoder_slots < PYNVVIDEOCODEC_MAX_RETAINED_DECODERS:
                    cls._active_decoder_slots += 1
                    create_slot = True
                    break
                cls._decoder_slot_cond.wait()

        if create_slot:
            try:
                slot = cls._create_decoder_slot()
            except Exception:
                with cls._decoder_slot_cond:
                    cls._active_decoder_slots -= 1
                    cls._decoder_slot_cond.notify()
                raise

        try:
            yield slot
        finally:
            with cls._decoder_slot_cond:
                cls._decoder_slots.append(slot)
                cls._decoder_slot_cond.notify()

    @staticmethod
    def _metadata_value(metadata, *names: str, default=None):
        for name in names:
            value = getattr(metadata, name, None)
            if value is not None:
                return value
        return default

    @classmethod
    def _read_source_metadata(
        cls,
        file_path: str,
        nvc,
    ) -> PyNvVideoCodecSourceMetadata:
        with cls._borrow_decoder_slot() as decoder_slot:
            with cls._torch_stream_context(decoder_slot.stream):
                decoder = decoder_slot.get_decoder(
                    file_path, nvc, device_index=cls._DEVICE_INDEX
                )
                metadata = decoder.get_stream_metadata()
                total_frames_num = len(decoder)
            width = int(cls._metadata_value(metadata, "width", default=0))
            height = int(cls._metadata_value(metadata, "height", default=0))
            original_fps = float(
                cls._metadata_value(
                    metadata,
                    "average_fps",
                    "avg_frame_rate",
                    "frame_rate",
                    "frameRate",
                    default=0.0,
                )
            )
            duration = float(
                cls._metadata_value(metadata, "duration", default=0.0)
                or (total_frames_num / original_fps if original_fps > 0 else 0.0)
            )
            if total_frames_num <= 0:
                raise ValueError("Could not determine video frame count")
            if width <= 0 or height <= 0:
                raise ValueError("Could not determine video dimensions")
            return PyNvVideoCodecSourceMetadata(
                source=VideoSourceMetadata(total_frames_num, original_fps, duration),
                width=width,
                height=height,
            )

    @classmethod
    def _decode_to_pinned_host(
        cls,
        file_path: str,
        frame_idx: list[int],
        nvc,
    ) -> npt.NDArray:
        import torch

        if not frame_idx:
            return np.empty((0,), dtype=np.uint8)

        with cls._borrow_decoder_slot() as decoder_slot:
            stream = decoder_slot.stream
            with cls._torch_stream_context(stream):
                decoder = decoder_slot.get_decoder(
                    file_path, nvc, device_index=cls._DEVICE_INDEX
                )
                decoded_frames = decoder.get_batch_frames_by_index(frame_idx)
                if len(decoded_frames) < len(frame_idx):
                    logger.warning(
                        "pynvvideocodec video loading: expected %d frames but got %d.",
                        len(frame_idx),
                        len(decoded_frames),
                    )
                torch_frames = [torch.from_dlpack(frame) for frame in decoded_frames]
                if not torch_frames:
                    return np.empty((0,), dtype=np.uint8)
                device_frames = torch.stack(torch_frames)
                if device_frames.ndim != 4:
                    raise ValueError(
                        "PyNvVideoCodec returned frames with unexpected shape "
                        f"{tuple(device_frames.shape)}"
                    )
                device_frames = device_frames.permute(0, 3, 1, 2).contiguous()
                host_frames = torch.empty(
                    device_frames.shape,
                    dtype=device_frames.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                host_frames.copy_(device_frames, non_blocking=True)
                stream.synchronize()
                host_array = host_frames.numpy()
                del decoded_frames, torch_frames, device_frames
                return host_array

    @classmethod
    def decode_frames_pynvvideocodec(
        cls,
        data: bytes,
        target: VideoTargetMetadata,
        **kwargs,
    ) -> tuple[npt.NDArray, VideoSourceMetadata, list[int], list[int]]:
        import PyNvVideoCodec as nvc

        from vllm.multimodal.gpu_ipc_memory import get_mm_gpu_ipc_pool

        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        try:
            with os.fdopen(temp_fd, "wb") as temp_file:
                temp_file.write(data)

            gpu_source = cls._read_source_metadata(temp_path, nvc)
            check_frame_pixel_limit(gpu_source.width, gpu_source.height)
            source = cls._prepare_source(gpu_source.source)
            frame_idx = cls.compute_frames_index_to_sample(
                source=source, target=target, **kwargs
            )
            raw_frame_bytes = len(frame_idx) * gpu_source.height * gpu_source.width * 3
            pool = get_mm_gpu_ipc_pool()
            if pool is None or raw_frame_bytes == 0:
                frames = cls._decode_to_pinned_host(temp_path, frame_idx, nvc)
            else:
                with pool.acquire(raw_frame_bytes):
                    frames = cls._decode_to_pinned_host(temp_path, frame_idx, nvc)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temp_path)

        valid_frame_indices = frame_idx[: int(frames.shape[0])]
        return frames, source, frame_idx, valid_frame_indices
