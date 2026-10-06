# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import os
import tempfile
import threading
from contextlib import contextmanager, suppress
from typing import ClassVar, NamedTuple

import numpy as np
import numpy.typing as npt

from vllm.logger import init_logger
from vllm.utils.mem_constants import MiB_bytes

from .base import (
    PYNVVIDEOCODEC_DEFAULT_HW_DECODERS,
    VideoSourceMetadata,
    VideoTargetMetadata,
    check_frame_pixel_limit,
)

logger = init_logger(__name__)


def decode_pynvvideocodec(
    loader_cls,
    data: bytes,
    target: VideoTargetMetadata,
    sampling_kwargs: dict,
    *,
    hw_decoders: int = PYNVVIDEOCODEC_DEFAULT_HW_DECODERS,
    max_width: int | None = None,
    max_height: int | None = None,
) -> tuple[npt.NDArray, VideoSourceMetadata, list[int], list[int]]:
    max_width, max_height = validate_pynvvideocodec_max_dimensions(
        max_width, max_height
    )
    PyNvVideoCodecVideoBackendMixin._configure_decoder_slots(hw_decoders)
    return PyNvVideoCodecVideoBackendMixin.decode_frames_pynvvideocodec(
        loader_cls,
        data,
        target,
        max_width=max_width,
        max_height=max_height,
        **sampling_kwargs,
    )


class PyNvVideoCodecSourceMetadata(NamedTuple):
    """Metadata needed before GPU video decode."""

    source: VideoSourceMetadata
    width: int
    height: int


# Per-decoder upper bound reserved for persistent PyNvVideoCodec surfaces.
PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES = 128 * MiB_bytes
PYNVVIDEOCODEC_DECODER_CACHE_SIZE = 2
# Bound native frame surfaces retained by each batch decode call.
PYNVVIDEOCODEC_DECODE_BATCH_SIZE = 32
# Per-API-server CUDA context and driver allocation, measured with
# PyNvVideoCodec 2.0.4 on H100.
PYNVVIDEOCODEC_CUDA_CONTEXT_BYTES = int(1.8 * 1024 * MiB_bytes)


def validate_pynvvideocodec_hw_decoders(hw_decoders: object) -> int:
    if (
        isinstance(hw_decoders, bool)
        or not isinstance(hw_decoders, int)
        or hw_decoders < 1
    ):
        raise ValueError("hw_decoders must be a positive integer")
    return hw_decoders


def validate_pynvvideocodec_max_dimensions(
    max_width: object,
    max_height: object,
) -> tuple[int | None, int | None]:
    if max_width is None and max_height is None:
        return None, None
    if max_width is None or max_height is None:
        raise ValueError("max_width and max_height must be set together")
    if isinstance(max_width, bool) or not isinstance(max_width, int) or max_width < 1:
        raise ValueError("max_width must be a positive integer")
    if (
        isinstance(max_height, bool)
        or not isinstance(max_height, int)
        or max_height < 1
    ):
        raise ValueError("max_height must be a positive integer")
    return max_width, max_height


def _pynvvideocodec_exception_types(nvc) -> tuple[type[Exception], ...]:
    return tuple(
        exception_type
        for name in dir(nvc)
        if name.startswith("PyNvVCException")
        and isinstance((exception_type := getattr(nvc, name)), type)
        and issubclass(exception_type, Exception)
    )


def _pynvvc_frames_to_nhwc(frames):
    """Return a stacked PyNvVideoCodec frame batch as contiguous NHWC."""
    if frames.shape[-1] != 3 and frames.shape[-3] == 3:
        frames = frames.permute(0, 2, 3, 1)
    return frames.contiguous()


class PyNvVideoCodecDecoderSlot:
    """A retained PyNv decoder slot and its CUDA stream.

    The decoder is reused across requests: ``reconfigure_decoder`` repoints the
    existing decoder at each new source instead of paying a fresh
    ``SimpleDecoder`` construction per request. Construction (CUVID parser +
    decoder + surface-pool allocation) is the dominant per-request cost, so
    reconfiguring is far cheaper. Decoder dimensions are bounded at construction
    so resolution changes do not replace its native frame pool. A single decoder
    serves both metadata (``len``/``get_stream_metadata``) and frame decode -- no
    separate metadata decoder.
    """

    def __init__(self, stream) -> None:
        self.stream = stream
        self.decoder = None
        self.source_path: str | None = None
        self.max_width = 0
        self.max_height = 0

    def invalidate(self) -> None:
        decoder = self.decoder
        self.decoder = None
        self.source_path = None
        self.max_width = 0
        self.max_height = 0
        if decoder is not None:
            with suppress(Exception):
                decoder.stop()

    def recover(self) -> None:
        """Release decoder resources and drain work after a failed request."""
        self.invalidate()
        gc.collect()

        with suppress(Exception):
            self.stream.synchronize()

        with suppress(Exception):
            import torch

            with torch.cuda.device(self.stream.device):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    def _construct(
        self,
        file_path: str,
        nvc,
        device_index: int,
        max_width: int,
        max_height: int,
    ) -> None:
        decoder = nvc.SimpleDecoder(
            file_path,
            output_color_type=nvc.OutputColorType.RGB,
            use_device_memory=True,
            need_scanned_stream_metadata=True,
            gpu_id=device_index,
            cuda_stream=self.stream.cuda_stream,
            decoder_cache_size=PYNVVIDEOCODEC_DECODER_CACHE_SIZE,
            max_width=max_width,
            max_height=max_height,
        )
        self.decoder = decoder
        self.source_path = file_path
        self.max_width = max_width
        self.max_height = max_height

    def get_decoder(
        self,
        file_path: str,
        nvc,
        device_index: int,
        max_width: int,
        max_height: int,
    ):
        if self.decoder is None:
            self._construct(file_path, nvc, device_index, max_width, max_height)
        elif self.source_path != file_path:
            if max_width > self.max_width or max_height > self.max_height:
                self.recover()
                self._construct(file_path, nvc, device_index, max_width, max_height)
            else:
                try:
                    self.decoder.reconfigure_decoder(file_path)
                    self.source_path = file_path
                except Exception:
                    # reconfigure unsupported/unsafe for this source -> rebuild.
                    self.recover()
                    self._construct(file_path, nvc, device_index, max_width, max_height)
        return self.decoder


class _PyNvDecoderPool:
    """Process-wide singleton managing PyNvVideoCodec decoder slot state.

    Prevents subclass counter shadowing (GHSA-j682-9xp5-rrf3) by storing
    all mutable pool state in a single module-level instance rather than
    in ClassVar attributes that get shadowed by Python's augmented
    assignment semantics on subclasses.
    """

    def __init__(self) -> None:
        self.slots: list[PyNvVideoCodecDecoderSlot] = []
        self.active: int = 0
        self.cond: threading.Condition = threading.Condition()
        self.max_slots: int | None = None

    def configure(self, hw_decoders: int) -> None:
        with self.cond:
            if self.max_slots is None:
                self.max_slots = hw_decoders
            elif self.max_slots != hw_decoders:
                raise RuntimeError(
                    "PyNvVideoCodec decoder count is already configured as "
                    f"{self.max_slots}, got {hw_decoders}"
                )


_pynv_decoder_pool = _PyNvDecoderPool()


class PyNvVideoCodecVideoBackendMixin:
    """PyNvVideoCodec utilities for GPU-backed frame decode."""

    _DEVICE_INDEX: ClassVar[int] = 0

    @classmethod
    def _create_decoder_slot(cls) -> PyNvVideoCodecDecoderSlot:
        import torch

        return PyNvVideoCodecDecoderSlot(torch.cuda.Stream(device=cls._DEVICE_INDEX))

    @classmethod
    def _configure_decoder_slots(cls, hw_decoders: object) -> None:
        hw_decoders = validate_pynvvideocodec_hw_decoders(hw_decoders)
        _pynv_decoder_pool.configure(hw_decoders)

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
        pool = _pynv_decoder_pool
        create_slot = False
        with pool.cond:
            if pool.max_slots is None:
                raise RuntimeError("PyNvVideoCodec decoder slots are not configured")
            while True:
                if pool.slots:
                    slot = pool.slots.pop()
                    break
                if pool.active < pool.max_slots:
                    pool.active += 1
                    create_slot = True
                    break
                pool.cond.wait()

        if create_slot:
            try:
                slot = cls._create_decoder_slot()
            except Exception:
                with pool.cond:
                    pool.active -= 1
                    pool.cond.notify()
                raise

        borrow_succeeded = False
        try:
            yield slot
            borrow_succeeded = True
        finally:
            if not borrow_succeeded:
                slot.recover()
            with pool.cond:
                if borrow_succeeded:
                    pool.slots.append(slot)
                else:
                    pool.active -= 1
                pool.cond.notify()

    @staticmethod
    def _metadata_value(metadata, *names: str, default=None):
        for name in names:
            value = getattr(metadata, name, None)
            if value is not None:
                return value
        return default

    @classmethod
    def _validate_decoder_caps(
        cls,
        file_path: str,
        nvc,
        max_width: int | None = None,
        max_height: int | None = None,
    ) -> tuple[int, int]:
        demuxer = nvc.CreateDemuxer(file_path)
        width = demuxer.Width()
        height = demuxer.Height()
        caps = nvc.GetDecoderCaps(
            cls._DEVICE_INDEX,
            demuxer.GetNvCodecId(),
            demuxer.ChromaFormat(),
            demuxer.BitDepth(),
        )
        width_min = int(caps["width_min"])
        width_max = int(caps["width_max"])
        height_min = int(caps["height_min"])
        height_max = int(caps["height_max"])
        mb_num_max = int(caps["mb_num_max"])
        macroblock_count = ((width + 15) // 16) * ((height + 15) // 16)
        if (
            not caps["supported"]
            or width < width_min
            or width > width_max
            or height < height_min
            or height > height_max
            or macroblock_count > mb_num_max
        ):
            raise ValueError("Invalid or unsupported video file.")

        if max_width is None or max_height is None:
            return width_max, height_max

        configured_mb_count = ((max_width + 15) // 16) * ((max_height + 15) // 16)
        if (
            max_width < width_min
            or max_width > width_max
            or max_height < height_min
            or max_height > height_max
            or configured_mb_count > mb_num_max
        ):
            raise ValueError(
                "Configured PyNvVideoCodec max_width and max_height are "
                "outside the hardware decoder limits"
            )
        if width > max_width or height > max_height:
            raise ValueError("Invalid or unsupported video file.")
        return max_width, max_height

    @classmethod
    def _read_source_metadata(
        cls,
        file_path: str,
        nvc,
        max_width: int,
        max_height: int,
    ) -> PyNvVideoCodecSourceMetadata:
        with cls._borrow_decoder_slot() as decoder_slot:
            with cls._torch_stream_context(decoder_slot.stream):
                decoder = decoder_slot.get_decoder(
                    file_path,
                    nvc,
                    device_index=cls._DEVICE_INDEX,
                    max_width=max_width,
                    max_height=max_height,
                )
                try:
                    metadata = decoder.get_stream_metadata()
                    total_frames_num = len(decoder)
                finally:
                    del decoder
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
        max_width: int,
        max_height: int,
    ) -> npt.NDArray:
        import torch

        if not frame_idx:
            return np.empty((0,), dtype=np.uint8)

        with cls._borrow_decoder_slot() as decoder_slot:
            stream = decoder_slot.stream
            with cls._torch_stream_context(stream):
                decoder = decoder_slot.get_decoder(
                    file_path,
                    nvc,
                    device_index=cls._DEVICE_INDEX,
                    max_width=max_width,
                    max_height=max_height,
                )
                host_frames = None
                decoded_count = 0
                for start in range(0, len(frame_idx), PYNVVIDEOCODEC_DECODE_BATCH_SIZE):
                    chunk_indices = frame_idx[
                        start : start + PYNVVIDEOCODEC_DECODE_BATCH_SIZE
                    ]
                    invalid_video = False
                    try:
                        decoded_frames = decoder.get_batch_frames_by_index(
                            chunk_indices
                        )
                    except Exception as exc:
                        decoder = None
                        if not isinstance(
                            exc,
                            _pynvvideocodec_exception_types(nvc) + (IndexError,),
                        ):
                            raise
                        invalid_video = True
                    if invalid_video:
                        raise ValueError("Invalid or unsupported video file.")

                    torch_frames = [
                        torch.from_dlpack(frame) for frame in decoded_frames
                    ]
                    if not torch_frames:
                        stream.synchronize()
                        break
                    device_frames = _pynvvc_frames_to_nhwc(torch.stack(torch_frames))
                    if device_frames.ndim != 4:
                        raise ValueError(
                            "PyNvVideoCodec returned frames with unexpected shape "
                            f"{tuple(device_frames.shape)}"
                        )
                    if host_frames is None:
                        host_frames = torch.empty(
                            (len(frame_idx), *device_frames.shape[1:]),
                            dtype=device_frames.dtype,
                            device="cpu",
                            pin_memory=True,
                        )
                    elif tuple(device_frames.shape[1:]) != tuple(host_frames.shape[1:]):
                        raise ValueError(
                            "PyNvVideoCodec returned frames with inconsistent shapes"
                        )

                    next_count = decoded_count + len(device_frames)
                    host_frames[decoded_count:next_count].copy_(
                        device_frames, non_blocking=True
                    )
                    stream.synchronize()
                    decoded_count = next_count
                    del decoded_frames, torch_frames, device_frames

                    if decoded_count < start + len(chunk_indices):
                        break

                if decoded_count < len(frame_idx):
                    logger.warning(
                        "pynvvideocodec video loading: expected %d frames but got %d.",
                        len(frame_idx),
                        decoded_count,
                    )
                if host_frames is None:
                    return np.empty((0,), dtype=np.uint8)
                return host_frames[:decoded_count].numpy()

    @classmethod
    def decode_frames_pynvvideocodec(
        cls,
        loader_cls,
        data: bytes,
        target: VideoTargetMetadata,
        max_width: int | None = None,
        max_height: int | None = None,
        **kwargs,
    ) -> tuple[npt.NDArray, VideoSourceMetadata, list[int], list[int]]:
        import PyNvVideoCodec as nvc

        from vllm.multimodal.gpu_ipc_memory import get_mm_gpu_ipc_pool

        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        try:
            with os.fdopen(temp_fd, "wb") as temp_file:
                temp_file.write(data)

            invalid_video = False
            try:
                decoder_max_width, decoder_max_height = cls._validate_decoder_caps(
                    temp_path,
                    nvc,
                    max_width=max_width,
                    max_height=max_height,
                )
            except Exception as exc:
                if not isinstance(exc, _pynvvideocodec_exception_types(nvc)):
                    raise
                invalid_video = True
            if invalid_video:
                raise ValueError("Invalid or unsupported video file.")

            try:
                gpu_source = cls._read_source_metadata(
                    temp_path, nvc, decoder_max_width, decoder_max_height
                )
            except Exception as exc:
                if not isinstance(exc, _pynvvideocodec_exception_types(nvc)):
                    raise
                invalid_video = True
            if invalid_video:
                raise ValueError("Invalid or unsupported video file.")
            check_frame_pixel_limit(gpu_source.width, gpu_source.height)
            source = loader_cls._prepare_source(gpu_source.source)
            frame_idx = loader_cls.compute_frames_index_to_sample(
                source=source, target=target, **kwargs
            )
            raw_frame_bytes = len(frame_idx) * gpu_source.height * gpu_source.width * 3
            pool = get_mm_gpu_ipc_pool()
            if pool is None or raw_frame_bytes == 0:
                frames = cls._decode_to_pinned_host(
                    temp_path,
                    frame_idx,
                    nvc,
                    decoder_max_width,
                    decoder_max_height,
                )
            else:
                with pool.acquire(raw_frame_bytes):
                    frames = cls._decode_to_pinned_host(
                        temp_path,
                        frame_idx,
                        nvc,
                        decoder_max_width,
                        decoder_max_height,
                    )
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temp_path)

        valid_frame_indices = frame_idx[: int(frames.shape[0])]
        return frames, source, frame_idx, valid_frame_indices
