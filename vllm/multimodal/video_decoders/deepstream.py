# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from typing import Any, ClassVar

import numpy.typing as npt

from vllm.logger import init_logger

from .base import VideoSourceMetadata, VideoTargetMetadata

logger = init_logger(__name__)


def decode_deepstream(
    loader_cls,
    data: bytes,
    target: VideoTargetMetadata,
    sampling_kwargs: dict,
    *,
    pool_size: int | None = None,
    timeout_sec: float = 120.0,
) -> tuple[npt.NDArray, VideoSourceMetadata, list[int], list[int]]:
    from nvidia.deepstream_videodecode import probe_metadata

    total_frames, original_fps, duration, _width, _height, codec = probe_metadata(data)
    source = loader_cls._prepare_source(
        VideoSourceMetadata(total_frames, original_fps, duration)
    )
    frame_idx = loader_cls.compute_frames_index_to_sample(
        source=source, target=target, **sampling_kwargs
    )
    frames, valid = loader_cls.decode_indices(
        data,
        frame_idx,
        source,
        codec=codec,
        pool_size=pool_size,
        timeout_sec=timeout_sec,
    )
    return frames, source, frame_idx, valid


class DeepStreamVideoBackendMixin:
    """NVIDIA DeepStream (NVDEC) GPU-decode codec utilities.

    Decoding runs on a shared pool of daemon threads inside one CUDA
    context (see the ``nvidia-deepstream-videodecode-cu13`` package). The
    container bytes are pushed into an ``appsrc`` GStreamer pipeline, so no
    local file path is required — HTTP and base64 sources decode identically
    to local files.

    Like the OpenCV/PyAV mixins, this provides only the codec layer.
    Frame *selection* lives in the loader's
    ``compute_frames_index_to_sample`` and arrives here as an explicit
    list of frame indices.
    """

    # Process-wide lazy decode pool, shared across all DeepStream backends.
    _pool: ClassVar[Any] = None
    _pool_lock: ClassVar[Any] = None

    @classmethod
    def _get_pool(cls, pool_size: int | None = None):
        """Lazy-initialize the shared decode pool on first use.

        ``pool_size`` (number of decode worker threads) comes from
        ``--media-io-kwargs`` (``{"video": {"pool_size": N}}``); when unset it
        defaults to the existing ``VLLM_MEDIA_LOADING_THREAD_COUNT`` so no
        DeepStream-specific env var is needed. The pool is a process-wide
        singleton, so the first decode's value wins.
        """
        if cls._pool is not None:
            return cls._pool
        if cls._pool_lock is None:
            cls._pool_lock = threading.Lock()
        with cls._pool_lock:
            if cls._pool is not None:
                return cls._pool
            import os

            from nvidia.deepstream_videodecode import DecodePool

            if pool_size is None:
                pool_size = int(os.environ.get("VLLM_MEDIA_LOADING_THREAD_COUNT", 8))
            pool_size = max(1, min(int(pool_size), 16))
            logger.info(
                "[DeepStream] initializing decode pool with %d workers",
                pool_size,
            )
            cls._pool = DecodePool(num_workers=pool_size)
            return cls._pool

    @classmethod
    def decode_indices(
        cls,
        data: bytes,
        frame_indices: list[int],
        source: VideoSourceMetadata,
        codec: str = "",
        pool_size: int | None = None,
        timeout_sec: float = 120.0,
    ) -> tuple[npt.NDArray, list[int]]:
        """Decode the requested frame indices from raw container bytes.

        The whole stream is decoded; the pool keeps exactly the frames whose
        decode-order index is in ``frame_indices`` (1:1, frame-exact) and
        sends EOS once the last one is matched.

        ``codec`` (e.g. ``"h264"``/``"hevc"``) lets the pool keep its NVDEC
        session warm across same-codec streams and rebuild only on a codec
        change. Frames are returned as a CPU NHWC uint8 array so the
        upstream multimodal parser sees the same shape as the other
        backends.
        """
        if not frame_indices:
            raise ValueError("DeepStream backend received no frame indices")

        result = cls._get_pool(pool_size).decode(
            data,
            target_indices=frame_indices,
            codec=codec,
            max_frames=len(frame_indices),
            timeout_sec=timeout_sec,
        )
        if result.error:
            raise ValueError(f"DeepStream decode failed: {result.error}")
        if result.frames is None or result.n_kept == 0:
            raise ValueError("DeepStream decode produced no frames")

        valid = frame_indices[: result.n_kept]
        # GPU -> CPU NHWC uint8 at the codec boundary (one PCIe copy); keeps
        # the array shape identical to the OpenCV/PyAV backends. Copy into
        # PINNED host memory (reused across calls by PyTorch's pinned caching
        # allocator) so the D2H runs at full PCIe bandwidth (~13 GB/s) rather
        # than the ~1 GB/s pageable path that plain ``.cpu()`` takes — ~12x
        # faster for a 1080p x8 frame batch (~46ms -> ~4ms). ``numpy()`` keeps
        # the pinned tensor alive via the array's base.
        import torch

        gpu = result.frames
        if gpu.is_cuda:
            host = torch.empty(gpu.shape, dtype=gpu.dtype, pin_memory=True)
            host.copy_(gpu, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            arr = host.numpy()
        else:
            arr = gpu.numpy()
        return arr, valid
