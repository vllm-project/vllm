# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from abc import abstractmethod
from typing import Any, ClassVar, Literal, cast

import numpy as np
import numpy.typing as npt
import torch

from vllm.logger import init_logger
from vllm.multimodal.video_decoders import (
    PYNVVIDEOCODEC_CUDA_CONTEXT_BYTES as PYNVVIDEOCODEC_CUDA_CONTEXT_BYTES,
)
from vllm.multimodal.video_decoders import (
    PYNVVIDEOCODEC_DECODER_CACHE_SIZE as PYNVVIDEOCODEC_DECODER_CACHE_SIZE,
)
from vllm.multimodal.video_decoders import (
    PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES as PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES,
)
from vllm.multimodal.video_decoders import (
    PYNVVIDEOCODEC_MAX_RETAINED_DECODERS as PYNVVIDEOCODEC_MAX_RETAINED_DECODERS,
)
from vllm.multimodal.video_decoders import (
    PYNVVIDEOCODEC_VIDEO_BACKEND,
    DeepStreamVideoBackendMixin,
    OpenCVVideoBackendMixin,
    PyAVVideoBackendMixin,
    PyNvVideoCodecVideoBackendMixin,
    TorchCodecVideoBackendMixin,
    VideoDecoderBackend,
    VideoSourceMetadata,
    VideoTargetMetadata,
    check_frame_pixel_limit,
    decode_video,
    resolve_video_backend_kwargs,
)
from vllm.multimodal.video_decoders import (
    PyNvVideoCodecDecoderSlot as PyNvVideoCodecDecoderSlot,
)
from vllm.utils.import_utils import PlaceholderModule
from vllm.utils.registry import ExtensionManager

try:
    import cv2
except ImportError:
    cv2 = PlaceholderModule("cv2")


logger = init_logger(__name__)


class VideoLoaderRegistry(ExtensionManager):
    def __init__(self) -> None:
        super().__init__()
        self.processor2backend: dict[str, str] = {}
        self._requires_gpu: dict[str, bool] = {}

    @staticmethod
    def _normalize_registered_video_processors(
        video_processor: str | tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        if video_processor is None:
            return ()

        if isinstance(video_processor, str):
            return (video_processor,)

        if all(isinstance(processor, str) for processor in video_processor):
            return video_processor

        raise TypeError(
            "video_processor must be a class name or a tuple of class names"
        )

    def register(
        self,
        name: str,
        *,
        video_processor: str | tuple[str, ...] | None = None,
        requires_gpu: bool = False,
    ):
        processors = self._normalize_registered_video_processors(video_processor)

        def wrap(cls_to_register):
            self.name2class[name] = cls_to_register
            self._requires_gpu[name] = requires_gpu
            for processor_name in processors:
                self.processor2backend[processor_name] = name
            return cls_to_register

        return wrap

    def get_backend_for_video_processor(
        self,
        video_processor: str | None,
    ) -> str | None:
        if video_processor is None:
            return None

        return self.processor2backend.get(video_processor)

    def backend_requires_gpu(self, name: str) -> bool:
        return self._requires_gpu.get(name, False)


def get_video_loader_backend_for_processor(
    video_processor: str | None,
) -> str | None:
    return VIDEO_LOADER_REGISTRY.get_backend_for_video_processor(video_processor)


def resize_video(frames: npt.NDArray, size: tuple[int, int]) -> npt.NDArray:
    num_frames, _, _, channels = frames.shape
    new_height, new_width = size
    resized_frames = np.empty(
        (num_frames, new_height, new_width, channels), dtype=frames.dtype
    )

    for i, frame in enumerate(frames):
        resized_frame = cv2.resize(frame, (new_width, new_height))
        resized_frames[i] = resized_frame
    return resized_frames


def rescale_video_size(frames: npt.NDArray, size_factor: float) -> npt.NDArray:
    _, height, width, _ = frames.shape
    new_height = int(height * size_factor)
    new_width = int(width * size_factor)

    return resize_video(frames, (new_height, new_width))


def sample_frames_from_video(frames: npt.NDArray, num_frames: int) -> npt.NDArray:
    total_frames = frames.shape[0]
    if num_frames == -1:
        return frames

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    sampled_frames = frames[frame_indices, ...]
    return sampled_frames


class VideoLoader:
    @classmethod
    def compute_frames_index_to_sample(
        cls,
        source: VideoSourceMetadata,
        target: VideoTargetMetadata,
        **kwargs,
    ) -> list[int]:
        """Return the list of frame indices to sample from the video."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_bytes(
        cls,
        data: bytes,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        """Load video frames from bytes and return (frames_array, metadata_dict)."""
        raise NotImplementedError

    @classmethod
    def create_hf_metadata(
        cls,
        source: VideoSourceMetadata,
        valid_frame_indices: list[int],
        video_backend: str,
    ):
        return {
            "total_num_frames": source.total_frames_num,
            "fps": source.original_fps,
            "duration": source.duration,
            "video_backend": video_backend,
            "frames_indices": valid_frame_indices,
            "do_sample_frames": len(valid_frame_indices) == source.total_frames_num,
        }


VIDEO_LOADER_REGISTRY = VideoLoaderRegistry()


@VIDEO_LOADER_REGISTRY.register("opencv")
class VideoBackend(
    VideoLoader,
    OpenCVVideoBackendMixin,
    PyAVVideoBackendMixin,
    TorchCodecVideoBackendMixin,
    PyNvVideoCodecVideoBackendMixin,
    DeepStreamVideoBackendMixin,
):
    """Uniform-sampling video backend.

    Samples ``num_frames`` uniformly across the video (or one frame every
    ``1/fps`` seconds, whichever produces fewer frames). The decoding codec
    is selected via the ``backend`` kwarg (``"opencv"``, ``"pyav"``,
    ``"torchcodec"``, ``"pynvvideocodec"``, or ``"deepstream"``),
    which can be passed through ``--media-io-kwargs``. Defaults to ``"opencv"``.
    """

    _sampling_suffix: ClassVar[str] = ""

    @classmethod
    def compute_frames_index_to_sample(
        cls,
        source: VideoSourceMetadata,
        target: VideoTargetMetadata,
        **kwargs,
    ) -> list[int]:
        total_frames_num = source.total_frames_num
        duration = source.duration
        num_frames = target.num_frames
        fps = target.fps
        # resample video to target num_frames and fps
        # - the minimum of the two will be used
        num_frames_to_sample = total_frames_num
        if num_frames > 0:
            num_frames_to_sample = min(num_frames, total_frames_num)
        if fps > 0:
            num_frames_to_sample = min(num_frames_to_sample, math.floor(duration * fps))
        num_frames_to_sample = max(1, num_frames_to_sample)

        if num_frames_to_sample == total_frames_num:
            return list(range(num_frames_to_sample))
        return np.linspace(
            0, total_frames_num - 1, num_frames_to_sample, dtype=int
        ).tolist()

    @classmethod
    def _prepare_source(cls, source: VideoSourceMetadata) -> VideoSourceMetadata:
        """Sampling-algorithm-specific metadata adjustment hook."""
        return source

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = -1,
        max_duration: int = 300,
        frame_recovery: bool = False,
        *,
        backend: VideoDecoderBackend = "opencv",
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        """Load sampled frames from raw video bytes.

        Args:
            data: Raw video bytes.
            num_frames: Target number of frames to sample (``-1`` for all).
            fps: Target FPS for sampling (``-1`` for original).
            max_duration: Maximum duration in seconds — only used by the
                dynamic subclass; ignored here.
            frame_recovery: Enable forward-scan recovery for failed frames.
                Only honored by the OpenCV codec.
            backend: Decoding codec — ``"opencv"``, ``"pyav"``,
                ``"torchcodec"``, ``"pynvvideocodec"`` or ``"deepstream"``.
            num_ffmpeg_threads: Number of FFmpeg decoding threads, only used by
                TorchCodec: ``0`` (default) relies on the FFmpeg default value
                which is ``min(cpu_count + 1, 16)``.
                OpenCV will always use ``min(cpu_count, 16)`` while pyav will
                always use ``min(cpu_count, (height + 15) / 16)``.
            seek_mode: Seek mode for the TorchCodec decoder, only used by
                TorchCodec: ``"exact"`` (default) guarantees frame-accurate
                sampling by scanning the file on creation, while
                ``"approximate"`` skips that scan for faster decoder creation
                at the cost of relying on the file's metadata. See
                https://meta-pytorch.org/torchcodec/stable/generated_examples/decoding/approximate_mode.html
                for details.

        Returns:
            Tuple of ``(frames_array, metadata_dict)``.
        """
        target = VideoTargetMetadata(
            num_frames=num_frames, fps=fps, max_duration=max_duration
        )
        sampling_kwargs, backend_kwargs = resolve_video_backend_kwargs(backend, kwargs)
        frames, source, frame_idx, valid = decode_video(
            backend,
            cls,
            data,
            target,
            sampling_kwargs,
            backend_kwargs,
            frame_recovery=frame_recovery,
        )

        if len(valid) < len(frame_idx):
            logger.warning(
                "%s video loading: expected %d frames but got %d.",
                backend,
                len(frame_idx),
                len(valid),
            )

        return frames, cls.create_hf_metadata(
            source=source,
            video_backend=f"{backend}{cls._sampling_suffix}",
            valid_frame_indices=valid,
        )


@VIDEO_LOADER_REGISTRY.register(PYNVVIDEOCODEC_VIDEO_BACKEND, requires_gpu=True)
class PyNvVideoCodecVideoBackend(VideoBackend):
    """Hardware-accelerated video backend using PyNvVideoCodec.

    The backend first opens the stream only to read metadata and compute the
    sampled frame indices. It then acquires the raw decoded RGB byte count from
    the process-local multimodal GPU memory pool before decoding the selected
    frames into VRAM. Decoded frames are copied into pinned host memory before
    the lease is released, so downstream preprocessing continues to receive a
    CPU ``np.ndarray`` in NHWC RGB format.
    """

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = -1,
        max_duration: int = 300,
        frame_recovery: bool = False,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        kwargs.pop("backend", None)
        return super().load_bytes(
            data,
            num_frames=num_frames,
            fps=fps,
            max_duration=max_duration,
            frame_recovery=frame_recovery,
            backend=PYNVVIDEOCODEC_VIDEO_BACKEND,
            **kwargs,
        )


@VIDEO_LOADER_REGISTRY.register(
    "qwen3_vl",
    video_processor="Qwen3VLVideoProcessor",
)
class Qwen3VLVideoBackend(VideoBackend):
    @classmethod
    def compute_frames_index_to_sample(
        cls,
        source: VideoSourceMetadata,
        target: VideoTargetMetadata,
        **kwargs,
    ) -> list[int]:
        total_frames_num = source.total_frames_num
        original_fps = source.original_fps
        fps = target.fps
        max_frame_idx = source.total_frames_num - 1
        min_frames = kwargs.get("min_frames", 4)
        max_frames = kwargs.get("max_frames", 768)

        # Refer to:
        # https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py#L119-L125
        num_frames = int(total_frames_num / original_fps * fps)
        num_frames = min(max(num_frames, min_frames), max_frames, total_frames_num)
        indices = np.linspace(0, max_frame_idx, num_frames).round().astype(int).tolist()
        return indices

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = 2,
        max_duration: int = 300,
        frame_recovery: bool = False,
        *,
        backend: Literal[
            "opencv", "pyav", "torchcodec", "pynvvideocodec", "deepstream"
        ] = "opencv",
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        return super().load_bytes(
            data,
            num_frames=num_frames,
            fps=fps,
            max_duration=max_duration,
            frame_recovery=frame_recovery,
            backend=backend,
            **kwargs,
        )


@VIDEO_LOADER_REGISTRY.register(
    "qwen2_vl",
    video_processor="Qwen2VLVideoProcessor",
)
class Qwen2VLVideoBackend(VideoBackend):
    """Qwen2-VL / Qwen2.5-VL fps-based video backend.

    Ports transformers' ``Qwen2VLVideoProcessor.sample_frames`` (fps mode),
    shared by Qwen2-VL and Qwen2.5-VL (the latter has no video processor of its
    own): sample ``total / original_fps * fps`` frames, clamp to
    ``[min_frames, max_frames]`` (4 and 768), floor to a multiple of
    ``temporal_patch_size`` (2), and take indices with the exact
    ``torch.arange(0, total, total / n)`` call so they match HF byte-for-byte.

    ``num_frames`` is ignored (fps-driven, like the Qwen3-VL loader). The
    float32 step can emit an out-of-range tail index (e.g. 451 for a 451-frame
    clip); it is clamped to the last valid frame.
    """

    @classmethod
    def compute_frames_index_to_sample(
        cls,
        source: VideoSourceMetadata,
        target: VideoTargetMetadata,
        **kwargs,
    ) -> list[int]:
        # Refer to:
        # https://github.com/huggingface/transformers/blob/v5.7.0/src/transformers/models/qwen2_vl/video_processing_qwen2_vl.py#L122-L190
        total_frames_num = source.total_frames_num
        original_fps = source.original_fps
        temporal_patch_size = kwargs.get("temporal_patch_size", 2)
        min_frames = kwargs.get("min_frames", 4)
        max_frames = kwargs.get("max_frames", 768)

        # vLLM reports original_fps == 0 for clips with unknown/variable fps
        # (VFR, malformed, streaming); fail loudly instead of dividing by zero.
        if original_fps <= 0:
            raise ValueError(
                "Qwen2-VL video sampling needs a known source fps, but the "
                "container reported 0 (variable or unknown frame rate)."
            )

        max_frames = (
            math.floor(min(max_frames, total_frames_num) / temporal_patch_size)
            * temporal_patch_size
        )
        n = total_frames_num / original_fps * target.fps
        n = min(max(n, min_frames), max_frames, total_frames_num)
        n = math.floor(n / temporal_patch_size) * temporal_patch_size

        # ``torch.arange`` matches transformers' float32 index math exactly
        # (numpy's float64 diverges by a frame on some inputs); clamp the tail
        # because that step can emit an index == total_frames_num.
        indices = torch.arange(0, total_frames_num, total_frames_num / n).int()
        return torch.clamp(indices, max=total_frames_num - 1).tolist()

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = 2,
        max_duration: int = 300,
        frame_recovery: bool = False,
        *,
        backend: Literal[
            "opencv", "pyav", "torchcodec", "pynvvideocodec", "deepstream"
        ] = "opencv",
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        return super().load_bytes(
            data,
            num_frames=num_frames,
            fps=fps,
            max_duration=max_duration,
            frame_recovery=frame_recovery,
            backend=backend,
            **kwargs,
        )


@VIDEO_LOADER_REGISTRY.register(
    "opencv_dynamic",
    video_processor="Glm4vVideoProcessor",
)
class DynamicVideoBackend(VideoBackend):
    """Duration-aware dynamic-sampling video backend.

    Samples at ``fps`` up to ``max_duration`` seconds, falling back to
    uniform sampling across the full duration when the video is longer
    than ``max_duration``. Codec is selectable the same way as
    :class:`VideoBackend`.
    """

    _sampling_suffix: ClassVar[str] = "_dynamic"

    @classmethod
    def _prepare_source(cls, source: VideoSourceMetadata) -> VideoSourceMetadata:
        # Estimate duration from frame count and fps when the container
        # does not report it (common for WebM/streaming inputs).
        if source.duration:
            return source
        if source.original_fps > 0:
            max_frame_idx = source.total_frames_num - 1
            duration = round(max_frame_idx / source.original_fps) + 1
        else:
            duration = 0
        return VideoSourceMetadata(
            source.total_frames_num, source.original_fps, duration
        )

    @classmethod
    def compute_frames_index_to_sample(
        cls,
        source: VideoSourceMetadata,
        target: VideoTargetMetadata,
        **kwargs,
    ) -> list[int]:
        total_frames_num = source.total_frames_num
        duration = source.duration
        original_fps = source.original_fps
        max_duration = target.max_duration
        fps = target.fps
        max_frame_idx = source.total_frames_num - 1

        # Refer to:
        # https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/glm4v/video_processing_glm4v.py#L103-L140
        frame_indices_list: list[int]
        if duration <= max_duration:
            n = int(math.floor(duration * fps))
            frame_indices_list = sorted(
                {
                    min(max_frame_idx, int(math.ceil(i * original_fps / fps)))
                    for i in range(n)
                }
            )
        else:
            num_samples = int(max_duration * fps)
            if num_samples >= total_frames_num:
                frame_indices_list = list(range(total_frames_num))
            else:
                target_seconds = np.linspace(0, duration, num_samples, endpoint=True)
                frame_indices_list = sorted(
                    {
                        min(max_frame_idx, int(math.ceil(t * original_fps)))
                        for t in target_seconds
                    }
                )
        return frame_indices_list

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = 2,
        max_duration: int = 300,
        frame_recovery: bool = False,
        *,
        backend: Literal[
            "opencv", "pyav", "torchcodec", "pynvvideocodec", "deepstream"
        ] = "opencv",
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        return super().load_bytes(
            data,
            num_frames=num_frames,
            fps=fps,
            max_duration=max_duration,
            frame_recovery=frame_recovery,
            backend=backend,
            **kwargs,
        )


@VIDEO_LOADER_REGISTRY.register(
    "glm46v",
    video_processor="Glm46VVideoProcessor",
)
class GLM46VVideoBackend(VideoBackend):
    """GLM-4.6V dynamic FPS video backend.

    Faithfully replicates the frame sampling logic from transformers'
    ``Glm46VVideoProcessor.sample_frames``:

    - Dynamic FPS thresholds based on effective video duration:
      ``{≤30s: 3fps, ≤300s: 1fps, >300s: 0.5fps}``
    - ``temporal_patch_size`` multiplier (default 2) applied to extract count
    - Duration capped at 2400s, frame count capped at 640
    - Even frame count enforced (append last frame if odd)
    """

    # Match transformers defaults
    _DYNAMIC_FPS_THRESHOLDS: ClassVar[dict[int, float]] = {
        30: 3.0,
        300: 1.0,
        2400: 0.5,
    }
    _MAX_FRAME_COUNT_DYNAMIC: ClassVar[int] = 640
    _MAX_DURATION: ClassVar[int] = 2400

    @classmethod
    def compute_frames_index_to_sample(
        cls,
        source: VideoSourceMetadata,
        target: VideoTargetMetadata,
        **kwargs,
    ) -> list[int]:
        # Refer to:
        # https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/glm46v/video_processing_glm46v.py#L97-L102
        total_frames_num = source.total_frames_num
        original_fps = source.original_fps
        duration = source.duration
        temporal_patch_size = kwargs.get("temporal_patch_size", 2)

        max_frame_idx = total_frames_num - 1

        # Estimate duration from frame count and fps when not reported
        if not duration and original_fps > 0:
            duration = round(max_frame_idx / original_fps) + 1

        effective_duration = min(duration, cls._MAX_DURATION)

        # Select target_fps from dynamic thresholds
        if effective_duration <= 30:
            target_fps = cls._DYNAMIC_FPS_THRESHOLDS[30]
        elif effective_duration <= 300:
            target_fps = cls._DYNAMIC_FPS_THRESHOLDS[300]
        else:
            target_fps = cls._DYNAMIC_FPS_THRESHOLDS[2400]

        extract_t = int(effective_duration * target_fps * temporal_patch_size)
        extract_t = min(extract_t, cls._MAX_FRAME_COUNT_DYNAMIC)

        duration_per_frame = 1 / original_fps if original_fps > 0 else 0
        max_second = int(duration) if duration else 0

        if total_frames_num < extract_t:
            frame_indices = np.linspace(
                0, total_frames_num - 1, extract_t, dtype=int
            ).tolist()
        else:
            frame_indices = []
            current_second = 0.0
            inv_fps = 1 / (temporal_patch_size * target_fps)
            for frame_index in range(total_frames_num):
                if frame_index * duration_per_frame >= current_second:
                    current_second += inv_fps
                    frame_indices.append(frame_index)
                    if current_second >= max_second:
                        break

        if len(frame_indices) < extract_t:
            if len(frame_indices) == 0:
                start, end = 0, max(total_frames_num - 1, 0)
            else:
                start, end = frame_indices[0], frame_indices[-1]
            frame_indices = np.linspace(start, end, extract_t, dtype=int).tolist()
        elif len(frame_indices) > extract_t:
            frame_indices = np.linspace(
                0, total_frames_num - 1, extract_t, dtype=int
            ).tolist()

        # Deduplicate
        seen: set[int] = set()
        uniq: list[int] = []
        for idx in frame_indices:
            if idx not in seen:
                seen.add(idx)
                uniq.append(idx)

        # Ensure even frame count
        if len(uniq) & 1:
            uniq.append(uniq[-1])

        return uniq

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = -1,
        max_duration: int = 300,
        frame_recovery: bool = False,
        *,
        backend: Literal[
            "opencv", "pyav", "torchcodec", "pynvvideocodec", "deepstream"
        ] = "opencv",
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        return super().load_bytes(
            data,
            num_frames=num_frames,
            fps=fps,
            max_duration=max_duration,
            frame_recovery=frame_recovery,
            backend=backend,
            **kwargs,
        )


@VIDEO_LOADER_REGISTRY.register(
    "glmga",
    video_processor="GlmgaVideoProcessor",
)
class GLMGAVideoBackend(VideoBackend):
    @classmethod
    def _prepare_source(cls, source: VideoSourceMetadata) -> VideoSourceMetadata:
        # Estimate duration from frame count and fps when the container
        # does not report it (common for WebM/streaming inputs).
        if source.duration:
            return source
        if source.original_fps > 0:
            max_frame_idx = source.total_frames_num - 1
            duration = round(max_frame_idx / source.original_fps) + 1
        else:
            duration = 0
        return VideoSourceMetadata(
            source.total_frames_num, source.original_fps, duration
        )

    @classmethod
    def compute_frames_index_to_sample(
        cls,
        source: VideoSourceMetadata,
        target: VideoTargetMetadata,
        **kwargs,
    ) -> list[int]:
        total_frames_num = source.total_frames_num
        duration = source.duration
        original_fps = source.original_fps
        target_fps = target.fps
        max_frame_idx = source.total_frames_num - 1
        max_frames = kwargs.get("max_frames", 640)

        duration = duration or round(max_frame_idx / original_fps) + 1

        extract_t = int(duration * target_fps)
        extract_t = min(extract_t, max_frames)

        duration_per_frame = 1 / original_fps

        if total_frames_num < extract_t:
            frame_indices = [
                math.floor(i * total_frames_num / extract_t) for i in range(extract_t)
            ]
        else:
            frame_indices = []
            current_second = 0.0
            inv_fps = 1 / target_fps
            for frame_index in range(total_frames_num):
                if frame_index * duration_per_frame >= current_second:
                    current_second += inv_fps
                    frame_indices.append(frame_index)
                    if current_second >= duration - inv_fps:
                        break

        if len(frame_indices) < extract_t:
            if len(frame_indices) == 0:
                start, end = 0, max(total_frames_num - 1, 0)
            else:
                start, end = frame_indices[0], frame_indices[-1]
            frame_indices = np.linspace(start, end, extract_t, dtype=int).tolist()
        elif len(frame_indices) > extract_t:
            frame_indices = np.linspace(
                0, total_frames_num - 1, extract_t, dtype=int
            ).tolist()

        seen, uniq = set(), []
        for idx in frame_indices:
            if idx not in seen:
                seen.add(idx)
                uniq.append(idx)

        return uniq

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = 2,
        max_duration: int = 300,
        frame_recovery: bool = False,
        *,
        backend: Literal[
            "opencv", "pyav", "torchcodec", "pynvvideocodec", "deepstream"
        ] = "opencv",
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        frames, metadata = super().load_bytes(
            data,
            num_frames=num_frames,
            fps=fps,
            max_duration=max_duration,
            frame_recovery=frame_recovery,
            backend=backend,
            **kwargs,
        )
        # Ensure even frame count — matches HF's sample_frames even-padding
        # and _preprocess temporal_patch_size divisibility check.
        if frames.shape[0] & 1:
            frames = np.concatenate([frames, frames[-1:]], axis=0)
        return frames, metadata


@VIDEO_LOADER_REGISTRY.register(
    "molmo2",
    video_processor="Molmo2VideoProcessor",
)
class Molmo2VideoBackend(VideoLoader, OpenCVVideoBackendMixin):
    @classmethod
    def get_candidate_target_fps(
        cls,
        video_fps: float,
        sampling_fps: float,
        max_fps: float = 8.0,
    ) -> list[float]:
        """
        Return the subset of `video_fps` factors that remain multiples
        of `sampling_fps`.

        Examples:
            >>> get_candidate_target_fps(video_fps=6, sampling_fps=2)
            [2, 6]
            >>> get_candidate_target_fps(video_fps=5, sampling_fps=1)
            [1, 5]
            >>> get_candidate_target_fps(video_fps=2, sampling_fps=2)
            [2]
            >>> get_candidate_target_fps(video_fps=5, sampling_fps=2)
            Traceback (most recent call last):
                ...
            ValueError: sampling_fps=2 must divide video_fps=5 to produce
                consistent frame steps.
        """
        video_fps = int(video_fps)
        sampling_fps = int(sampling_fps)
        max_fps = int(max_fps)

        if sampling_fps is None:
            raise ValueError("sampling_fps must be provided")
        if video_fps <= 0 or sampling_fps <= 0:
            raise ValueError(
                "video_fps and sampling_fps must be positive "
                f"(got {video_fps}, {sampling_fps})"
            )
        if video_fps % sampling_fps != 0:
            raise ValueError(
                f"sampling_fps={sampling_fps} must divide video_fps={video_fps}."
            )

        candidates = []
        for candidate in range(sampling_fps, video_fps + 1, sampling_fps):
            if candidate > max_fps:
                break
            if video_fps % candidate == 0:
                candidates.append(float(candidate))

        return candidates

    @classmethod
    def get_target_fps(
        cls,
        video_fps: float,
        max_frames: int,
        total_frames: int,
        frame_sample_mode: str,
        candidate_target_fps: list[float],
    ) -> float | None:
        """
        Get the target fps that best spans the videoand has the most frames sampled
        """
        num_frames_sampled = 0
        selected_target_fps = None
        for target_fps in candidate_target_fps:
            step_size = max(int(video_fps / target_fps), 1)
            num_frames_sampled_at_fps = int(total_frames / step_size)
            if num_frames_sampled == 0:
                if (
                    "uniform" in frame_sample_mode
                    and num_frames_sampled_at_fps > max_frames
                ):
                    break
                selected_target_fps = target_fps
                num_frames_sampled = num_frames_sampled_at_fps

            else:
                # the candidate sampling fps increases so frame count can't decrease
                assert num_frames_sampled <= num_frames_sampled_at_fps
                if num_frames_sampled_at_fps > max_frames:
                    # choose the sampling fps that spans the video
                    continue

                elif num_frames_sampled_at_fps > num_frames_sampled:
                    # both are less than max_frames; choose the one with higher
                    # density of frames sampled
                    selected_target_fps = target_fps
                    num_frames_sampled = num_frames_sampled_at_fps
        return selected_target_fps

    @classmethod
    def get_frame_times_and_chosen_fps(
        cls,
        selected_target_fps: float | None,
        total_frames: int,
        max_frames: int,
        video_fps: float,
    ) -> tuple[float | None, npt.NDArray]:
        if selected_target_fps is None:
            frame_indices = np.linspace(
                0, total_frames, max_frames, endpoint=False, dtype=int
            )
        else:
            step_size = max(int(video_fps / selected_target_fps), 1)
            frame_indices = np.arange(0, total_frames, step_size)
        if len(frame_indices) > max_frames:
            frame_indices = frame_indices[:max_frames]
        return selected_target_fps, frame_indices

    @classmethod
    def sample_times(
        cls,
        duration: float,
        max_frames: int,
        frame_sample_mode: str,
        max_fps: int | None,
        candidate_target_fps: list[float] | None = None,
        **kwargs,
    ) -> npt.NDArray:
        if frame_sample_mode == "fps":
            assert candidate_target_fps is not None
            # Try larger and larger FPSs until we hit one that can't span the video
            sampling_fps = candidate_target_fps[0]
            for candidate_fps in candidate_target_fps[1:]:
                if max_frames / candidate_fps < duration:
                    break
                sampling_fps = candidate_fps
            times = np.arange(0, max_frames) / sampling_fps
            times = times[times < duration]
            return times
        elif frame_sample_mode == "uniform_last_frame":
            if max_fps is not None:
                max_duration = (
                    max_frames - 1
                ) / max_fps  # -1 to include the last frame
                if max_duration < duration:
                    times = np.linspace(
                        0, duration, num=max_frames, endpoint=True, dtype=np.float64
                    )
                else:
                    times = np.arange(0.0, stop=duration, step=1 / max_fps)
                    times = np.concatenate([times, [duration]], axis=0)
                    assert len(times) <= max_frames
            else:
                times = np.linspace(
                    0, duration, num=max_frames, endpoint=True, dtype=np.float64
                )
            return times
        else:
            raise NotImplementedError(frame_sample_mode)

    @classmethod
    def compute_frames_index_to_sample(
        cls,
        source: VideoSourceMetadata,
        target: VideoTargetMetadata,
        **kwargs,
    ):
        max_fps = kwargs.get("max_fps")
        frame_sample_mode = kwargs.get("frame_sample_mode")
        if frame_sample_mode is None:
            return list(range(0, source.total_frames_num))

        if frame_sample_mode not in {"uniform_last_frame", "fps"}:
            raise NotImplementedError(
                f"Unsupported frame_sample_mode: {frame_sample_mode}"
            )

        duration = source.duration
        video_fps = source.original_fps
        total_num_frames = source.total_frames_num
        num_frames = target.num_frames
        sampling_fps = target.fps

        if frame_sample_mode == "uniform_last_frame" and max_fps is not None:
            if total_num_frames <= 2:
                indices = np.arange(total_num_frames).astype(int)
            elif duration > (num_frames - 1) / max_fps:  # -1 to include the last frame
                # uniform fallback
                indices = np.linspace(
                    0,
                    total_num_frames - 1,
                    num=min(num_frames, total_num_frames),
                    endpoint=True,
                ).astype(int)
            else:
                float_indices = np.arange(
                    0.0,
                    stop=total_num_frames - 1,
                    step=float(video_fps / max_fps),
                )
                if np.round(float_indices[-1]) != total_num_frames - 1:
                    float_indices = np.concatenate(
                        [float_indices, [total_num_frames - 1]], axis=0
                    )
                indices = np.round(float_indices).astype(int)
                assert indices[-1] < total_num_frames
                assert len(float_indices) <= num_frames
        elif frame_sample_mode == "uniform_last_frame":
            indices = np.linspace(
                0,
                total_num_frames - 1,
                num=min(num_frames, total_num_frames),
                endpoint=True,
            ).astype(int)
        elif frame_sample_mode == "fps":
            candidate_target_fps = cls.get_candidate_target_fps(video_fps, sampling_fps)
            selected_target_fps = cls.get_target_fps(
                video_fps,
                num_frames,
                total_num_frames,
                frame_sample_mode,
                candidate_target_fps,
            )
            _, indices = cls.get_frame_times_and_chosen_fps(
                selected_target_fps,
                total_num_frames,
                num_frames,
                video_fps,
            )
        return indices.tolist()

    @classmethod
    def load_bytes_opencv(
        cls,
        data: bytes,
        frame_sample_mode: str | None = None,
        num_frames: int = -1,
        max_fps: int = 2,
        sampling_fps: int = 2,
        frame_recovery: bool = False,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        cap = cls.open_video_capture(data)
        check_frame_pixel_limit(
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        source = OpenCVVideoBackendMixin.get_video_metadata(cap)
        target = VideoTargetMetadata(
            num_frames=num_frames,
            fps=sampling_fps,
            max_duration=source.duration,
        )

        frame_idx = cls.compute_frames_index_to_sample(
            source=source,
            target=target,
            frame_sample_mode=frame_sample_mode,
            max_fps=max_fps,
        )

        frames, valid_frame_indices = cls.read_frames(
            cap,
            frame_idx,
            total_frames_num=source.total_frames_num,
            frame_recovery=frame_recovery,
        )

        metadata = cls.create_hf_metadata(
            source=source,
            video_backend="opencv",
            valid_frame_indices=valid_frame_indices,
        )

        return frames, metadata

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        frame_sample_mode = cast(str | None, kwargs.pop("frame_sample_mode", None))
        max_fps = cast(int, kwargs.pop("max_fps", 2))
        sampling_fps = cast(int, kwargs.pop("sampling_fps", 2))
        out = cls.load_bytes_opencv(
            data,
            frame_sample_mode,
            num_frames,
            max_fps,
            sampling_fps,
            **kwargs,
        )
        return out


@VIDEO_LOADER_REGISTRY.register("nemotron_vl")
class NemotronVLVideoBackend(VideoBackend):
    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = -1,
        max_duration: int = 300,
        frame_recovery: bool = False,
        *,
        backend: Literal[
            "opencv", "pyav", "torchcodec", "pynvvideocodec", "deepstream"
        ] = "opencv",
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        frames, metadata = super().load_bytes(
            data,
            num_frames=num_frames,
            fps=fps,
            max_duration=max_duration,
            frame_recovery=frame_recovery,
            backend=backend,
            **kwargs,
        )

        metadata = dict(metadata)
        metadata["original_video_bytes"] = data

        return frames, metadata


@VIDEO_LOADER_REGISTRY.register("openpangu")
class OpenCVDynamicOpenPanguVideoBackend(VideoLoader, OpenCVVideoBackendMixin):
    @classmethod
    def compute_frames_index_to_sample(
        cls,
        source: VideoSourceMetadata,
        target: VideoTargetMetadata,
        **kwargs,
    ) -> list[int]:
        total_frames_num = source.total_frames_num
        original_fps = source.original_fps
        num_frames = target.num_frames
        fps = target.fps

        # The timestamp of the rightmost frame, cannot be used to calculate frame 0.
        if total_frames_num >= 1 and original_fps > 0:
            total_duration = (total_frames_num - 1) / original_fps
        else:
            total_duration = 0

        # `fps` is the FPS parameter passed in for sampling,
        # -1 indicates that sampling can be performed directly without FPS limitation.
        if fps > 0:
            # Num_frames is the maximum number of frames to sample.
            # If fewer frames are sampled at this sample_fps, the update duration will be longer. # noqa: E501
            if num_frames >= int(total_duration * fps) + 1:
                num_frames = int(total_duration * fps) + 1
                # Under the new maximum frame rate, the video duration of the rightmost frame, # noqa: E501
                # cannot be calculated for frame 0.
                total_duration = min(total_duration, (num_frames - 1) / fps)
        elif fps != -1:
            raise ValueError(
                f"requires dataset fps is -1 or greater than 0 but got {fps}"
            )

        sample_frame_timestamps = np.linspace(
            0, total_duration, num_frames, dtype=float
        )
        frames_indices = [
            min(total_frames_num - 1, round(t * original_fps))
            for t in sample_frame_timestamps
        ]
        return frames_indices

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = 2,
        max_duration: int = 300,
        frame_recovery: bool = False,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        """
        Load video frames with dynamic sampling based on duration.

        Args:
            data: Raw video bytes
            num_frames: Not used in dynamic backend
            fps: Target FPS for sampling (default: 2)
            max_duration: Maximum video duration to process (default: 300s)
            frame_recovery: Enable forward-scan recovery for failed frames

        Returns:
            Tuple of (frames_array, metadata_dict)
        """
        cap = cls.open_video_capture(data)
        check_frame_pixel_limit(
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        source = OpenCVVideoBackendMixin.get_video_metadata(cap)

        # recompute source metadata with adjusted duration to ensure correct
        # sampling indices computation
        target = VideoTargetMetadata(
            num_frames=num_frames,
            fps=fps,
            max_duration=max_duration,
        )

        frame_indices_list = cls.compute_frames_index_to_sample(
            source=source,
            target=target,
        )

        frames, valid_frame_indices = cls.read_frames(
            cap,
            frame_indices_list,
            total_frames_num=source.total_frames_num,
            frame_recovery=frame_recovery,
        )

        # Use transformers.video_utils.VideoMetadata format
        metadata = cls.create_hf_metadata(
            source=source,
            video_backend="opencv_dynamic",
            valid_frame_indices=valid_frame_indices,
        )
        return frames, metadata
