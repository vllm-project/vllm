# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pybase64
from PIL import Image

from vllm import envs
from vllm.logger import init_logger

from ..video import VIDEO_LOADER_REGISTRY
from .base import MediaIO, MediaWithBytes
from .image import ImageMediaIO

logger = init_logger(__name__)

_DEFAULT_NUM_FRAMES = 32
_INVALID_NUM_FRAMES_MESSAGE = "num_frames must be greater than 0 or -1"


class VideoMediaIO(MediaIO[MediaWithBytes[tuple[npt.NDArray, dict[str, Any]]]]):
    """Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

    @classmethod
    def merge_kwargs(
        cls,
        default_kwargs: dict[str, Any] | None,
        runtime_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any]:
        runtime_kwargs = cls._enforce_runtime_num_frames_policy(
            default_kwargs,
            runtime_kwargs,
        )
        if runtime_kwargs:
            # Decoder GPU memory is reserved from the startup value.
            runtime_kwargs = dict(runtime_kwargs)
            runtime_kwargs.pop("hw_decoders", None)
            runtime_kwargs.pop("pool_size", None)

            # Block request-level selection of GPU video backends that
            # were not configured (and VRAM-reserved) at startup.
            for key in ("video_backend", "backend"):
                requested = runtime_kwargs.get(key)
                if requested and VIDEO_LOADER_REGISTRY.backend_requires_gpu(requested):
                    static_val = (default_kwargs or {}).get(key)
                    if static_val != requested:
                        logger.warning_once(
                            "Stripping request-level %s=%r: GPU video "
                            "backend not configured at startup.",
                            key,
                            requested,
                        )
                        runtime_kwargs = {
                            k: v for k, v in runtime_kwargs.items() if k != key
                        }

        merged = super().merge_kwargs(default_kwargs, runtime_kwargs)
        # A request num_frames override replaces a default fps because it is
        # a complete frame-count selection. An fps-only request must retain
        # num_frames because that field is also the frame ceiling.
        if (
            runtime_kwargs
            and "num_frames" in runtime_kwargs
            and "fps" not in runtime_kwargs
        ):
            merged.pop("fps", None)
        return merged

    @classmethod
    def _enforce_runtime_num_frames_policy(
        cls,
        default_kwargs: dict[str, Any] | None,
        runtime_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not runtime_kwargs or "num_frames" not in runtime_kwargs:
            return runtime_kwargs

        requested_num_frames = cls._validate_num_frames(runtime_kwargs["num_frames"])
        num_frames_cap = cls._get_finite_num_frames_cap(default_kwargs)
        if num_frames_cap is None or 0 < requested_num_frames <= num_frames_cap:
            return runtime_kwargs

        logger.warning_once(
            "Clamping request-level num_frames=%r to finite video frame cap %d.",
            requested_num_frames,
            num_frames_cap,
        )
        return {**runtime_kwargs, "num_frames": num_frames_cap}

    @classmethod
    def _get_finite_num_frames_cap(
        cls,
        default_kwargs: dict[str, Any] | None,
    ) -> int | None:
        configured_num_frames = (default_kwargs or {}).get(
            "num_frames",
            _DEFAULT_NUM_FRAMES,
        )
        configured_num_frames = cls._validate_num_frames(configured_num_frames)
        if configured_num_frames == -1:
            return None
        return configured_num_frames

    @staticmethod
    def _validate_num_frames(num_frames: Any) -> int:
        if type(num_frames) is not int or num_frames == 0 or num_frames < -1:
            raise ValueError(_INVALID_NUM_FRAMES_MESSAGE)
        return num_frames

    def __init__(
        self,
        image_io: ImageMediaIO,
        num_frames: int = _DEFAULT_NUM_FRAMES,
        **kwargs,
    ) -> None:
        super().__init__()

        self.image_io = image_io
        self.num_frames = num_frames
        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality, merged with
        # per-request runtime media_io_kwargs via merge_kwargs().
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.

        # Allow per-request override of video backend via kwargs.
        # This enables users to specify a different backend than the
        # global VLLM_VIDEO_LOADER_BACKEND env var, e.g.:
        #   --media-io-kwargs '{"video": {"video_backend": "torchcodec"}}'
        video_loader_backend = (
            kwargs.pop("video_backend", None) or envs.VLLM_VIDEO_LOADER_BACKEND
        )
        self.kwargs = kwargs
        self.video_loader = VIDEO_LOADER_REGISTRY.load(video_loader_backend)

    def load_bytes(
        self, data: bytes
    ) -> MediaWithBytes[tuple[npt.NDArray, dict[str, Any]]]:
        video = self.video_loader.load_bytes(
            data, num_frames=self.num_frames, **self.kwargs
        )
        return MediaWithBytes(video, data)

    def load_base64(
        self, media_type: str, data: str
    ) -> MediaWithBytes[tuple[npt.NDArray, dict[str, Any]]]:
        if media_type.lower() == "video/jpeg":
            load_frame = partial(
                self.image_io.load_base64,
                "image/jpeg",
            )

            if self.num_frames > 0:
                frame_parts = data.split(",", self.num_frames)[: self.num_frames]
            elif self.num_frames == 0:
                raise ValueError(_INVALID_NUM_FRAMES_MESSAGE)
            else:
                frame_parts = data.split(",")

            frames = np.stack(
                [np.asarray(load_frame(frame_data)) for frame_data in frame_parts]
            )
            total = int(frames.shape[0])
            fps = float(self.kwargs.get("fps", 1))

            # validate and extract frames_indices
            frames_indices = self.kwargs.get("frames_indices")
            if frames_indices is not None:
                if not (
                    isinstance(frames_indices, list)
                    and all(isinstance(i, int) for i in frames_indices)
                ):
                    raise ValueError("frames_indices must be a list of integers")
                if len(frames_indices) != total:
                    raise ValueError(
                        f"frames_indices length ({len(frames_indices)}) must "
                        f"match number of frames sent ({total})"
                    )
            else:
                frames_indices = list(range(total))

            # validate and extract total_num_frames
            total_num_frames = self.kwargs.get("total_num_frames", total)
            if not isinstance(total_num_frames, int) or total_num_frames < 1:
                raise ValueError("total_num_frames must be a positive integer")
            if total_num_frames < total:
                raise ValueError(
                    f"total_num_frames ({total_num_frames}) must be >= "
                    f"number of frames sent ({total})"
                )

            # validate and extract duration
            duration = self.kwargs.get("duration")
            if duration is not None:
                if not isinstance(duration, (int, float)) or duration < 0:
                    raise ValueError("duration must be a non-negative number")
            else:
                duration = total_num_frames / fps if fps > 0 else 0.0

            metadata = {
                "total_num_frames": total_num_frames,
                "fps": fps,
                "duration": duration,
                "video_backend": "jpeg_sequence",
                "frames_indices": frames_indices,
                "do_sample_frames": self.kwargs.get("do_sample_frames", False),
            }
            return MediaWithBytes((frames, metadata), data.encode())

        return self.load_bytes(pybase64.b64decode(data))

    def load_file(
        self, filepath: Path
    ) -> MediaWithBytes[tuple[npt.NDArray, dict[str, Any]]]:
        with filepath.open("rb") as f:
            data = f.read()

        return self.load_bytes(data)

    def encode_base64(
        self,
        media: npt.NDArray,
        *,
        video_format: str = "JPEG",
    ) -> str:
        video = media

        if video_format == "JPEG":
            encode_frame = partial(
                self.image_io.encode_base64,
                image_format=video_format,
            )

            return ",".join(encode_frame(Image.fromarray(frame)) for frame in video)

        msg = "Only JPEG format is supported for now."
        raise NotImplementedError(msg)
