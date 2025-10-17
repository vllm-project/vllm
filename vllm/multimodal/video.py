# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import math
from abc import abstractmethod
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from vllm import envs

from .base import MediaIO
from .image import ImageMediaIO


def resize_video(frames: npt.NDArray, size: tuple[int, int]) -> npt.NDArray:
    num_frames, _, _, channels = frames.shape
    new_height, new_width = size
    resized_frames = np.empty(
        (num_frames, new_height, new_width, channels), dtype=frames.dtype
    )
    # lazy import cv2 to avoid bothering users who only use text models
    import cv2

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
    @abstractmethod
    def load_bytes(
        cls, data: bytes, num_frames: int = -1, **kwargs
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        raise NotImplementedError


class VideoLoaderRegistry:
    def __init__(self) -> None:
        self.name2class: dict[str, type] = {}

    def register(self, name: str):
        def wrap(cls_to_register):
            self.name2class[name] = cls_to_register
            return cls_to_register

        return wrap

    @staticmethod
    def load(cls_name: str) -> VideoLoader:
        cls = VIDEO_LOADER_REGISTRY.name2class.get(cls_name)
        assert cls is not None, f"VideoLoader class {cls_name} not found"
        return cls()


VIDEO_LOADER_REGISTRY = VideoLoaderRegistry()


@VIDEO_LOADER_REGISTRY.register("opencv")
class OpenCVVideoBackend(VideoLoader):
    def get_cv2_video_api(self):
        import cv2.videoio_registry as vr

        api_pref = None
        for backend in vr.getStreamBufferedBackends():
            if not vr.hasBackend(backend):
                continue
            if not vr.isBackendBuiltIn(backend):
                _, abi, api = vr.getStreamBufferedBackendPluginVersion(backend)
                if abi < 1 or (abi == 1 and api < 2):
                    continue
            api_pref = backend
            break
        return api_pref

    @classmethod
    def _get_frame_indices_to_sample(
        cls,
        total_frames_num: int,
        max_num_frames_to_sample: int,
        **kwargs,
    ) -> list[int]:
        full_read = (
            max_num_frames_to_sample == -1
            or total_frames_num < max_num_frames_to_sample
        )
        if full_read:
            frame_idx = list(range(0, total_frames_num))
        else:
            uniform_sampled_frames = np.linspace(
                0, total_frames_num - 1, max_num_frames_to_sample, dtype=int
            )
            frame_idx = uniform_sampled_frames.tolist()
        return frame_idx

    @classmethod
    def _sample_frames_from_video(
        cls,
        cap,
        frame_indices: list[int],
        allow_missing_frames: bool = False,
    ) -> npt.NDArray:
        import cv2

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = np.empty((len(frame_indices), height, width, 3), dtype=np.uint8)

        i = 0
        for idx in range(max(frame_indices) + 1):
            ok = cap.grab()
            if not ok:
                break
            if idx in frame_indices:
                ret, frame = cap.retrieve()
                if ret:
                    frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    i += 1

        if not allow_missing_frames and i != len(frame_indices):
            raise ValueError(
                f"Expected reading {len(frame_indices)} frames, "
                f"but only loaded {i} frames from video."
            )

        return frames

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        import cv2

        backend = cls().get_cv2_video_api()
        cap = cv2.VideoCapture(BytesIO(data), backend, [])
        if not cap.isOpened():
            raise ValueError("Could not open video stream")

        total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames_num / original_fps if original_fps > 0 else 0

        # resample video to target num_frames
        frame_indices = cls._get_frame_indices_to_sample(total_frames_num, num_frames)
        num_frames = len(frame_indices)
        frames = cls._sample_frames_from_video(cap, frame_indices)

        # Use transformers transformers.video_utils.VideoMetadata format
        # NOTE(Isotr0py): For models like Qwen3-VL/GLM4.5V, this metadata
        # can cause incorrect timestamp calculation without num_frames=-1.
        metadata = {
            "total_num_frames": num_frames,
            "fps": num_frames / duration,
            "duration": duration,
            "video_backend": "opencv",
            "frames_indices": frame_indices,
            # extra field used to control hf processor's video
            # sampling behavior
            "do_sample_frames": num_frames == total_frames_num,
        }

        return frames, metadata


@VIDEO_LOADER_REGISTRY.register("opencv_dynamic")
class OpenCVDynamicVideoBackend(OpenCVVideoBackend):
    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = 2,
        max_duration: int = 300,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        import cv2

        backend = cls().get_cv2_video_api()
        cap = cv2.VideoCapture(BytesIO(data), backend, [])
        if not cap.isOpened():
            raise ValueError("Could not open video stream")

        total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames_num / original_fps if original_fps > 0 else 0

        # resample video to target num_frames
        max_frame_idx = total_frames_num - 1
        duration = duration or round(max_frame_idx / original_fps) + 1

        # Refer to:
        # https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/glm4v/video_processing_glm4v.py#L103-L140
        frame_indices: range | list[int]
        if duration <= max_duration:
            n = int(math.floor(duration * fps))
            frame_indices = sorted(
                {
                    min(max_frame_idx, int(math.ceil(i * original_fps / fps)))
                    for i in range(n)
                }
            )
        else:
            num_samples = int(max_duration * fps)
            if num_samples >= total_frames_num:
                frame_indices = range(total_frames_num)
            else:
                target_seconds = np.linspace(0, duration, num_samples, endpoint=True)
                frame_indices = sorted(
                    {
                        min(max_frame_idx, int(math.ceil(t * original_fps)))
                        for t in target_seconds
                    }
                )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = np.empty((len(frame_indices), height, width, 3), dtype=np.uint8)

        i = 0
        for idx in range(total_frames_num):
            ok = cap.grab()
            if not ok:
                break
            if idx in frame_indices:
                ret, frame = cap.retrieve()
                if ret:
                    frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    i += 1

        assert i == len(frame_indices), (
            f"Expected reading {len(frame_indices)} frames, "
            f"but only loaded {i} frames from video."
        )

        # Use transformers transformers.video_utils.VideoMetadata format
        metadata = {
            "total_num_frames": total_frames_num,
            "fps": original_fps,
            "duration": duration,
            "video_backend": "opencv_dynamic",
            "frames_indices": list(frame_indices),
            "do_sample_frames": False,
        }

        return frames, metadata


@VIDEO_LOADER_REGISTRY.register("opencv_nemotron_vl_v2")
class OpenCVNemotronVideoBackend(OpenCVVideoBackend):
    @classmethod
    def _get_frame_indices_to_sample(
        cls,
        total_frames_num: int,
        max_num_frames_to_sample: int,
        fps: int,
        duration_seconds: float,
        **kwargs,
    ) -> list[int]:
        # Determine target number of samples:
        max_samples = total_frames_num
        if max_num_frames_to_sample > 0:  # Hard upper bound
            max_samples = min(max_num_frames_to_sample, max_samples)
        if fps > 0:  # If fps is provided, use it to limit the number of samples
            max_samples = min(max_samples, math.floor(duration_seconds * fps))
        max_samples = max(1, max_samples)  # to make sure we have at least one sample

        # Uniform coverage of the entire timeline within the cap
        # Use linspace over [0, total_frames-1]
        raw = np.linspace(0, total_frames_num - 1, max_samples, endpoint=True)
        return np.unique(raw.round().astype(int)).tolist()

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = -1,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        """
        Args:
            num_frames (int): Maximum number of frames to load.
            A total sampled number of frames will never be larger
            than this value. Set it -1 to remove the upper limit.

            fps (int): Desired video sampling rate. A real samping
            rate may be lower if we encounter long video and
            num_frames upper limit is set to positive value.
        """
        import cv2

        backend = cls().get_cv2_video_api()
        cap = cv2.VideoCapture(BytesIO(data), backend, [])
        if not cap.isOpened():
            raise ValueError("Could not open video stream")

        total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames_num == 0:
            raise ValueError("CAP_PROP_FRAME_COUNT returned 0")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if not (original_fps > 0):
            print(
                f"WARNING: CAP_PROP_FPS returned {original_fps}. "
                f"We will use 30 FPS as default fallback."
            )
            original_fps = 30

        duration = total_frames_num / original_fps

        frame_indices = cls._get_frame_indices_to_sample(
            total_frames_num, num_frames, fps, duration
        )

        effective_fps = len(frame_indices) / duration
        print(
            f"Video [{total_frames_num} fames]({duration:.2f}sec "
            f"at {original_fps:.2f}fps) sampled "
            f"into frame [{len(frame_indices)}] indexes {frame_indices} "
            f"at {effective_fps:.2f}fps."
        )

        frames = cls._sample_frames_from_video(
            cap, frame_indices, allow_missing_frames=True
        )

        # Use transformers transformers.video_utils.VideoMetadata format
        metadata = {
            "total_num_frames": total_frames_num,
            "fps": original_fps,
            "duration": duration,
            "video_backend": "opencv_nemotron_vl_v2",
            "frames_indices": frame_indices,
            "do_sample_frames": False,
        }

        return frames, metadata


class VideoMediaIO(MediaIO[npt.NDArray]):
    def __init__(
        self,
        image_io: ImageMediaIO,
        num_frames: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()

        self.image_io = image_io
        self.num_frames = num_frames
        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality.
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.
        self.kwargs = kwargs
        video_loader_backend = envs.VLLM_VIDEO_LOADER_BACKEND
        self.video_loader = VIDEO_LOADER_REGISTRY.load(video_loader_backend)

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, dict[str, Any]]:
        return self.video_loader.load_bytes(
            data, num_frames=self.num_frames, **self.kwargs
        )

    def load_base64(
        self, media_type: str, data: str
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        if media_type.lower() == "video/jpeg":
            load_frame = partial(
                self.image_io.load_base64,
                "image/jpeg",
            )

            return np.stack(
                [np.asarray(load_frame(frame_data)) for frame_data in data.split(",")]
            ), {}

        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, dict[str, Any]]:
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
