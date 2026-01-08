# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import math
from abc import abstractmethod
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from PIL import Image

from vllm import envs
from vllm.logger import init_logger
from vllm.utils.registry import ExtensionManager

from .base import MediaIO
from .image import ImageMediaIO

logger = init_logger(__name__)


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

    @staticmethod
    def _read_frames(
        cap,
        frame_indices: set[int],
        num_expected_frames: int,
        max_frame_idx: int,
    ) -> tuple[npt.NDArray, int, list[int]]:
        import cv2

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = np.empty((num_expected_frames, height, width, 3), dtype=np.uint8)

        i = 0
        valid_frame_indices = []
        for idx in range(max_frame_idx + 1):
            ok = cap.grab()
            if not ok:
                # Frame is broken/unreadable, log warning
                if idx in frame_indices:
                    logger.warning(
                        "Failed to grab frame %d during video loading. "
                        "This frame will be skipped.",
                        idx,
                    )
                continue
            if idx in frame_indices:
                ret, frame = cap.retrieve()
                if ret:
                    frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    valid_frame_indices.append(idx)
                    i += 1
                else:
                    # retrieve() failed even though grab() succeeded
                    logger.warning(
                        "Failed to retrieve frame %d during video loading. "
                        "This frame will be skipped.",
                        idx,
                    )

        valid_num_frames = len(valid_frame_indices)
        if valid_num_frames < num_expected_frames:
            logger.warning(
                "Video loading completed with %d broken/unreadable frames. "
                "Expected %d frames but only loaded %d frames.",
                num_expected_frames - valid_num_frames,
                num_expected_frames,
                valid_num_frames,
            )

        return frames[:valid_num_frames], valid_num_frames, valid_frame_indices


VIDEO_LOADER_REGISTRY = ExtensionManager()


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
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = -1,
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

        # resample video to target num_frames and fps
        # - the minimum of the two will be used
        num_frames_to_sample = total_frames_num
        if num_frames > 0:
            num_frames_to_sample = min(num_frames, total_frames_num)
        if fps > 0:
            num_frames_to_sample = min(num_frames_to_sample, math.floor(duration * fps))
        num_frames_to_sample = max(1, num_frames_to_sample)  # at least one sample

        if num_frames_to_sample == total_frames_num:
            frame_idx = list(range(0, num_frames_to_sample))
        else:
            uniform_sampled_frames = np.linspace(
                0, total_frames_num - 1, num_frames_to_sample, dtype=int
            )
            frame_idx = uniform_sampled_frames.tolist()

        # Convert to set for O(1) lookup performance
        frame_idx_set = set(frame_idx)
        frames, valid_num_frames, valid_frame_indices = cls._read_frames(
            cap, frame_idx_set, num_frames_to_sample, max(frame_idx)
        )

        # Use transformers transformers.video_utils.VideoMetadata format
        # NOTE(Isotr0py): For models like Qwen3-VL/GLM4.5V, this metadata
        # can cause incorrect timestamp calculation without num_frames=-1.
        metadata = {
            "total_num_frames": total_frames_num,
            "fps": original_fps,
            "duration": duration,
            "video_backend": "opencv",
            "frames_indices": valid_frame_indices,
            # extra field used to control hf processor's video
            # sampling behavior
            "do_sample_frames": valid_num_frames == total_frames_num,
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

        # Convert to set for O(1) lookup performance
        frame_indices_set = set(frame_indices_list)
        frames, valid_num_frames, valid_frame_indices = cls._read_frames(
            cap,
            frame_indices_set,
            len(frame_indices_list),
            total_frames_num - 1,
        )

        # Use transformers transformers.video_utils.VideoMetadata format
        metadata = {
            "total_num_frames": total_frames_num,
            "fps": original_fps,
            "duration": duration,
            "video_backend": "opencv_dynamic",
            "frames_indices": valid_frame_indices,
            "do_sample_frames": False,
        }

        return frames, metadata


@VIDEO_LOADER_REGISTRY.register("molmo2")
class Molmo2VideoBackend(VideoLoader):
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
    def _sample_frames(
        cls,
        total_num_frames: int,
        video_fps: float,
        duration: float,
        frame_sample_mode: str,
        num_frames: int,
        max_fps: int,
        sampling_fps: int,
    ) -> npt.NDArray:
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
        else:
            raise NotImplementedError(frame_sample_mode)

        return indices

    @classmethod
    def load_bytes_opencv(
        cls,
        data: bytes,
        frame_sample_mode: str | None = None,
        num_frames: int = -1,
        max_fps: int = 2,
        sampling_fps: int = 2,
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

        if frame_sample_mode is None:
            # Use transformers transformers.video_utils.VideoMetadata format
            frame_idx = list(range(0, total_frames_num))
            frame_idx_set = set(frame_idx)
            frames, valid_num_frames, valid_frame_indices = cls._read_frames(
                cap, frame_idx_set, total_frames_num, max(frame_idx)
            )
            do_sample_frames = valid_num_frames == total_frames_num
            metadata = {
                "total_num_frames": total_frames_num,
                "fps": original_fps,
                "duration": duration,
                "video_backend": "opencv",
                "do_sample_frames": do_sample_frames,
            }
            if not do_sample_frames:
                metadata["frames_indices"] = valid_frame_indices
            return frames, metadata

        frame_idx = cls._sample_frames(
            total_frames_num,
            original_fps,
            duration,
            frame_sample_mode,
            num_frames,
            max_fps,
            sampling_fps,
        ).tolist()

        frames, valid_num_frames, valid_frame_indices = cls._read_frames(
            cap,
            set(frame_idx),
            len(frame_idx),
            total_frames_num - 1,
        )

        metadata = {
            "total_num_frames": total_frames_num,
            "fps": original_fps,
            "duration": duration,
            "video_backend": "opencv",
            "frames_indices": valid_frame_indices,
            "do_sample_frames": False,
        }

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


class VideoMediaIO(MediaIO[tuple[npt.NDArray, dict[str, Any]]]):
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

        # Allow per-request override of video backend via kwargs.
        # This enables users to specify a different backend than the
        # global VLLM_VIDEO_LOADER_BACKEND env var, e.g.:
        #   --media-io-kwargs '{"video": {"video_backend": "torchcodec"}}'
        video_loader_backend = (
            kwargs.pop("video_backend", None) or envs.VLLM_VIDEO_LOADER_BACKEND
        )
        self.kwargs = kwargs
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
