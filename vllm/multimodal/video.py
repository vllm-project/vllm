# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import math
from abc import abstractmethod
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from PIL import Image

if TYPE_CHECKING:
    import cv2

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
    def _can_use_for_recovery(
        idx: int,
        failed_frames: list[int],
        next_target_map: dict[int, int],
        total_frames: int,
    ) -> bool:
        """Check if current frame can recover the oldest failed frame."""
        if not failed_frames:
            return False
        oldest_failed = failed_frames[0]
        limit = next_target_map.get(oldest_failed, total_frames)
        return idx < limit

    @staticmethod
    def _read_frames_with_recovery(
        cap: "cv2.VideoCapture",
        frame_indices: list[int],
        total_frames: int,
    ) -> tuple[npt.NDArray, list[int], dict[int, int]]:
        """
        Read frames with dynamic window forward-scan recovery.

        When a target frame fails to load, the next successfully grabbed
        frame (before the next target frame) will be used to recover it.

        Args:
            cap: OpenCV VideoCapture object
            frame_indices: Sorted list of target frame indices to load
            total_frames: Total number of frames in the video

        Returns:
            Tuple of (frames_array, valid_frame_indices, recovered_map)
            - frames_array: Array of loaded frames
            - valid_frame_indices: List of frame indices that were loaded
            - recovered_map: Dict mapping recovered_idx -> source_idx
        """
        import cv2

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        assert width > 0 and height > 0, (
            f"Invalid video frame size: width={width}, height={height}"
        )

        frame_idx_set = set(frame_indices)
        max_frame_idx = frame_indices[-1] if frame_indices else 0

        # Build map: target_idx -> next_target_idx (for recovery window)
        next_target_map: dict[int, int] = {}
        for k in range(len(frame_indices) - 1):
            next_target_map[frame_indices[k]] = frame_indices[k + 1]
        next_target_map[frame_indices[-1]] = total_frames

        frames_list: list[npt.NDArray] = []
        valid_frame_indices: list[int] = []
        failed_frames_idx: list[int] = []
        recovered_map: dict[int, int] = {}

        i = 0
        for idx in range(max_frame_idx + 1):
            is_target_frame = idx in frame_idx_set

            # Attempt to grab the current frame
            ok = cap.grab()

            if not ok:
                if is_target_frame:
                    logger.warning(
                        "Failed to grab frame %d during video loading.",
                        idx,
                    )
                    failed_frames_idx.append(idx)
                continue

            # Check if we should retrieve: target frame OR can recover a failed one
            can_recover = VideoLoader._can_use_for_recovery(
                idx, failed_frames_idx, next_target_map, total_frames
            )

            if is_target_frame or can_recover:
                ret, frame = cap.retrieve()

                if ret and frame is not None and frame.size > 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_list.append(rgb_frame)
                    valid_frame_indices.append(idx)
                    i += 1

                    if can_recover:
                        recovered_idx = failed_frames_idx.pop(0)
                        recovered_map[recovered_idx] = idx
                        logger.info(
                            "Recovered frame %d using frame %d (delay: %d)",
                            recovered_idx,
                            idx,
                            idx - recovered_idx,
                        )
                elif is_target_frame:
                    logger.warning(
                        "Failed to retrieve frame %d during video loading.",
                        idx,
                    )
                    failed_frames_idx.append(idx)

        # Log any remaining failed frames
        for failed_idx in failed_frames_idx:
            logger.warning(
                "Frame %d could not be recovered (end of video).",
                failed_idx,
            )

        # Stack frames
        if frames_list:
            frames = np.stack(frames_list)
        else:
            frames = np.empty((0, height, width, 3), dtype=np.uint8)

        return frames, valid_frame_indices, recovered_map

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
        max_duration: int = 300,
        frame_recovery: bool = False,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        """
        Load video frames from bytes.

        Args:
            data: Raw video bytes
            num_frames: Target number of frames to sample (-1 for all)
            fps: Target FPS for sampling (-1 for original)
            max_duration: Maximum duration (unused in base backend)
            frame_recovery: Enable forward-scan recovery for failed frames

        Returns:
            Tuple of (frames_array, metadata_dict)
        """
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

        if frame_recovery:
            frames, valid_frame_indices, recovered_map = cls._read_frames_with_recovery(
                cap, frame_idx, total_frames_num
            )
            valid_num_frames = len(valid_frame_indices)

            if recovered_map:
                logger.info(
                    "Frame recovery: %d frames recovered using forward scan.",
                    len(recovered_map),
                )
        else:
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

        if frame_recovery:
            frames, valid_frame_indices, recovered_map = cls._read_frames_with_recovery(
                cap, frame_indices_list, total_frames_num
            )
            valid_num_frames = len(valid_frame_indices)

            if recovered_map:
                logger.info(
                    "Frame recovery: %d frames recovered using forward scan.",
                    len(recovered_map),
                )
        else:
            frame_indices_set = set(frame_indices_list)
            frames, valid_num_frames, valid_frame_indices = cls._read_frames(
                cap, frame_indices_set, len(frame_indices_list), total_frames_num - 1
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
