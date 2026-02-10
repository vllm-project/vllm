# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import warnings
from abc import abstractmethod
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import cv2

from vllm.logger import init_logger
from vllm.utils.registry import ExtensionManager

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


@VIDEO_LOADER_REGISTRY.register("opencv_dynamic_openpangu")
class OpenCVDynamicOpenPanguVideoBackend(OpenCVVideoBackend):
    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = 32,
        fps: int = 1,
        max_duration: int = 300,
        frame_recovery: bool = False,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        """
        Load video frames with dynamic sampling based on duration.
        Assume that total_num_frames = 10 and fps = 1.
        The timestamp of frame 0 is 0.0.
        The timestamp of frame 1 is 1.0.…
        The timestamp of frame 9 (the last frame) should be 9.0, that is,
        (total_frames_num – 1) / original_fps.

        Args:
            data: Raw video bytes
            num_frames: Not used in dynamic backend
            fps: Target FPS for sampling (default: 2)

        Returns:
            Tuple of (frames_array, metadata_dict)
        """
        import cv2

        backend = cls().get_cv2_video_api()
        cap = cv2.VideoCapture(BytesIO(data), backend, [])
        if not cap.isOpened():
            raise ValueError("Could not open video stream")

        total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = float(cap.get(cv2.CAP_PROP_FPS))
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

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = np.empty((len(frames_indices), height, width, 3), dtype=np.uint8)

        i = 0
        for frame_idx in frames_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                i += 1
            else:
                # when get a bad frame,continuous finding a next good frame
                next_idx = frame_idx + 1
                while next_idx < total_frames_num:
                    ret, next_frame = cap.read()
                    if ret:
                        frames[i] = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)
                        i += 1
                        break
                    next_idx += 1

        if i != len(frames_indices):
            warnings.warn(
                f"Expected reading {len(frames_indices)} frames,"
                f"but only loaded {i} frames from video.",
                UserWarning,
                stacklevel=2,
            )

        # Use transformers transformers.video_utils.VideoMetadata format
        metadata = {
            "total_num_frames": total_frames_num,
            "fps": original_fps,
            "duration": total_duration,
            "video_backend": "opencv_dynamic_openpangu",
            "frames_indices": frames_indices,
            "do_sample_frames": False,
            "sample_frame_timestamps": sample_frame_timestamps,
        }
        return frames, metadata
