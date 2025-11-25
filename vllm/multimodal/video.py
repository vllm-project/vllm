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

    @classmethod
    def _load_frames_with_seeking(
        cls,
        cap,
        frame_indices: list[int],
        recovery_offset: int,
        existing_frames: dict[int, npt.NDArray] | None = None,
    ) -> tuple[npt.NDArray, list[int], dict[int, int]]:
        """
        Optimized Recovery Logic:
        1. Cache Check (Zero I/O): Reuse frames from sequential read if close enough.
        2. Forward Scan (Low I/O): Seek once to target+1, grab forward (handles gaps).
        3. Backward Scan (Med I/O): Seek to target-offset, grab forward (truncation).
        """
        import cv2

        if existing_frames is None:
            existing_frames = {}

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames_list = []
        loaded_indices = []
        recovered_frames = {}

        for target_idx in frame_indices:
            # 1. Attempt direct seek to the target
            # Direct seek might work if sequential failed due to buffering.
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()

            if ret:
                frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                loaded_indices.append(target_idx)
                continue

            # 2. Primary Load Failed. Start Recovery.
            if recovery_offset <= 0:
                logger.warning("Frame %d failed, no recovery enabled.", target_idx)
                continue

            logger.warning(
                "Frame %d failed, attempting recovery (offset=%d)",
                target_idx,
                recovery_offset,
            )

            found_recovery = False

            # --- Strategy A: Check Cache (History) ---
            # Fastest: Zero I/O. Reuse sequential frames we already hold.
            for i in range(1, recovery_offset + 1):
                recovery_idx = target_idx - i
                if recovery_idx in existing_frames:
                    # Found valid frame in history
                    frames_list.append(existing_frames[recovery_idx])  # Already RGB
                    loaded_indices.append(target_idx)
                    recovered_frames[target_idx] = recovery_idx

                    logger.warning(
                        "Recovered frame %d using cached frame %d (Zero-Seek)",
                        target_idx,
                        recovery_idx,
                    )
                    found_recovery = True
                    break

            if found_recovery:
                continue

            # --- Strategy B: Check Future (Forward Scan) ---
            # Optimization: Seek ONCE to (target+1), then grab() forward.
            # Handles "Corruption Gaps" where a few frames are broken.
            start_forward_scan = target_idx + 1
            if start_forward_scan < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_forward_scan)
                for i in range(1, recovery_offset + 1):
                    recovery_idx = target_idx + i
                    if recovery_idx >= total_frames:
                        break

                    # Stop if we hit any target frame (missing or existing).
                    if (recovery_idx in frame_indices) or (
                        recovery_idx in existing_frames
                    ):
                        break

                    ret_rec, frame_rec = cap.read()

                    if ret_rec:
                        frames_list.append(cv2.cvtColor(frame_rec, cv2.COLOR_BGR2RGB))
                        loaded_indices.append(target_idx)
                        recovered_frames[target_idx] = recovery_idx
                        found_recovery = True
                        logger.warning(
                            "Recovered frame %d using future frame %d (Forward Scan)",
                            target_idx,
                            recovery_idx,
                        )
                        break

            if found_recovery:
                continue

            # --- Strategy C: Check Past (Backward Scan) ---
            # Optimization: Seek ONCE to (target-offset), then read forward.
            # Handles "Truncation" where file ends early. We find the LAST valid frame.
            start_back_scan = max(0, target_idx - recovery_offset)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_back_scan)

            last_valid_frame = None
            last_valid_idx = None

            # Scan from start_back up to target
            frames_to_check = target_idx - start_back_scan
            for i in range(frames_to_check):
                curr_check = start_back_scan + i
                ret_rec, frame_rec = cap.read()
                if ret_rec:
                    last_valid_frame = frame_rec
                    last_valid_idx = curr_check

            if last_valid_frame is not None and last_valid_idx is not None:
                frames_list.append(cv2.cvtColor(last_valid_frame, cv2.COLOR_BGR2RGB))
                loaded_indices.append(target_idx)
                recovered_frames[target_idx] = last_valid_idx
                found_recovery = True
                logger.warning(
                    "Recovered frame %d using past frame %d (Backward Scan)",
                    target_idx,
                    last_valid_idx,
                )

            if not found_recovery:
                logger.warning(
                    "Frame %d failed to load after all recovery attempts", target_idx
                )

        if frames_list:
            frames = np.stack(frames_list)
        else:
            frames = np.empty((0, height, width, 3), dtype=np.uint8)

        return frames, loaded_indices, recovered_frames

    @staticmethod
    def _read_frames(
        cap,
        frame_indices: set[int],
        num_expected_frames: int,
        max_frame_idx: int,
        warn_on_failure: bool = True,
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
                # Frame is broken/unreadable
                if idx in frame_indices and warn_on_failure:
                    logger.warning(
                        "Failed to grab frame %d during video loading."
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
                    if warn_on_failure:
                        logger.warning(
                            "Failed to retrieve frame %d during video loading."
                            "This frame will be skipped.",
                            idx,
                        )

        valid_num_frames = len(valid_frame_indices)
        if valid_num_frames < num_expected_frames and warn_on_failure:
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
        recovery_offset: int = 0,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        if recovery_offset < 0:
            raise ValueError(
                f"recovery_offset must be non-negative, got {recovery_offset}"
            )
        import cv2

        backend = cls().get_cv2_video_api()
        stream = BytesIO(data)
        cap = cv2.VideoCapture(stream, backend, [])
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
        num_frames_to_sample = max(1, num_frames_to_sample)

        if num_frames_to_sample == total_frames_num:
            frame_idx = list(range(0, num_frames_to_sample))
        else:
            uniform_sampled_frames = np.linspace(
                0, total_frames_num - 1, num_frames_to_sample, dtype=int
            )
            frame_idx = uniform_sampled_frames.tolist()

        # Convert to set for O(1) lookup performance
        frame_idx_set = set(frame_idx)

        # Only warn in sequential read if we are NOT going to try recovery
        should_warn = recovery_offset == 0

        frames, valid_num_frames, valid_frame_indices = cls._read_frames(
            cap,
            frame_idx_set,
            num_frames_to_sample,
            max(frame_idx),
            warn_on_failure=should_warn,
        )

        # Recovery Logic
        if valid_num_frames < len(frame_idx) and recovery_offset > 0:
            missing_indices = sorted(list(frame_idx_set - set(valid_frame_indices)))

            logger.warning(
                "Sequential loading missing %d frames. Attempting recovery...",
                len(missing_indices),
            )

            # Build Cache from successfully loaded frames
            existing_frames_map = {
                idx: frames[i] for i, idx in enumerate(valid_frame_indices)
            }

            cap.release()
            recovery_stream = BytesIO(data)
            cap = cv2.VideoCapture(recovery_stream, backend, [])

            frames_seek, loaded_indices_seek, recovered_map = (
                cls._load_frames_with_seeking(
                    cap,
                    missing_indices,
                    recovery_offset,
                    existing_frames=existing_frames_map,
                )
            )

            if len(loaded_indices_seek) > 0:
                # 1. Concatenate un-ordered
                frames = np.concatenate((frames, frames_seek), axis=0)
                valid_frame_indices.extend(loaded_indices_seek)
                valid_num_frames += len(loaded_indices_seek)

                # 2. Sort by index to restore Temporal Order
                sorted_order = np.argsort(valid_frame_indices)
                frames = frames[sorted_order]
                valid_frame_indices = [valid_frame_indices[i] for i in sorted_order]

            remaining_missing = len(frame_idx) - valid_num_frames
            if remaining_missing == 0:
                logger.warning(
                    "Recovery successful. All %d frames loaded.", len(frame_idx)
                )
            else:
                logger.warning(
                    "Recovery finished but video is still missing %d frames.",
                    remaining_missing,
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
        recovery_offset: int = 0,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        if recovery_offset < 0:
            raise ValueError(
                f"recovery_offset must be non-negative, got {recovery_offset}"
            )
        import cv2

        backend = cls().get_cv2_video_api()
        stream = BytesIO(data)
        cap = cv2.VideoCapture(stream, backend, [])
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

        # Only warn in sequential read if we are NOT going to try recovery
        should_warn = recovery_offset == 0

        frames, valid_num_frames, valid_frame_indices = cls._read_frames(
            cap,
            frame_indices_set,
            len(frame_indices_list),
            total_frames_num - 1,
            warn_on_failure=should_warn,
        )

        # Recovery Logic
        if valid_num_frames < len(frame_indices_list) and recovery_offset > 0:
            missing_indices = sorted(list(frame_indices_set - set(valid_frame_indices)))

            logger.warning(
                "Sequential loading missing %d frames. Attempting recovery...",
                len(missing_indices),
            )

            # Build Cache from successfully loaded frames
            existing_frames_map = {
                idx: frames[i] for i, idx in enumerate(valid_frame_indices)
            }

            cap.release()
            recovery_stream = BytesIO(data)
            cap = cv2.VideoCapture(recovery_stream, backend, [])

            frames_seek, loaded_indices_seek, recovered_map = (
                cls._load_frames_with_seeking(
                    cap,
                    missing_indices,
                    recovery_offset,
                    existing_frames=existing_frames_map,
                )
            )

            if len(loaded_indices_seek) > 0:
                # 1. Concatenate un-ordered
                frames = np.concatenate((frames, frames_seek), axis=0)
                valid_frame_indices.extend(loaded_indices_seek)
                valid_num_frames += len(loaded_indices_seek)

                # 2. Sort by index to restore Temporal Order
                sorted_order = np.argsort(valid_frame_indices)
                frames = frames[sorted_order]
                valid_frame_indices = [valid_frame_indices[i] for i in sorted_order]

            remaining_missing = len(frame_indices_list) - valid_num_frames
            if remaining_missing == 0:
                logger.warning(
                    "Recovery successful. All %d frames loaded.",
                    len(frame_indices_list),
                )
            else:
                logger.warning(
                    "Recovery finished but video is still missing %d frames.",
                    remaining_missing,
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
