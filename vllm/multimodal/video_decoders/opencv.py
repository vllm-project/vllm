# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from io import BytesIO

import numpy as np
import numpy.typing as npt

from vllm.logger import init_logger
from vllm.utils.import_utils import PlaceholderModule

from .base import (
    VideoSourceMetadata,
    VideoTargetMetadata,
    check_frame_pixel_limit,
)

try:
    import cv2
    import cv2.videoio_registry as vr
except ImportError:
    cv2 = PlaceholderModule("cv2")
    vr = PlaceholderModule("cv2").placeholder_attr("videoio_registry")

logger = init_logger(__name__)


def decode_opencv(
    loader_cls,
    data: bytes,
    target: VideoTargetMetadata,
    sampling_kwargs: dict,
    *,
    frame_recovery: bool = False,
) -> tuple[npt.NDArray, VideoSourceMetadata, list[int], list[int]]:
    cap = OpenCVVideoBackendMixin.open_video_capture(data)
    check_frame_pixel_limit(
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    source = loader_cls._prepare_source(OpenCVVideoBackendMixin.get_video_metadata(cap))
    frame_idx = loader_cls.compute_frames_index_to_sample(
        source=source, target=target, **sampling_kwargs
    )
    frames, valid = OpenCVVideoBackendMixin.read_frames(
        cap,
        frame_idx,
        total_frames_num=source.total_frames_num,
        frame_recovery=frame_recovery,
    )
    return frames, source, frame_idx, valid


class OpenCVVideoBackendMixin:
    @staticmethod
    def get_cv2_video_api():
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
    def open_video_capture(cls, data: bytes) -> "cv2.VideoCapture":
        backend = cls.get_cv2_video_api()
        cap = cv2.VideoCapture(BytesIO(data), backend, [])
        if not cap.isOpened():
            raise ValueError("Could not open video stream")
        return cap

    @staticmethod
    def get_video_metadata(cap: "cv2.VideoCapture") -> VideoSourceMetadata:
        total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames_num / original_fps if original_fps > 0 else 0
        return VideoSourceMetadata(
            total_frames_num=total_frames_num,
            original_fps=original_fps,
            duration=duration,
        )

    @classmethod
    def _can_use_for_recovery(
        cls,
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

    @classmethod
    def _read_frames_with_recovery(
        cls,
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
                    logger.debug(
                        "Failed to grab frame %d during video loading.",
                        idx,
                    )
                    failed_frames_idx.append(idx)
                continue

            # Check if we should retrieve: target frame OR can recover a failed one
            can_recover = cls._can_use_for_recovery(
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
                    logger.debug(
                        "Failed to retrieve frame %d during video loading.",
                        idx,
                    )
                    failed_frames_idx.append(idx)

        # Log any remaining failed frames
        for failed_idx in failed_frames_idx:
            logger.debug(
                "Frame %d could not be recovered (end of video).",
                failed_idx,
            )

        # Stack frames
        if frames_list:
            frames = np.stack(frames_list)
        else:
            frames = np.empty((0, height, width, 3), dtype=np.uint8)

        return frames, valid_frame_indices, recovered_map

    @classmethod
    def _read_frames_no_recovery(
        cls,
        cap,
        frame_indices: set[int],
        max_frame_idx: int,
    ) -> tuple[npt.NDArray, list[int]]:
        num_expected_frames = len(frame_indices)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = np.empty((num_expected_frames, height, width, 3), dtype=np.uint8)

        i = 0
        valid_frame_indices = []
        for idx in range(max_frame_idx + 1):
            ok = cap.grab()
            if not ok:
                # Frame is broken/unreadable, skip it
                if idx in frame_indices:
                    logger.debug(
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
                    logger.debug(
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

        return frames[:valid_num_frames], valid_frame_indices

    @classmethod
    def read_frames(
        cls,
        cap: "cv2.VideoCapture",
        frame_idx: list[int],
        total_frames_num: int,
        *,
        frame_recovery: bool = False,
    ) -> tuple[npt.NDArray, list[int]]:
        if frame_recovery:
            num_frames_to_sample = len(frame_idx)
            frames, valid_frame_indices, recovered_map = cls._read_frames_with_recovery(
                cap, frame_idx, total_frames_num
            )

            if recovered_map:
                logger.info(
                    "Frame recovery: %d frames recovered using forward scan.",
                    len(recovered_map),
                )
        else:
            frame_idx_set = set(frame_idx)
            num_frames_to_sample = len(frame_idx_set)
            frames, valid_frame_indices = cls._read_frames_no_recovery(
                cap, frame_idx_set, max(frame_idx)
            )
        valid_num_frames = len(valid_frame_indices)
        if valid_num_frames < num_frames_to_sample:
            logger.warning(
                "Video loading completed with %d broken/unreadable frames. "
                "Expected to sample %d frames but only loaded %d frames.",
                num_frames_to_sample - valid_num_frames,
                num_frames_to_sample,
                valid_num_frames,
            )
        return frames, valid_frame_indices
