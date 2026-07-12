# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from io import BytesIO

import numpy as np
import numpy.typing as npt

from vllm.utils.import_utils import PlaceholderModule

from .base import (
    VideoSourceMetadata,
    VideoTargetMetadata,
    check_frame_pixel_limit,
)

try:
    import av
except ImportError:
    av = PlaceholderModule("av")  # type: ignore[assignment]


def decode_pyav(
    loader_cls,
    data: bytes,
    target: VideoTargetMetadata,
    sampling_kwargs: dict,
) -> tuple[npt.NDArray, VideoSourceMetadata, list[int], list[int]]:
    with av.open(BytesIO(data)) as container:
        stream = container.streams.video[0]
        check_frame_pixel_limit(stream.width, stream.height)
        source = loader_cls._prepare_source(loader_cls.get_metadata(container))
        frame_idx = loader_cls.compute_frames_index_to_sample(
            source=source, target=target, **sampling_kwargs
        )
        frames, valid = loader_cls.decode_frames(
            container, frame_idx, source.original_fps, source.duration
        )
    return frames, source, frame_idx, valid


class PyAVVideoBackendMixin:
    """PyAV (in-process FFmpeg bindings) codec utilities.

    Reads stream metadata and decodes target frames via per-frame
    ``container.seek()``. The seek releases the GIL between frames and
    scales with the number of sampled frames rather than the video
    length, enabling concurrent decoding under serving load.
    """

    @staticmethod
    def get_metadata(
        container: "av.container.InputContainer",
    ) -> VideoSourceMetadata:
        if not container.streams.video:
            raise ValueError("No video streams found in container")
        stream = container.streams.video[0]
        total_frames = stream.frames or 0
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
        if total_frames == 0 and duration > 0 and fps > 0:
            total_frames = int(duration * fps)
        return VideoSourceMetadata(total_frames, fps, duration)

    @staticmethod
    def decode_frames(
        container: "av.container.InputContainer",
        frame_indices: list[int],
        fps: float,
        duration: float,
    ) -> tuple[npt.NDArray, list[int]]:
        """Decode target frames via per-frame seek + forward decode to PTS."""
        stream = container.streams.video[0]
        # SLICE parallelizes within a single frame without the
        # one-frame-per-thread latency penalty of FRAME threading.
        stream.thread_type = "SLICE"
        time_base = stream.time_base

        frames_list: list[npt.NDArray] = []
        valid_indices: list[int] = []
        frame_interval = 1.0 / fps if fps > 0 else 0.1
        max_ts = max(0.0, duration - frame_interval) if duration > 0 else float("inf")

        decoder = None
        last_pts = None
        for idx in frame_indices:
            ts = min(idx / fps, max_ts) if fps > 0 else 0.0
            pts = int(ts / time_base)
            # seek() snaps backward to a keyframe; reuse the running decoder
            # while targets advance monotonically to avoid re-decoding the
            # GOP prefix once per requested frame.
            if decoder is None or last_pts is None or pts <= last_pts:
                container.seek(pts, stream=stream)
                decoder = container.decode(video=0)
            chosen = None
            for frame in decoder:
                if frame.pts is not None and frame.pts >= pts:
                    chosen = frame
                    last_pts = frame.pts
                    break
            if chosen is not None:
                frames_list.append(chosen.to_ndarray(format="rgb24"))
                valid_indices.append(idx)
            else:
                decoder = None

        if not frames_list:
            return np.empty((0,), dtype=np.uint8), valid_indices
        return np.stack(frames_list), valid_indices
