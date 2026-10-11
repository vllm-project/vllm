# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Media decoding for Clef-Omni, as its ``joint_schema_model.py`` decodes it:
16 kHz mono audio, and video frames sampled at 2 fps with their soundtrack."""

import itertools
import math
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import TypeVar

import numpy as np
import pybase64

import vllm.envs as envs
from vllm.multimodal.media.base import MediaIO
from vllm.utils.import_utils import PlaceholderModule

try:
    import av
except ImportError:
    av = PlaceholderModule("av")  # type: ignore[assignment]

AUDIO_SAMPLE_RATE = 16_000
VIDEO_FPS = 2.0
MAX_VIDEO_FRAME_PIXELS = 256 * 32 * 32
#: The video processor's ``max_frames``.
MAX_VIDEO_FRAMES = 768

_T = TypeVar("_T")


class DecodedMediaIO(MediaIO[_T]):
    """Fetched bytes, decoded by ``decode``."""

    def __init__(
        self, decode: Callable[[bytes], _T], max_bytes: int | None = None
    ) -> None:
        self.decode = decode
        self.max_bytes = max_bytes

    def get_max_bytes(self) -> int | None:
        return self.max_bytes

    def load_bytes(self, data: bytes) -> _T:
        if self.max_bytes is not None and len(data) > self.max_bytes:
            raise ValueError(f"media is larger than {self.max_bytes} bytes")
        return self.decode(data)

    def load_base64(self, media_type: str, data: str) -> _T:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> _T:
        return self.load_bytes(filepath.read_bytes())


def decode_audio(data: bytes) -> np.ndarray | None:
    """The first audio stream as 16 kHz mono samples, or None without one."""
    max_samples = AUDIO_SAMPLE_RATE * envs.VLLM_MAX_AUDIO_DECODE_DURATION_S
    with av.open(BytesIO(data)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            return None
        resampler = av.AudioResampler(
            format="flt", layout="mono", rate=AUDIO_SAMPLE_RATE
        )
        chunks: list[np.ndarray] = []
        num_samples = 0
        # The final None flushes the resampler.
        for frame in itertools.chain(container.decode(stream), (None,)):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))
                num_samples += chunks[-1].size
            if max_samples and num_samples > max_samples:
                raise ValueError(
                    "audio is longer than "
                    f"{envs.VLLM_MAX_AUDIO_DECODE_DURATION_S} seconds"
                )
    if not chunks:
        return None
    samples = np.concatenate(chunks).astype(np.float32, copy=False)
    return samples if samples.size else None


def decode_audio_clip(data: bytes) -> np.ndarray:
    if (samples := decode_audio(data)) is None:
        raise ValueError("audio has no decodable audio stream")
    return samples


def decode_video(data: bytes) -> tuple[np.ndarray, np.ndarray | None]:
    """RGB frames sampled at 2 fps, each at most 256 * 32 * 32 pixels, padded
    to an even count, and the soundtrack if there is one."""
    frames: list[np.ndarray] = []
    with av.open(BytesIO(data)) as container:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise ValueError("video has no video stream")
        max_pixels = envs.VLLM_MAX_IMAGE_PIXELS
        if max_pixels and stream.width * stream.height > max_pixels:
            raise ValueError(f"video frames are larger than {max_pixels} pixels")
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.time is None or frame.time + 1e-6 < len(frames) / VIDEO_FPS:
                continue
            if len(frames) == MAX_VIDEO_FRAMES:
                raise ValueError(
                    f"video is longer than {MAX_VIDEO_FRAMES / VIDEO_FPS:.0f} seconds"
                )
            scale = min(
                1.0, math.sqrt(MAX_VIDEO_FRAME_PIXELS / (frame.width * frame.height))
            )
            width = max(2, round(frame.width * scale / 2) * 2)
            height = max(2, round(frame.height * scale / 2) * 2)
            frames.append(frame.to_ndarray(format="rgb24", width=width, height=height))
    if not frames:
        raise ValueError("video has no decodable frames")
    if len(frames) % 2:
        frames.append(frames[-1])
    return np.stack(frames), decode_audio(data)
