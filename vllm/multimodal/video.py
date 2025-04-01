# SPDX-License-Identifier: Apache-2.0

import base64
from functools import partial
from io import BytesIO
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from vllm.utils import PlaceholderModule

from .base import MediaIO
from .image import ImageMediaIO

try:
    import decord
except ImportError:
    decord = PlaceholderModule("decord")  # type: ignore[assignment]


def resize_video(frames: npt.NDArray, size: tuple[int, int]) -> npt.NDArray:
    num_frames, _, _, channels = frames.shape
    new_height, new_width = size
    resized_frames = np.empty((num_frames, new_height, new_width, channels),
                              dtype=frames.dtype)
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


def sample_frames_from_video(frames: npt.NDArray,
                             num_frames: int) -> npt.NDArray:
    total_frames = frames.shape[0]
    if num_frames == -1:
        return frames

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    sampled_frames = frames[frame_indices, ...]
    return sampled_frames


class VideoMediaIO(MediaIO[npt.NDArray]):

    def __init__(
        self,
        image_io: ImageMediaIO,
        *,
        num_frames: int = 32,
    ) -> None:
        super().__init__()

        self.image_io = image_io
        self.num_frames = num_frames

    def load_bytes(self, data: bytes) -> npt.NDArray:
        vr = decord.VideoReader(BytesIO(data), num_threads=1)
        total_frame_num = len(vr)

        num_frames = self.num_frames
        if total_frame_num > num_frames:
            uniform_sampled_frames = np.linspace(0,
                                                 total_frame_num - 1,
                                                 num_frames,
                                                 dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
        else:
            frame_idx = list(range(0, total_frame_num))

        return vr.get_batch(frame_idx).asnumpy()

    def load_base64(self, media_type: str, data: str) -> npt.NDArray:
        if media_type.lower() == "video/jpeg":
            load_frame = partial(
                self.image_io.load_base64,
                "image/jpeg",
            )

            return np.stack([
                np.array(load_frame(frame_data))
                for frame_data in data.split(",")
            ])

        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> npt.NDArray:
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

            return ",".join(
                encode_frame(Image.fromarray(frame)) for frame in video)

        msg = "Only JPEG format is supported for now."
        raise NotImplementedError(msg)
