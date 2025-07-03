# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
from abc import abstractmethod
from functools import partial
from io import BytesIO
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from vllm import envs

from .base import MediaIO
from .image import ImageMediaIO


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


class VideoLoader:

    @classmethod
    @abstractmethod
    def load_bytes(cls, data: bytes, num_frames: int = -1) -> npt.NDArray:
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
                if (abi < 1 or (abi == 1 and api < 2)):
                    continue
            api_pref = backend
            break
        return api_pref

    @classmethod
    def load_bytes(cls, data: bytes, num_frames: int = -1) -> npt.NDArray:
        import cv2

        backend = cls().get_cv2_video_api()
        cap = cv2.VideoCapture(BytesIO(data), backend, [])
        if not cap.isOpened():
            raise ValueError("Could not open video stream")

        total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        full_read = num_frames == -1 or total_frames_num < num_frames
        if full_read:
            num_frames = total_frames_num
            frame_idx = list(range(0, num_frames))
        else:
            uniform_sampled_frames = np.linspace(0,
                                                 total_frames_num - 1,
                                                 num_frames,
                                                 dtype=int)
            frame_idx = uniform_sampled_frames.tolist()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = np.empty((len(frame_idx), height, width, 3), dtype=np.uint8)

        i = 0
        for idx in range(total_frames_num):
            ok = cap.grab()  # next img
            if not ok:
                break
            if idx in frame_idx:  # only decompress needed
                ret, frame = cap.retrieve()
                if ret:
                    frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    i += 1
        # we expect all frames loaded
        assert i == num_frames, (f"Expected reading {num_frames} frames, "
                                 f"but only loaded {i} frames from video.")
        return frames


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
        video_loader_backend = envs.VLLM_VIDEO_LOADER_BACKEND
        self.video_loader = VIDEO_LOADER_REGISTRY.load(video_loader_backend)

    def load_bytes(self, data: bytes) -> npt.NDArray:
        return self.video_loader.load_bytes(data, self.num_frames)

    def load_base64(self, media_type: str, data: str) -> npt.NDArray:
        if media_type.lower() == "video/jpeg":
            load_frame = partial(
                self.image_io.load_base64,
                "image/jpeg",
            )

            return np.stack([
                np.asarray(load_frame(frame_data))
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
