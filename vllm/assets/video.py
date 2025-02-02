# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Literal

import cv2
import numpy as np
import numpy.typing as npt
from huggingface_hub import hf_hub_download
from PIL import Image

from vllm.multimodal.video import sample_frames_from_video

from .base import get_cache_dir


@lru_cache
def download_video_asset(filename: str) -> str:
    """
    Download and open an image from huggingface
    repo: raushan-testing-hf/videos-test
    """
    video_directory = get_cache_dir() / "video-example-data"
    video_directory.mkdir(parents=True, exist_ok=True)

    video_path = video_directory / filename
    video_path_str = str(video_path)
    if not video_path.exists():
        video_path_str = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test",
            filename=filename,
            repo_type="dataset",
            cache_dir=video_directory,
        )
    return video_path_str


def video_to_ndarrays(path: str, num_frames: int = -1) -> npt.NDArray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    frames = np.stack(frames)
    frames = sample_frames_from_video(frames, num_frames)
    if len(frames) < num_frames:
        raise ValueError(f"Could not read enough frames from video file {path}"
                         f" (expected {num_frames} frames, got {len(frames)})")
    return frames


def video_to_pil_images_list(path: str,
                             num_frames: int = -1) -> List[Image.Image]:
    frames = video_to_ndarrays(path, num_frames)
    return [
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for frame in frames
    ]


@dataclass(frozen=True)
class VideoAsset:
    name: Literal["sample_demo_1.mp4"]
    num_frames: int = -1

    @property
    def pil_images(self) -> List[Image.Image]:
        video_path = download_video_asset(self.name)
        ret = video_to_pil_images_list(video_path, self.num_frames)
        return ret

    @property
    def np_ndarrays(self) -> npt.NDArray:
        video_path = download_video_asset(self.name)
        ret = video_to_ndarrays(video_path, self.num_frames)
        return ret
