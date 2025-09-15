# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ClassVar, Literal, Optional

import cv2
import numpy as np
import numpy.typing as npt
from huggingface_hub import hf_hub_download
from PIL import Image

from vllm.utils import PlaceholderModule

from .base import get_cache_dir

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]


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

    num_frames = num_frames if num_frames > 0 else total_frames
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    for idx in range(total_frames):
        ok = cap.grab()  # next img
        if not ok:
            break
        if idx in frame_indices:  # only decompress needed
            ret, frame = cap.retrieve()
            if ret:
                # OpenCV uses BGR format, we need to convert it to RGB
                # for PIL and transformers compatibility
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    frames = np.stack(frames)
    if len(frames) < num_frames:
        raise ValueError(f"Could not read enough frames from video file {path}"
                         f" (expected {num_frames} frames, got {len(frames)})")
    return frames


def video_to_pil_images_list(path: str,
                             num_frames: int = -1) -> list[Image.Image]:
    frames = video_to_ndarrays(path, num_frames)
    return [Image.fromarray(frame) for frame in frames]


def video_get_metadata(path: str, num_frames: int = -1) -> dict[str, Any]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    if num_frames == -1 or num_frames > total_frames:
        num_frames = total_frames

    metadata = {
        "total_num_frames": num_frames,
        "fps": fps,
        "duration": duration,
        "video_backend": "opencv",
        "frames_indices": list(range(num_frames)),
        # extra field used to control hf processor's video
        # sampling behavior
        "do_sample_frames": num_frames == total_frames,
    }
    return metadata


VideoAssetName = Literal["baby_reading"]


@dataclass(frozen=True)
class VideoAsset:
    name: VideoAssetName
    num_frames: int = -1

    _NAME_TO_FILE: ClassVar[dict[VideoAssetName, str]] = {
        "baby_reading": "sample_demo_1.mp4",
    }

    @property
    def filename(self) -> str:
        return self._NAME_TO_FILE[self.name]

    @property
    def video_path(self) -> str:
        return download_video_asset(self.filename)

    @property
    def pil_images(self) -> list[Image.Image]:
        ret = video_to_pil_images_list(self.video_path, self.num_frames)
        return ret

    @property
    def np_ndarrays(self) -> npt.NDArray:
        ret = video_to_ndarrays(self.video_path, self.num_frames)
        return ret

    @property
    def metadata(self) -> dict[str, Any]:
        ret = video_get_metadata(self.video_path, self.num_frames)
        return ret

    def get_audio(self, sampling_rate: Optional[float] = None) -> npt.NDArray:
        """
        Read audio data from the video asset, used in Qwen2.5-Omni examples.
        
        See also: examples/offline_inference/qwen2_5_omni/only_thinker.py
        """
        return librosa.load(self.video_path, sr=sampling_rate)[0]
