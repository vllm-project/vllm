# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image


def random_image(rng: np.random.RandomState, min_wh: int, max_wh: int):
    w, h = rng.randint(min_wh, max_wh, size=(2, ))
    arr = rng.randint(0, 255, size=(w, h, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def random_video(
    rng: np.random.RandomState,
    min_frames: int,
    max_frames: int,
    min_wh: int,
    max_wh: int,
):
    num_frames = rng.randint(min_frames, max_frames)
    w, h = rng.randint(min_wh, max_wh, size=(2, ))
    return rng.randint(0, 255, size=(num_frames, w, h, 3), dtype=np.uint8)


def random_audio(
    rng: np.random.RandomState,
    min_len: int,
    max_len: int,
    sr: int,
):
    audio_len = rng.randint(min_len, max_len)
    return rng.rand(audio_len), sr


def create_video_from_image(
    image_path: str,
    video_path: str,
    num_frames: int = 10,
    fps: float = 1.0,
):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    video_writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for _ in range(num_frames):
        video_writer.write(image)

    video_writer.release()
    return video_path


def cosine_similarity(A: npt.NDArray,
                      B: npt.NDArray,
                      axis: int = -1) -> npt.NDArray:
    """Compute cosine similarity between two vectors."""
    return (np.sum(A * B, axis=axis) /
            (np.linalg.norm(A, axis=axis) * np.linalg.norm(B, axis=axis)))


def normalize_image(image: npt.NDArray) -> npt.NDArray:
    """Normalize image to [0, 1] range."""
    return image.astype(np.float32) / 255.0