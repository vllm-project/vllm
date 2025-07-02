# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import numpy.typing as npt
import pytest

from vllm import envs
from vllm.multimodal.image import ImageMediaIO
from vllm.multimodal.video import (VIDEO_LOADER_REGISTRY, VideoLoader,
                                   VideoMediaIO)

NUM_FRAMES = 10
FAKE_OUTPUT_1 = np.random.rand(NUM_FRAMES, 1280, 720, 3)
FAKE_OUTPUT_2 = np.random.rand(NUM_FRAMES, 1280, 720, 3)


@VIDEO_LOADER_REGISTRY.register("test_video_loader_1")
class TestVideoLoader1(VideoLoader):

    @classmethod
    def load_bytes(cls, data: bytes, num_frames: int = -1) -> npt.NDArray:
        return FAKE_OUTPUT_1


@VIDEO_LOADER_REGISTRY.register("test_video_loader_2")
class TestVideoLoader2(VideoLoader):

    @classmethod
    def load_bytes(cls, data: bytes, num_frames: int = -1) -> npt.NDArray:
        return FAKE_OUTPUT_2


def test_video_loader_registry():
    custom_loader_1 = VIDEO_LOADER_REGISTRY.load("test_video_loader_1")
    output_1 = custom_loader_1.load_bytes(b"test")
    np.testing.assert_array_equal(output_1, FAKE_OUTPUT_1)

    custom_loader_2 = VIDEO_LOADER_REGISTRY.load("test_video_loader_2")
    output_2 = custom_loader_2.load_bytes(b"test")
    np.testing.assert_array_equal(output_2, FAKE_OUTPUT_2)


def test_video_loader_type_doesnt_exist():
    with pytest.raises(AssertionError):
        VIDEO_LOADER_REGISTRY.load("non_existing_video_loader")


@VIDEO_LOADER_REGISTRY.register("assert_10_frames_1_fps")
class Assert10Frames1FPSVideoLoader(VideoLoader):

    @classmethod
    def load_bytes(cls,
                   data: bytes,
                   num_frames: int = -1,
                   fps: float = -1.0,
                   **kwargs) -> npt.NDArray:
        assert num_frames == 10, "bad num_frames"
        assert fps == 1.0, "bad fps"
        return FAKE_OUTPUT_2


def test_video_media_io_kwargs():
    envs.VLLM_VIDEO_LOADER_BACKEND = "assert_10_frames_1_fps"
    imageio = ImageMediaIO()

    # Verify that different args pass/fail assertions as expected.
    videoio = VideoMediaIO(imageio, **{"num_frames": 10, "fps": 1.0})
    _ = videoio.load_bytes(b"test")

    videoio = VideoMediaIO(
        imageio, **{
            "num_frames": 10,
            "fps": 1.0,
            "not_used": "not_used"
        })
    _ = videoio.load_bytes(b"test")

    with pytest.raises(AssertionError, match="bad num_frames"):
        videoio = VideoMediaIO(imageio, **{})
        _ = videoio.load_bytes(b"test")

    with pytest.raises(AssertionError, match="bad num_frames"):
        videoio = VideoMediaIO(imageio, **{"num_frames": 9, "fps": 1.0})
        _ = videoio.load_bytes(b"test")

    with pytest.raises(AssertionError, match="bad fps"):
        videoio = VideoMediaIO(imageio, **{"num_frames": 10, "fps": 2.0})
        _ = videoio.load_bytes(b"test")
