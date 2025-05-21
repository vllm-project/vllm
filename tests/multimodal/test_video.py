# SPDX-License-Identifier: Apache-2.0
import numpy as np
import numpy.typing as npt
import pytest

from vllm.multimodal.video import VIDEO_LOADER_REGISTRY, VideoLoader

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
