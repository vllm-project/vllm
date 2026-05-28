# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from vllm.assets.base import get_vllm_public_assets
from vllm.multimodal.video import (
    VIDEO_LOADER_REGISTRY,
    VideoLoader,
)

from .utils import create_long_gop_video, create_video_from_image

pytestmark = pytest.mark.cpu_test

ASSETS_DIR = Path(__file__).parent / "assets"
assert ASSETS_DIR.exists()

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


def test_video_backend_handles_broken_frames(monkeypatch: pytest.MonkeyPatch):
    """
    Regression test for handling videos with broken frames.
    This test uses a pre-corrupted video file (assets/corrupted.mp4) that
    contains broken frames to verify the video loader handles
    them gracefully without crashing and returns accurate metadata.
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv")

        # Load the pre-corrupted video file that contains broken frames
        corrupted_video_path = ASSETS_DIR / "corrupted.mp4"

        with open(corrupted_video_path, "rb") as f:
            video_data = f.read()

        loader = VIDEO_LOADER_REGISTRY.load("opencv")
        frames, metadata = loader.load_bytes(
            video_data, num_frames=-1, backend="opencv"
        )

        # Verify metadata consistency:
        # frames_indices must match actual loaded frames
        assert frames.shape[0] == len(metadata["frames_indices"]), (
            f"Frames array size must equal frames_indices length. "
            f"Got {frames.shape[0]} frames but "
            f"{len(metadata['frames_indices'])} indices"
        )

        # Verify that broken frames were skipped:
        # loaded frames should be less than total
        assert frames.shape[0] < metadata["total_num_frames"], (
            f"Should load fewer frames than total due to broken frames. "
            f"Expected fewer than {metadata['total_num_frames']} frames, "
            f"but loaded {frames.shape[0]} frames"
        )


# ============================================================================
# Frame Recovery Tests
# ============================================================================


def test_video_recovery_simulated_failures(monkeypatch: pytest.MonkeyPatch):
    """
    Test that frame recovery correctly uses the next valid frame when
    target frames fail to load.

    Uses corrupted.mp4 and mocks VideoCapture.grab() to fail on specific
    frame indices (in addition to the real corruption at frame 17), then
    verifies recovery produces more frames.
    """
    import cv2

    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv")

        # Load corrupted.mp4 (26 frames, frame 17 is genuinely corrupted)
        video_path = ASSETS_DIR / "corrupted.mp4"
        with open(video_path, "rb") as f:
            video_data = f.read()

        # Simulate additional failures on frames 3 and 10
        # (in addition to the real corruption at frame 17)
        fail_on_frames = {3, 10}

        # Store original VideoCapture class
        original_video_capture = cv2.VideoCapture

        class MockVideoCapture:
            """Wrapper that simulates grab() failures on specific frames."""

            def __init__(self, *args, **kwargs):
                self._cap = original_video_capture(*args, **kwargs)
                self._current_frame = -1

            def grab(self):
                self._current_frame += 1
                if self._current_frame in fail_on_frames:
                    return False  # Simulate failure
                return self._cap.grab()

            def retrieve(self):
                return self._cap.retrieve()

            def get(self, prop):
                return self._cap.get(prop)

            def isOpened(self):
                return self._cap.isOpened()

            def release(self):
                return self._cap.release()

        # Patch cv2.VideoCapture
        m.setattr(cv2, "VideoCapture", MockVideoCapture)

        loader = VIDEO_LOADER_REGISTRY.load("opencv")

        # Use num_frames=8 which samples: [0, 3, 7, 10, 14, 17, 21, 25]
        # Frame 3: mocked failure, recovery window [3, 7) -> use frame 4
        # Frame 10: mocked failure, recovery window [10, 14) -> use frame 11
        # Frame 17: real corruption, recovery window [17, 21) -> use frame 18

        # Test WITHOUT recovery - should have fewer frames due to failures
        frames_no_recovery, meta_no = loader.load_bytes(
            video_data, num_frames=8, frame_recovery=False, backend="opencv"
        )

        # Test WITH recovery - should recover using next valid frames
        frames_with_recovery, meta_yes = loader.load_bytes(
            video_data, num_frames=8, frame_recovery=True, backend="opencv"
        )

        # With recovery should have MORE frames than without
        # Without: 5 frames (3, 10, 17 all fail)
        # With: 8 frames (all recovered)
        assert frames_with_recovery.shape[0] > frames_no_recovery.shape[0], (
            f"Recovery should produce more frames. "
            f"Without: {frames_no_recovery.shape[0]}, "
            f"With: {frames_with_recovery.shape[0]}"
        )

        # Verify metadata consistency
        assert frames_no_recovery.shape[0] == len(meta_no["frames_indices"])
        assert frames_with_recovery.shape[0] == len(meta_yes["frames_indices"])

        # Verify temporal order is preserved
        assert meta_yes["frames_indices"] == sorted(meta_yes["frames_indices"])


def test_video_recovery_with_corrupted_file(monkeypatch: pytest.MonkeyPatch):
    """
    Test frame recovery with an actual corrupted video file using sparse sampling.

    This test uses corrupted.mp4 which has genuine H.264 codec errors on
    frame 17. With num_frames=8, the target frames are [0, 3, 7, 10, 14, 17, 21, 25].
    Frame 17 is corrupted but frames 18-20 are readable, so recovery can use
    frame 18 to fill in for the failed frame 17.

    This test verifies:
    1. Without recovery: frame 17 is skipped (7 frames loaded)
    2. With recovery: frame 18 fills in for frame 17 (8 frames loaded)
    3. Recovery produces MORE frames than without recovery
    4. Metadata is consistent with loaded frames
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv")

        corrupted_video_path = ASSETS_DIR / "corrupted.mp4"

        with open(corrupted_video_path, "rb") as f:
            video_data = f.read()

        loader = VIDEO_LOADER_REGISTRY.load("opencv")

        # Use num_frames=8 which makes frame 17 a target with recovery window [17, 21)
        # Target frames: [0, 3, 7, 10, 14, 17, 21, 25]
        # Frame 17 is corrupted, but frames 18-20 are readable for recovery

        # Test without recovery - frame 17 will be skipped
        frames_no_recovery, meta_no_recovery = loader.load_bytes(
            video_data, num_frames=8, frame_recovery=False, backend="opencv"
        )

        # Test with recovery - frame 18 should fill in for frame 17
        frames_with_recovery, meta_with_recovery = loader.load_bytes(
            video_data, num_frames=8, frame_recovery=True, backend="opencv"
        )

        # Verify metadata consistency for both modes
        assert frames_no_recovery.shape[0] == len(meta_no_recovery["frames_indices"]), (
            "Frame count must match indices without recovery"
        )
        assert frames_with_recovery.shape[0] == len(
            meta_with_recovery["frames_indices"]
        ), "Frame count must match indices with recovery"

        # KEY ASSERTION: Recovery should produce MORE frames than without recovery
        # Without recovery: 7 frames (frame 17 skipped)
        # With recovery: 8 frames (frame 18 used for frame 17)
        assert frames_with_recovery.shape[0] > frames_no_recovery.shape[0], (
            f"Recovery should produce more frames with sparse sampling. "
            f"Got {frames_with_recovery.shape[0]} with recovery vs "
            f"{frames_no_recovery.shape[0]} without"
        )

        # Verify we got all 8 requested frames with recovery
        assert frames_with_recovery.shape[0] == 8, (
            f"With recovery, should load all 8 requested frames. "
            f"Got {frames_with_recovery.shape[0]}"
        )

        # Verify the video metadata is correct
        expected_total_frames = 26
        assert meta_with_recovery["total_num_frames"] == expected_total_frames, (
            f"Expected {expected_total_frames} total frames in metadata"
        )


def test_video_recovery_dynamic_backend(monkeypatch: pytest.MonkeyPatch):
    """
    Test that frame_recovery works with the dynamic video backend.

    The dynamic backend samples frames based on fps/duration rather than
    loading all frames. This test verifies recovery works in that context.
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv_dynamic")

        corrupted_video_path = ASSETS_DIR / "corrupted.mp4"

        with open(corrupted_video_path, "rb") as f:
            video_data = f.read()

        loader = VIDEO_LOADER_REGISTRY.load("opencv_dynamic")

        # Test without recovery
        frames_no_recovery, meta_no = loader.load_bytes(
            video_data,
            fps=2,
            max_duration=10,
            frame_recovery=False,
            backend="opencv",
        )

        # Test with frame_recovery enabled
        frames_with_recovery, meta_with = loader.load_bytes(
            video_data, fps=2, max_duration=10, frame_recovery=True, backend="opencv"
        )

        # Verify basic properties
        assert frames_no_recovery.shape[0] > 0, (
            "Should load some frames without recovery"
        )
        assert frames_with_recovery.shape[0] > 0, (
            "Should load some frames with recovery"
        )
        assert "do_sample_frames" in meta_with
        assert meta_with["do_sample_frames"] is False  # Dynamic backend always False
        assert frames_with_recovery.shape[0] == len(meta_with["frames_indices"])

        # Key assertion: recovery should help when corrupted frames are sampled
        # We expect recovery to produce >= frames than without recovery
        assert frames_with_recovery.shape[0] >= frames_no_recovery.shape[0], (
            f"Recovery should produce at least as many frames. "
            f"Got {frames_with_recovery.shape[0]} with recovery vs "
            f"{frames_no_recovery.shape[0]} without"
        )


@pytest.fixture
def dummy_video_path(tmp_path):
    image_path = get_vllm_public_assets(
        filename="stop_sign.jpg", s3_prefix="vision_model_images"
    )

    video_path = tmp_path / "test_RGB_video.mp4"
    create_video_from_image(str(image_path), str(video_path), num_frames=1800, fps=30)
    return video_path


# ============================================================================
# PyAV Backend Tests
# ============================================================================


def test_pyav_backend_loads_frames(dummy_video_path, monkeypatch: pytest.MonkeyPatch):
    """Test that the pyav codec backend can load frames from a valid video."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv")

        with open(dummy_video_path, "rb") as f:
            video_data = f.read()

        loader = VIDEO_LOADER_REGISTRY.load("opencv")
        frames, metadata = loader.load_bytes(video_data, num_frames=8, backend="pyav")

        assert frames.ndim == 4
        assert frames.shape[3] == 3  # RGB
        assert frames.shape[0] == 8
        assert frames.shape[0] == len(metadata["frames_indices"])
        assert metadata["video_backend"] == "pyav"
        assert "total_num_frames" in metadata
        assert "fps" in metadata
        assert "duration" in metadata


def test_pyav_dynamic_backend_loads_frames(
    dummy_video_path, monkeypatch: pytest.MonkeyPatch
):
    """Test that the pyav codec with dynamic sampling can load frames."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv_dynamic")

        with open(dummy_video_path, "rb") as f:
            video_data = f.read()

        loader = VIDEO_LOADER_REGISTRY.load("opencv_dynamic")
        frames, metadata = loader.load_bytes(
            video_data, fps=2, max_duration=10, backend="pyav"
        )

        assert frames.ndim == 4
        assert frames.shape[3] == 3  # RGB
        assert frames.shape[0] > 0
        assert frames.shape[0] == len(metadata["frames_indices"])
        assert metadata["video_backend"] == "pyav_dynamic"


def test_pyav_backend_returns_target_frames_not_keyframes():
    """Regression test: PyAV must decode forward past the seek keyframe.

    container.seek() snaps backward to the nearest keyframe. With a long GOP
    (here: one keyframe at frame 0), a decoder that does not advance forward
    to the target PTS collapses every sampled slot onto the keyframe. This
    test encodes a per-frame marker on the green channel and verifies the
    returned frames are distinct, ordered, and match the requested indices.
    """
    num_frames = 50
    num_sampled = 4
    height, width = 64, 64

    video_bytes = create_long_gop_video(
        num_frames=num_frames, width=width, height=height
    )

    loader = VIDEO_LOADER_REGISTRY.load("opencv")
    frames, metadata = loader.load_bytes(
        video_bytes, num_frames=num_sampled, backend="pyav"
    )
    assert frames.shape == (num_sampled, height, width, 3)

    requested = list(metadata["frames_indices"])
    assert len(requested) == num_sampled

    actual = [int(f[height // 2, width // 2, 1]) for f in frames]

    assert len(set(actual)) == num_sampled, (
        f"PyAV returned only {len(set(actual))} distinct frames for "
        f"{num_sampled} requested indices: markers={actual}, "
        f"requested={requested}. Keyframe-snap regression."
    )

    assert actual == sorted(actual), f"Returned frames out of order: markers={actual}"

    for marker, want_idx in zip(actual, requested):
        assert abs(marker - want_idx) <= 10, (
            f"Frame mismatch: requested index {want_idx}, "
            f"got marker {marker} (tolerance ±10)"
        )


@pytest.mark.parametrize(
    "loader_key, kwargs, expected_num_frames",
    [
        # uniform sampling + opencv codec
        pytest.param(
            "opencv",
            {"num_frames": 32, "backend": "opencv"},
            32,
            id="opencv-num_frames",
        ),
        pytest.param("opencv", {"fps": 2, "backend": "opencv"}, 120, id="opencv-fps"),
        pytest.param(
            "opencv",
            {"num_frames": 500, "fps": 2, "backend": "opencv"},
            120,
            id="opencv-num_frames_wins_fps",
        ),
        # dynamic sampling + opencv codec
        pytest.param(
            "opencv_dynamic",
            {"fps": 1, "max_duration": 60, "backend": "opencv"},
            60,
            id="opencv_dynamic-within_max_duration",
        ),
        pytest.param(
            "opencv_dynamic",
            {"fps": 2, "max_duration": 30, "backend": "opencv"},
            60,
            id="opencv_dynamic-exceeds_max_duration",
        ),
        pytest.param(
            "openpangu", {"num_frames": 32, "fps": -1}, 32, id="openpangu-num_frames"
        ),
        pytest.param(
            "molmo2",
            {"num_frames": 32, "frame_sample_mode": "uniform_last_frame"},
            32,
            id="molmo2-uniform_last_frame",
        ),
        pytest.param(
            "molmo2",
            {"fps": 2, "frame_sample_mode": "fps"},
            119,
            id="molmo2-fps",
        ),
        # uniform sampling + pyav codec (same frame counts as opencv)
        pytest.param(
            "opencv",
            {"num_frames": 32, "backend": "pyav"},
            32,
            id="pyav-num_frames",
        ),
        pytest.param("opencv", {"fps": 2, "backend": "pyav"}, 120, id="pyav-fps"),
        pytest.param(
            "opencv",
            {"num_frames": 500, "fps": 2, "backend": "pyav"},
            120,
            id="pyav-num_frames_wins_fps",
        ),
        # dynamic sampling + pyav codec
        pytest.param(
            "opencv_dynamic",
            {"fps": 1, "max_duration": 60, "backend": "pyav"},
            60,
            id="pyav_dynamic-within_max_duration",
        ),
        pytest.param(
            "opencv_dynamic",
            {"fps": 2, "max_duration": 30, "backend": "pyav"},
            60,
            id="pyav_dynamic-exceeds_max_duration",
        ),
    ],
)
def test_video_loader_frames_sampling(
    dummy_video_path,
    monkeypatch: pytest.MonkeyPatch,
    loader_key: str,
    kwargs: dict,
    expected_num_frames: int,
):
    """Test video loader frames sampling functionality."""
    monkeypatch.setenv("VLLM_VIDEO_LOADER_BACKEND", loader_key)
    loader = VIDEO_LOADER_REGISTRY.load(loader_key)

    with open(dummy_video_path, "rb") as f:
        long_video_bytes = f.read()

    frames, _ = loader.load_bytes(long_video_bytes, **kwargs)

    assert frames.ndim == 4
    assert frames.shape[3] == 3  # RGB
    assert frames.shape[0] == expected_num_frames
