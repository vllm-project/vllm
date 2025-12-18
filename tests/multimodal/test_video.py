# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from vllm.assets.base import get_vllm_public_assets
from vllm.assets.video import video_to_ndarrays, video_to_pil_images_list
from vllm.multimodal.image import ImageMediaIO
from vllm.multimodal.video import VIDEO_LOADER_REGISTRY, VideoLoader, VideoMediaIO

from .utils import cosine_similarity, create_video_from_image, normalize_image

pytestmark = pytest.mark.cpu_test

ASSETS_DIR = Path(__file__).parent / "assets"
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
    def load_bytes(
        cls, data: bytes, num_frames: int = -1, fps: float = -1.0, **kwargs
    ) -> npt.NDArray:
        assert num_frames == 10, "bad num_frames"
        assert fps == 1.0, "bad fps"
        return FAKE_OUTPUT_2


def test_video_media_io_kwargs(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "assert_10_frames_1_fps")
        imageio = ImageMediaIO()

        # Verify that different args pass/fail assertions as expected.
        videoio = VideoMediaIO(imageio, **{"num_frames": 10, "fps": 1.0})
        _ = videoio.load_bytes(b"test")

        videoio = VideoMediaIO(
            imageio, **{"num_frames": 10, "fps": 1.0, "not_used": "not_used"}
        )
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


@pytest.mark.parametrize("is_color", [True, False])
@pytest.mark.parametrize("fourcc, ext", [("mp4v", "mp4"), ("XVID", "avi")])
def test_opencv_video_io_colorspace(is_color: bool, fourcc: str, ext: str):
    """
    Test all functions that use OpenCV for video I/O return RGB format.
    Both RGB and grayscale videos are tested.
    """
    image_path = get_vllm_public_assets(
        filename="stop_sign.jpg", s3_prefix="vision_model_images"
    )
    image = Image.open(image_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        if not is_color:
            image_path = f"{tmpdir}/test_grayscale_image.png"
            image = image.convert("L")
            image.save(image_path)
            # Convert to gray RGB for comparison
            image = image.convert("RGB")
        video_path = f"{tmpdir}/test_RGB_video.{ext}"
        create_video_from_image(
            image_path,
            video_path,
            num_frames=2,
            is_color=is_color,
            fourcc=fourcc,
        )

        frames = video_to_ndarrays(video_path)
        for frame in frames:
            sim = cosine_similarity(
                normalize_image(np.array(frame)), normalize_image(np.array(image))
            )
            assert np.sum(np.isnan(sim)) / sim.size < 0.001
            assert np.nanmean(sim) > 0.99

        pil_frames = video_to_pil_images_list(video_path)
        for frame in pil_frames:
            sim = cosine_similarity(
                normalize_image(np.array(frame)), normalize_image(np.array(image))
            )
            assert np.sum(np.isnan(sim)) / sim.size < 0.001
            assert np.nanmean(sim) > 0.99

        io_frames, _ = VideoMediaIO(ImageMediaIO()).load_file(Path(video_path))
        for frame in io_frames:
            sim = cosine_similarity(
                normalize_image(np.array(frame)), normalize_image(np.array(image))
            )
            assert np.sum(np.isnan(sim)) / sim.size < 0.001
            assert np.nanmean(sim) > 0.99


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
        frames, metadata = loader.load_bytes(video_data, num_frames=-1)

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


@VIDEO_LOADER_REGISTRY.register("test_video_backend_override_1")
class TestVideoBackendOverride1(VideoLoader):
    """Test loader that returns FAKE_OUTPUT_1 to verify backend selection."""

    @classmethod
    def load_bytes(
        cls, data: bytes, num_frames: int = -1, **kwargs
    ) -> tuple[npt.NDArray, dict]:
        return FAKE_OUTPUT_1, {"video_backend": "test_video_backend_override_1"}


@VIDEO_LOADER_REGISTRY.register("test_video_backend_override_2")
class TestVideoBackendOverride2(VideoLoader):
    """Test loader that returns FAKE_OUTPUT_2 to verify backend selection."""

    @classmethod
    def load_bytes(
        cls, data: bytes, num_frames: int = -1, **kwargs
    ) -> tuple[npt.NDArray, dict]:
        return FAKE_OUTPUT_2, {"video_backend": "test_video_backend_override_2"}


def test_video_media_io_backend_kwarg_override(monkeypatch: pytest.MonkeyPatch):
    """
    Test that video_backend kwarg can override the VLLM_VIDEO_LOADER_BACKEND
    environment variable.

    This allows users to dynamically select a different video backend
    via --media-io-kwargs without changing the global env var, which is
    useful when plugins set a default backend but a specific request
    needs a different one.
    """
    with monkeypatch.context() as m:
        # Set the env var to one backend
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "test_video_backend_override_1")

        imageio = ImageMediaIO()

        # Without video_backend kwarg, should use env var backend
        videoio_default = VideoMediaIO(imageio, num_frames=10)
        frames_default, metadata_default = videoio_default.load_bytes(b"test")
        np.testing.assert_array_equal(frames_default, FAKE_OUTPUT_1)
        assert metadata_default["video_backend"] == "test_video_backend_override_1"

        # With video_backend kwarg, should override env var
        videoio_override = VideoMediaIO(
            imageio, num_frames=10, video_backend="test_video_backend_override_2"
        )
        frames_override, metadata_override = videoio_override.load_bytes(b"test")
        np.testing.assert_array_equal(frames_override, FAKE_OUTPUT_2)
        assert metadata_override["video_backend"] == "test_video_backend_override_2"


def test_video_media_io_backend_kwarg_not_passed_to_loader(
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Test that video_backend kwarg is consumed by VideoMediaIO and NOT passed
    through to the underlying video loader's load_bytes method.

    This ensures the kwarg is properly popped from kwargs before forwarding.
    """

    @VIDEO_LOADER_REGISTRY.register("test_reject_video_backend_kwarg")
    class RejectVideoBackendKwargLoader(VideoLoader):
        """Test loader that fails if video_backend is passed through."""

        @classmethod
        def load_bytes(
            cls, data: bytes, num_frames: int = -1, **kwargs
        ) -> tuple[npt.NDArray, dict]:
            # This should never receive video_backend in kwargs
            if "video_backend" in kwargs:
                raise AssertionError(
                    "video_backend should be consumed by VideoMediaIO, "
                    "not passed to loader"
                )
            return FAKE_OUTPUT_1, {"received_kwargs": list(kwargs.keys())}

    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "test_reject_video_backend_kwarg")

        imageio = ImageMediaIO()

        # Even when video_backend is provided, it should NOT be passed to loader
        videoio = VideoMediaIO(
            imageio,
            num_frames=10,
            video_backend="test_reject_video_backend_kwarg",
            other_kwarg="should_pass_through",
        )

        # This should NOT raise AssertionError
        frames, metadata = videoio.load_bytes(b"test")
        np.testing.assert_array_equal(frames, FAKE_OUTPUT_1)
        # Verify other kwargs are still passed through
        assert "other_kwarg" in metadata["received_kwargs"]


def test_video_media_io_backend_env_var_fallback(monkeypatch: pytest.MonkeyPatch):
    """
    Test that when video_backend kwarg is None or not provided,
    VideoMediaIO falls back to VLLM_VIDEO_LOADER_BACKEND env var.
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "test_video_backend_override_2")

        imageio = ImageMediaIO()

        # Explicit None should fall back to env var
        videoio_none = VideoMediaIO(imageio, num_frames=10, video_backend=None)
        frames_none, metadata_none = videoio_none.load_bytes(b"test")
        np.testing.assert_array_equal(frames_none, FAKE_OUTPUT_2)
        assert metadata_none["video_backend"] == "test_video_backend_override_2"

        # Not providing video_backend should also fall back to env var
        videoio_missing = VideoMediaIO(imageio, num_frames=10)
        frames_missing, metadata_missing = videoio_missing.load_bytes(b"test")
        np.testing.assert_array_equal(frames_missing, FAKE_OUTPUT_2)
        assert metadata_missing["video_backend"] == "test_video_backend_override_2"


# ============================================================================
# Frame Recovery Tests
# ============================================================================


def test_video_recovery_simulated_failures(monkeypatch: pytest.MonkeyPatch):
    """
    Test that frame recovery correctly uses the next valid frame when
    target frames fail to load.

    Uses simulate_corruption.mp4 and mocks VideoCapture.grab() to fail
    on specific frame indices, then verifies recovery produces more frames.
    """
    import cv2

    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv")

        # Load the test video
        video_path = ASSETS_DIR / "simulate_corruption.mp4"
        with open(video_path, "rb") as f:
            video_data = f.read()

        # Get video info to determine which frames to fail
        # We'll fail frames at indices that would be sampled
        # For a video sampled at 10 frames, simulate failures on 2 of them
        fail_on_frames = {5, 15, 25}  # Fail on these frame indices

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

        # Test WITHOUT recovery - should have fewer frames due to failures
        frames_no_recovery, meta_no = loader.load_bytes(
            video_data, num_frames=30, frame_recovery=False
        )

        # Test WITH recovery - should recover using next valid frames
        frames_with_recovery, meta_yes = loader.load_bytes(
            video_data, num_frames=30, frame_recovery=True
        )

        # With recovery should have MORE frames than without
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
    Test frame recovery with an actual corrupted video file.

    This test verifies that frame_recovery=True produces at least as many
    frames as frame_recovery=False, and that metadata is consistent.
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv")

        corrupted_video_path = ASSETS_DIR / "corrupted.mp4"

        with open(corrupted_video_path, "rb") as f:
            video_data = f.read()

        loader = VIDEO_LOADER_REGISTRY.load("opencv")

        # Test without recovery
        frames_no_recovery, meta_no_recovery = loader.load_bytes(
            video_data, num_frames=-1, frame_recovery=False
        )

        # Test with recovery
        frames_with_recovery, meta_with_recovery = loader.load_bytes(
            video_data, num_frames=-1, frame_recovery=True
        )

        # With recovery, we should have at least as many frames as without
        assert frames_with_recovery.shape[0] >= frames_no_recovery.shape[0]

        # Both should have consistent metadata
        assert frames_no_recovery.shape[0] == len(meta_no_recovery["frames_indices"])
        num_with_recovery = len(meta_with_recovery["frames_indices"])
        assert frames_with_recovery.shape[0] == num_with_recovery


def test_video_recovery_dynamic_backend(monkeypatch: pytest.MonkeyPatch):
    """Test that frame_recovery works with the dynamic video backend."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv_dynamic")

        corrupted_video_path = ASSETS_DIR / "corrupted.mp4"

        with open(corrupted_video_path, "rb") as f:
            video_data = f.read()

        loader = VIDEO_LOADER_REGISTRY.load("opencv_dynamic")

        # Test with frame_recovery enabled
        frames, metadata = loader.load_bytes(
            video_data, fps=2, max_duration=10, frame_recovery=True
        )

        # Should have some frames loaded
        assert frames.shape[0] > 0
        assert "do_sample_frames" in metadata
        assert metadata["do_sample_frames"] is False  # Dynamic backend always False
        assert frames.shape[0] == len(metadata["frames_indices"])
