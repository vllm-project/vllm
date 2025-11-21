# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from vllm.assets.base import get_vllm_public_assets
from vllm.assets.video import video_to_ndarrays, video_to_pil_images_list
from vllm.multimodal.image import ImageMediaIO
from vllm.multimodal.video import VIDEO_LOADER_REGISTRY, VideoLoader, VideoMediaIO

from .utils import (
    cosine_similarity,
    create_video_from_image,
    normalize_image,
    random_video,
)

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
    contains broken/unreadable frames to verify the video loader handles
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


def test_video_recovery_functionality(caplog, monkeypatch):
    """Test video frame recovery functionality when sequential reading fails."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv")

        # Create a simple test video
        rng = np.random.RandomState(42)
        test_frames = random_video(rng, 10, 11, 64, 65)  # 10 frames, 64x64

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_path = temp_file.name

            # Create video file
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(temp_path, fourcc, 30, (64, 64))

            for frame in test_frames:
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_frame)

            video_writer.release()

            # Read the video file
            with open(temp_path, "rb") as f:
                video_data = f.read()

        # Clean up
        Path(temp_path).unlink()

        # Mock the _read_frames method to simulate missing frames
        # This will force the recovery logic to trigger
        original_read_frames = None

        def mock_read_frames(
            cap, frame_indices, num_expected, max_frame_idx, warn_on_failure=True
        ):
            # Simulate that only frames 0, 2, 4, 6, 8 are successfully read
            # (frames 1, 3, 5, 7, 9 fail)
            successful_indices = [0, 2, 4, 6, 8]
            successful_frames = []

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            for idx in successful_indices:
                if idx in frame_indices:
                    # Seek to the frame and read it
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        successful_frames.append(rgb_frame)

            if successful_frames:
                frames_array = np.stack(successful_frames)
            else:
                frames_array = np.empty((0, height, width, 3), dtype=np.uint8)

            return frames_array, len(successful_frames), successful_indices

        # Patch the _read_frames method
        from vllm.multimodal.video import OpenCVVideoBackend

        original_read_frames = OpenCVVideoBackend._read_frames
        OpenCVVideoBackend._read_frames = staticmethod(mock_read_frames)

        try:
            # Test with recovery enabled
            loader = VIDEO_LOADER_REGISTRY.load("opencv")

            with caplog.at_level("INFO"):
                frames, metadata = loader.load_bytes(
                    video_data, num_frames=10, recovery_offset=2
                )

            # Verify recovery was attempted and succeeded
            assert "Sequential loading missing" in caplog.text
            assert "Recovery successful" in caplog.text

            # Should have recovered some frames
            assert frames.shape[0] > 0
            assert len(metadata["frames_indices"]) == frames.shape[0]

            # Verify frames are in correct order (temporal order preserved)
            assert metadata["frames_indices"] == sorted(metadata["frames_indices"])

        finally:
            # Restore original method
            OpenCVVideoBackend._read_frames = original_read_frames


def test_video_recovery_disabled(caplog, monkeypatch):
    """Test that recovery is not attempted when recovery_offset is 0."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv")

        # Create a simple test video
        rng = np.random.RandomState(42)
        test_frames = random_video(rng, 6, 7, 64, 65)  # 6 frames, 64x64

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_path = temp_file.name

            # Create video file
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(temp_path, fourcc, 30, (64, 64))

            for frame in test_frames:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_frame)

            video_writer.release()

            with open(temp_path, "rb") as f:
                video_data = f.read()

        Path(temp_path).unlink()

        # Mock _read_frames to return no frames (simulating complete failure)
        def mock_read_frames_no_frames(
            cap, frame_indices, num_expected, max_frame_idx, warn_on_failure=True
        ):
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return np.empty((0, height, width, 3), dtype=np.uint8), 0, []

        from vllm.multimodal.video import OpenCVVideoBackend

        original_read_frames = OpenCVVideoBackend._read_frames
        OpenCVVideoBackend._read_frames = staticmethod(mock_read_frames_no_frames)

        try:
            loader = VIDEO_LOADER_REGISTRY.load("opencv")

            with caplog.at_level("INFO"):
                frames, metadata = loader.load_bytes(
                    video_data, num_frames=6, recovery_offset=0
                )

            # Verify no recovery messages when recovery_offset is 0
            assert "Sequential loading missing" not in caplog.text
            assert "Recovery" not in caplog.text

            # Should return empty frames
            assert frames.shape[0] == 0

        finally:
            OpenCVVideoBackend._read_frames = original_read_frames


def test_video_recovery_dynamic_backend(caplog, monkeypatch):
    """Test recovery functionality in the dynamic backend."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv_dynamic")

        # Create a test video with more frames
        rng = np.random.RandomState(42)
        test_frames = random_video(rng, 20, 21, 64, 65)  # 20 frames

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_path = temp_file.name

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(temp_path, fourcc, 30, (64, 64))

            for frame in test_frames:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_frame)

            video_writer.release()

            with open(temp_path, "rb") as f:
                video_data = f.read()

        Path(temp_path).unlink()

        # Mock to simulate partial frame loading
        def mock_read_frames_partial(
            cap, frame_indices, num_expected, max_frame_idx, warn_on_failure=True
        ):
            # Return only half the expected frames
            successful_count = max(1, num_expected // 2)
            successful_frames = []

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            for i in range(successful_count):
                # Read actual frames from the video
                ret, frame = cap.read()
                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    successful_frames.append(rgb_frame)

            if successful_frames:
                frames_array = np.stack(successful_frames)
            else:
                frames_array = np.empty((0, height, width, 3), dtype=np.uint8)

            successful_indices = list(range(successful_count))
            return frames_array, successful_count, successful_indices

        from vllm.multimodal.video import OpenCVDynamicVideoBackend

        original_read_frames = OpenCVDynamicVideoBackend._read_frames
        OpenCVDynamicVideoBackend._read_frames = staticmethod(mock_read_frames_partial)

        try:
            loader = VIDEO_LOADER_REGISTRY.load("opencv_dynamic")

            with caplog.at_level("INFO"):
                frames, metadata = loader.load_bytes(
                    video_data, fps=2, max_duration=10, recovery_offset=3
                )

            # Should have some frames loaded
            assert frames.shape[0] > 0
            assert "do_sample_frames" in metadata
            assert metadata["do_sample_frames"] is False  # Dynamic backend

        finally:
            OpenCVDynamicVideoBackend._read_frames = original_read_frames


def test_video_recovery_negative_offset_validation(monkeypatch):
    """Test that negative recovery_offset raises ValueError."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv")

        # Create minimal test data
        test_data = b"fake video data"

        # Test OpenCV backend
        loader = VIDEO_LOADER_REGISTRY.load("opencv")
        with pytest.raises(
            ValueError,
            match="recovery_offset must be non-negative, got -1",
        ):
            loader.load_bytes(test_data, recovery_offset=-1)

        # Test OpenCV dynamic backend
        with monkeypatch.context() as m2:
            m2.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv_dynamic")
            loader_dynamic = VIDEO_LOADER_REGISTRY.load("opencv_dynamic")
            with pytest.raises(
                ValueError,
                match="recovery_offset must be non-negative, got -5",
            ):
                loader_dynamic.load_bytes(test_data, recovery_offset=-5)


def test_video_recovery_failure_logging(caplog, monkeypatch):
    """Test that recovery failure is properly logged."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv")

        # Create a test video
        rng = np.random.RandomState(42)
        test_frames = random_video(rng, 5, 6, 64, 65)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_path = temp_file.name

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(temp_path, fourcc, 30, (64, 64))

            for frame in test_frames:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_frame)

            video_writer.release()

            with open(temp_path, "rb") as f:
                video_data = f.read()

        Path(temp_path).unlink()

        # Mock complete failure in both sequential and seeking reads
        def mock_read_frames_failure(
            cap, frame_indices, num_expected, max_frame_idx, warn_on_failure=True
        ):
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return np.empty((0, height, width, 3), dtype=np.uint8), 0, []

        def mock_seek_failure(cap, frame_indices, recovery_offset):
            # Return empty results - no recovery possible
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return np.empty((0, height, width, 3), dtype=np.uint8), [], {}

        from vllm.multimodal.video import OpenCVVideoBackend

        original_read_frames = OpenCVVideoBackend._read_frames
        original_seek_frames = OpenCVVideoBackend._load_frames_with_seeking

        OpenCVVideoBackend._read_frames = staticmethod(mock_read_frames_failure)
        OpenCVVideoBackend._load_frames_with_seeking = staticmethod(mock_seek_failure)

        try:
            loader = VIDEO_LOADER_REGISTRY.load("opencv")

            with caplog.at_level("WARNING"):
                frames, metadata = loader.load_bytes(
                    video_data, num_frames=5, recovery_offset=1
                )

            # Should log recovery attempt and failure
            assert "Sequential loading missing" in caplog.text
            assert "Recovery finished but video is still missing" in caplog.text

            # Should return empty frames
            assert frames.shape[0] == 0

        finally:
            OpenCVVideoBackend._read_frames = original_read_frames
            OpenCVVideoBackend._load_frames_with_seeking = original_seek_frames
