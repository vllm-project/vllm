# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import io
import threading
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pybase64
import pytest
from PIL import Image

from vllm import envs
from vllm.assets.base import get_vllm_public_assets
from vllm.assets.video import video_to_ndarrays, video_to_pil_images_list
from vllm.multimodal.media import ImageMediaIO, VideoMediaIO
from vllm.multimodal.video import VIDEO_LOADER_REGISTRY, VideoLoader

from ..utils import cosine_similarity, create_video_from_image, normalize_image

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

ASSETS_DIR = Path(__file__).parent.parent / "assets"
assert ASSETS_DIR.exists()


@pytest.fixture(autouse=True)
def _clear_video_decode_cache():
    import vllm.multimodal.media.video as video_media

    envs_cache_was_enabled = envs._is_envs_cache_enabled()
    envs.disable_envs_cache()
    video_media._VIDEO_DECODE_CACHE.clear()
    yield
    video_media._VIDEO_DECODE_CACHE.clear()
    if envs_cache_was_enabled:
        envs.enable_envs_cache()


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


def test_video_decode_cache_reuses_across_media_io_instances(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    class CountingVideoLoader(VideoLoader):
        calls = 0

        @classmethod
        def load_bytes(
            cls, data: bytes, num_frames: int = -1, **kwargs
        ) -> tuple[npt.NDArray, dict]:
            cls.calls += 1
            return np.array([[[[cls.calls]]]]), {"calls": cls.calls}

    VIDEO_LOADER_REGISTRY.register("test_counting_decode_cache")(CountingVideoLoader)

    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "test_counting_decode_cache")
        m.setenv("VLLM_VIDEO_DECODE_CACHE_SIZE", "2")
        envs.disable_envs_cache()
        imageio = ImageMediaIO()
        video_path = tmp_path / "video.bin"
        video_path.write_bytes(b"video")

        frames_1, metadata_1 = VideoMediaIO(imageio, num_frames=10).load_file(
            video_path
        )
        frames_1[0, 0, 0, 0] = 99
        metadata_1["calls"] = 99
        frames_2, metadata_2 = VideoMediaIO(imageio, num_frames=10).load_file(
            video_path
        )

    assert CountingVideoLoader.calls == 1
    assert frames_2[0, 0, 0, 0] == 1
    assert metadata_2["calls"] == 1


def test_video_decode_cache_key_includes_loader_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    class KwargsVideoLoader(VideoLoader):
        calls = 0

        @classmethod
        def load_bytes(
            cls, data: bytes, num_frames: int = -1, **kwargs
        ) -> tuple[npt.NDArray, dict]:
            cls.calls += 1
            return np.array([[[[cls.calls]]]]), {"calls": cls.calls}

    VIDEO_LOADER_REGISTRY.register("test_kwargs_decode_cache")(KwargsVideoLoader)

    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "test_kwargs_decode_cache")
        m.setenv("VLLM_VIDEO_DECODE_CACHE_SIZE", "2")
        envs.disable_envs_cache()
        imageio = ImageMediaIO()
        video_path = tmp_path / "video.bin"
        video_path.write_bytes(b"video")

        VideoMediaIO(imageio, num_frames=10, fps=1.0).load_file(video_path)
        VideoMediaIO(imageio, num_frames=10, fps=2.0).load_file(video_path)

    assert KwargsVideoLoader.calls == 2


def test_video_decode_cache_waits_for_inflight_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    decode_started = threading.Event()
    finish_decode = threading.Event()
    calls_lock = threading.Lock()

    class SlowVideoLoader(VideoLoader):
        calls = 0

        @classmethod
        def load_bytes(
            cls, data: bytes, num_frames: int = -1, **kwargs
        ) -> tuple[npt.NDArray, dict]:
            with calls_lock:
                cls.calls += 1
                calls = cls.calls
            decode_started.set()
            assert finish_decode.wait(timeout=5)
            return np.array([[[[calls]]]]), {"calls": calls}

    VIDEO_LOADER_REGISTRY.register("test_inflight_decode_cache")(SlowVideoLoader)

    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "test_inflight_decode_cache")
        m.setenv("VLLM_VIDEO_DECODE_CACHE_SIZE", "2")
        envs.disable_envs_cache()
        imageio = ImageMediaIO()
        video_path = tmp_path / "video.bin"
        video_path.write_bytes(b"video")
        results: list[tuple[npt.NDArray, dict]] = []
        errors: list[BaseException] = []

        def load_video() -> None:
            try:
                videoio = VideoMediaIO(imageio, num_frames=10)
                results.append(videoio.load_file(video_path))
            except BaseException as exc:
                errors.append(exc)

        owner_thread = threading.Thread(target=load_video)
        owner_thread.start()
        assert decode_started.wait(timeout=5)

        waiter_thread = threading.Thread(target=load_video)
        waiter_thread.start()
        finish_decode.set()

        owner_thread.join(timeout=5)
        waiter_thread.join(timeout=5)

    assert not owner_thread.is_alive()
    assert not waiter_thread.is_alive()
    assert not errors
    assert SlowVideoLoader.calls == 1
    assert len(results) == 2
    assert [metadata["calls"] for _, metadata in results] == [1, 1]


@pytest.mark.parametrize("is_color", [True, False])
@pytest.mark.parametrize("fourcc, ext", [("mp4v", "mp4"), ("XVID", "avi")])
def test_opencv_video_io_colorspace(tmp_path, is_color: bool, fourcc: str, ext: str):
    """
    Test all functions that use OpenCV for video I/O return RGB format.
    Both RGB and grayscale videos are tested.
    """
    image_path = get_vllm_public_assets(
        filename="stop_sign.jpg", s3_prefix="vision_model_images"
    )
    image = Image.open(image_path)

    if not is_color:
        image_path = f"{tmp_path}/test_grayscale_image.png"
        image = image.convert("L")
        image.save(image_path)
        # Convert to gray RGB for comparison
        image = image.convert("RGB")
    video_path = f"{tmp_path}/test_RGB_video.{ext}"
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


NUM_FRAMES = 10
FAKE_OUTPUT_1 = np.random.rand(NUM_FRAMES, 1280, 720, 3)
FAKE_OUTPUT_2 = np.random.rand(NUM_FRAMES, 1280, 720, 3)


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


def _make_jpeg_b64_frames(n: int, width: int = 8, height: int = 8) -> list[str]:
    """Return *n* tiny base64-encoded JPEG frames."""
    frames: list[str] = []
    for i in range(n):
        img = Image.new("RGB", (width, height), color=(i % 256, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        frames.append(pybase64.b64encode(buf.getvalue()).decode("ascii"))
    return frames


def test_load_base64_jpeg_returns_metadata():
    """Regression test: load_base64 with video/jpeg must return metadata.

    Previously, base64 JPEG frame sequences returned an empty dict for
    metadata, which broke downstream consumers that rely on fields like
    total_num_frames and fps. See PR #37301.
    """

    num_test_frames = 3

    b64_frames = _make_jpeg_b64_frames(num_test_frames)
    data = ",".join(b64_frames)

    imageio = ImageMediaIO()
    videoio = VideoMediaIO(imageio, num_frames=num_test_frames)
    frames, metadata = videoio.load_base64("video/jpeg", data)

    # Frames array shape: (num_frames, H, W, 3)
    assert frames.shape[0] == num_test_frames

    # All required metadata keys must be present
    required_keys = {
        "total_num_frames",
        "fps",
        "duration",
        "video_backend",
        "frames_indices",
        "do_sample_frames",
    }
    assert required_keys.issubset(metadata.keys()), (
        f"Missing metadata keys: {required_keys - metadata.keys()}"
    )

    assert metadata["total_num_frames"] == num_test_frames
    assert metadata["video_backend"] == "jpeg_sequence"
    assert metadata["frames_indices"] == list(range(num_test_frames))
    assert metadata["do_sample_frames"] is False
    # Default fps=1 → duration == num_frames
    assert metadata["fps"] == 1.0
    assert metadata["duration"] == float(num_test_frames)


def test_load_base64_jpeg_enforces_num_frames_limit():
    """Frames beyond num_frames must be truncated in the video/jpeg path.

    Without the limit an attacker can send thousands of base64 JPEG frames
    in a single request and exhaust server memory (OOM).
    """
    num_frames_limit = 4
    sent_frames = 20

    b64_frames = _make_jpeg_b64_frames(sent_frames)
    data = ",".join(b64_frames)

    imageio = ImageMediaIO()
    videoio = VideoMediaIO(imageio, num_frames=num_frames_limit)
    frames, metadata = videoio.load_base64("video/jpeg", data)

    assert frames.shape[0] == num_frames_limit
    assert metadata["total_num_frames"] == num_frames_limit
    assert metadata["frames_indices"] == list(range(num_frames_limit))


def test_load_base64_jpeg_no_limit_when_num_frames_negative():
    """When num_frames is -1, all frames should be loaded without truncation."""
    sent_frames = 10

    b64_frames = _make_jpeg_b64_frames(sent_frames)
    data = ",".join(b64_frames)

    imageio = ImageMediaIO()
    videoio = VideoMediaIO(imageio, num_frames=-1)
    frames, metadata = videoio.load_base64("video/jpeg", data)

    assert frames.shape[0] == sent_frames
    assert metadata["total_num_frames"] == sent_frames
    assert metadata["frames_indices"] == list(range(sent_frames))


def test_load_base64_jpeg_raises_on_zero_num_frames():
    """num_frames=0 is invalid and should raise ValueError."""
    b64_frames = _make_jpeg_b64_frames(3)
    data = ",".join(b64_frames)

    imageio = ImageMediaIO()
    videoio = VideoMediaIO(imageio, num_frames=0)

    with pytest.raises(ValueError, match="num_frames must be greater than 0 or -1"):
        videoio.load_base64("video/jpeg", data)
