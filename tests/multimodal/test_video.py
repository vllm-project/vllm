# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import sys
import threading
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from transformers import AutoVideoProcessor
from transformers.video_utils import VideoMetadata

from vllm.assets.base import get_vllm_public_assets
from vllm.multimodal.video import (
    PYNVVIDEOCODEC_DECODER_CACHE_SIZE,
    PYNVVIDEOCODEC_MAX_RETAINED_DECODERS,
    PYNVVIDEOCODEC_VIDEO_BACKEND,
    VIDEO_LOADER_REGISTRY,
    DynamicVideoBackend,
    GLM46VVideoBackend,
    Molmo2VideoBackend,
    PyNvVideoCodecDecoderSlot,
    PyNvVideoCodecVideoBackend,
    Qwen2VLVideoBackend,
    Qwen3VLVideoBackend,
    VideoLoader,
    VideoSourceMetadata,
    VideoTargetMetadata,
    get_video_loader_backend_for_processor,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.processor import get_video_processor_cls_name_from_config

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


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_pynvvideocodec_backend_accounts_raw_decoded_frames(
    monkeypatch: pytest.MonkeyPatch,
):
    decoder_cache_sizes = []

    class FakeMetadata:
        width = 10
        height = 20
        average_fps = 5.0
        duration = 2.0

    class FakeDecoder:
        def __init__(self, *args, **kwargs):
            decoder_cache_sizes.append(kwargs["decoder_cache_size"])

        def __len__(self):
            return 10

        def get_stream_metadata(self):
            return FakeMetadata()

    class FakeNvc:
        class OutputColorType:
            RGB = "rgb"

        SimpleDecoder = FakeDecoder

    class RecordingPool:
        def __init__(self):
            self.acquired: list[int] = []

        @contextmanager
        def acquire(self, size: int):
            self.acquired.append(size)
            yield

    def fake_decode(cls, file_path: str, frame_idx: list[int], nvc):
        return np.zeros((len(frame_idx), 20, 10, 3), dtype=np.uint8)

    pool = RecordingPool()
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", FakeNvc)
    monkeypatch.setattr(
        "vllm.multimodal.gpu_ipc_memory.get_mm_gpu_ipc_pool", lambda: pool
    )
    monkeypatch.setattr(
        PyNvVideoCodecVideoBackend, "_decode_to_pinned_host", classmethod(fake_decode)
    )

    loader = VIDEO_LOADER_REGISTRY.load(PYNVVIDEOCODEC_VIDEO_BACKEND)
    frames, metadata = loader.load_bytes(b"fake video", num_frames=4)

    assert frames.shape == (4, 20, 10, 3)
    assert pool.acquired == [4 * 20 * 10 * 3]
    assert decoder_cache_sizes == [PYNVVIDEOCODEC_DECODER_CACHE_SIZE]
    assert metadata["video_backend"] == PYNVVIDEOCODEC_VIDEO_BACKEND
    assert metadata["frames_indices"] == [0, 3, 6, 9]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_pynvvideocodec_codec_uses_dynamic_sampling_strategy(
    monkeypatch: pytest.MonkeyPatch,
):
    decoded_indices = []

    class FakeMetadata:
        width = 10
        height = 20
        average_fps = 5.0
        duration = 2.0

    class FakeDecoder:
        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 10

        def get_stream_metadata(self):
            return FakeMetadata()

    class FakeNvc:
        class OutputColorType:
            RGB = "rgb"

        SimpleDecoder = FakeDecoder

    class RecordingPool:
        def __init__(self):
            self.acquired: list[int] = []

        @contextmanager
        def acquire(self, size: int):
            self.acquired.append(size)
            yield

    def fake_decode(cls, file_path: str, frame_idx: list[int], nvc):
        decoded_indices.append(frame_idx)
        return np.zeros((len(frame_idx), 20, 10, 3), dtype=np.uint8)

    pool = RecordingPool()
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", FakeNvc)
    monkeypatch.setattr(
        "vllm.multimodal.gpu_ipc_memory.get_mm_gpu_ipc_pool", lambda: pool
    )
    monkeypatch.setattr(
        DynamicVideoBackend, "_decode_to_pinned_host", classmethod(fake_decode)
    )

    loader = VIDEO_LOADER_REGISTRY.load("opencv_dynamic")
    frames, metadata = loader.load_bytes(
        b"fake video",
        fps=2,
        max_duration=1,
        backend=PYNVVIDEOCODEC_VIDEO_BACKEND,
    )

    assert frames.shape == (2, 20, 10, 3)
    assert decoded_indices == [[0, 9]]
    assert pool.acquired == [2 * 20 * 10 * 3]
    assert metadata["video_backend"] == f"{PYNVVIDEOCODEC_VIDEO_BACKEND}_dynamic"
    assert metadata["frames_indices"] == [0, 9]


def test_pynvvideocodec_decoder_slots_are_bounded(monkeypatch: pytest.MonkeyPatch):
    class FakeSlot:
        pass

    create_count = 0
    old_slots = PyNvVideoCodecVideoBackend._decoder_slots
    old_active_slots = PyNvVideoCodecVideoBackend._active_decoder_slots
    old_cond = PyNvVideoCodecVideoBackend._decoder_slot_cond
    try:
        PyNvVideoCodecVideoBackend._decoder_slots = []
        PyNvVideoCodecVideoBackend._active_decoder_slots = 0
        PyNvVideoCodecVideoBackend._decoder_slot_cond = threading.Condition()

        def fake_create_slot(cls):
            nonlocal create_count
            create_count += 1
            return FakeSlot()

        monkeypatch.setattr(
            PyNvVideoCodecVideoBackend,
            "_create_decoder_slot",
            classmethod(fake_create_slot),
        )

        borrowed = threading.Event()
        seen_slots = []

        with ExitStack() as stack:
            retained_slots = [
                stack.enter_context(PyNvVideoCodecVideoBackend._borrow_decoder_slot())
                for _ in range(PYNVVIDEOCODEC_MAX_RETAINED_DECODERS)
            ]

            def borrow_extra_slot():
                with PyNvVideoCodecVideoBackend._borrow_decoder_slot() as extra_slot:
                    seen_slots.append(extra_slot)
                    borrowed.set()

            thread = threading.Thread(target=borrow_extra_slot)
            thread.start()
            assert not borrowed.wait(timeout=0.2)

        assert borrowed.wait(timeout=2.0)
        thread.join(timeout=2.0)
        assert not thread.is_alive()

        assert seen_slots[0] in retained_slots
        assert create_count == PYNVVIDEOCODEC_MAX_RETAINED_DECODERS
    finally:
        PyNvVideoCodecVideoBackend._decoder_slots = old_slots
        PyNvVideoCodecVideoBackend._active_decoder_slots = old_active_slots
        PyNvVideoCodecVideoBackend._decoder_slot_cond = old_cond


def test_pynvvideocodec_decoder_slot_retains_simple_decoder():
    events: list[tuple[object, ...]] = []

    class FakeStream:
        cuda_stream = "cuda-stream"

    class FakeDecoder:
        def __init__(self, file_path: str, **kwargs):
            events.append(
                (
                    "create",
                    file_path,
                    kwargs["gpu_id"],
                    kwargs["cuda_stream"],
                    kwargs["decoder_cache_size"],
                )
            )

        def reconfigure_decoder(self, file_path: str):
            events.append(("reconfigure", file_path))

    class FakeNvc:
        class OutputColorType:
            RGB = "rgb"

        SimpleDecoder = FakeDecoder

    slot = PyNvVideoCodecDecoderSlot(FakeStream())

    decoder = slot.get_decoder("first.mp4", FakeNvc, device_index=7)
    assert slot.get_decoder("first.mp4", FakeNvc, device_index=7) is decoder
    assert slot.get_decoder("second.mp4", FakeNvc, device_index=7) is decoder

    assert events == [
        (
            "create",
            "first.mp4",
            7,
            "cuda-stream",
            PYNVVIDEOCODEC_DECODER_CACHE_SIZE,
        ),
        ("reconfigure", "second.mp4"),
    ]
    assert slot.source_path == "second.mp4"


# ============================================================================
# Video Processor → Video Loader Tests (via model repo)
# ============================================================================


@pytest.mark.parametrize(
    "model_repo, expected_loader_cls, hf_sample_kwargs",
    [
        pytest.param(
            "allenai/Molmo2-4B",
            Molmo2VideoBackend,
            None,
            marks=pytest.mark.skip(
                reason="Video processor not aligned, investigate later.",
            ),
            id="molmo2",
        ),
        pytest.param(
            "zai-org/GLM-4.1V-9B-Thinking",
            DynamicVideoBackend,
            None,
            id="glm4v",
        ),
        pytest.param(
            "zai-org/GLM-4.6V-Flash",
            GLM46VVideoBackend,
            None,
            id="glm46v",
        ),
        pytest.param(
            "Qwen/Qwen3-VL-4B-Instruct",
            Qwen3VLVideoBackend,
            None,
            id="qwen3vl",
        ),
        # Qwen2-VL/Qwen2.5-VL ship no ``video_processor_type`` in their
        # preprocessor config, so resolution relies on the model_type ->
        # video processor fallback in get_video_processor_cls_name_from_config.
        # They also ship no default fps/num_frames, so the HF sampler needs an
        # explicit target rate; pass fps=2 to match the loader default.
        pytest.param(
            "Qwen/Qwen2-VL-7B-Instruct",
            Qwen2VLVideoBackend,
            {"fps": 2},
            id="qwen2vl",
        ),
        pytest.param(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            Qwen2VLVideoBackend,
            {"fps": 2},
            id="qwen2_5_vl",
        ),
    ],
)
def test_video_processor_from_model_repo(
    model_repo: str,
    expected_loader_cls: type,
    hf_sample_kwargs: dict[str, int | float] | None,
):
    """Test that a model repo resolves to the correct video loader backend.

    The test downloads the preprocessor config from HuggingFace Hub,
    extracts the ``video_processor_type`` field, and verifies it maps
    to the expected backend and loader class.  When a corresponding HF
    ``VideoProcessor.sample_frames`` implementation exists, the test
    also verifies that the vLLM backend produces identical frame indices.
    """
    video_processor = get_video_processor_cls_name_from_config(model_repo)
    assert video_processor is not None, (
        f"Model repo {model_repo!r} did not contain a video_processor_type "
        f"in its preprocessor config"
    )

    backend = get_video_loader_backend_for_processor(video_processor)
    loader = VIDEO_LOADER_REGISTRY.load(backend)
    assert isinstance(loader, expected_loader_cls), (
        f"{model_repo!r}: backend={backend!r} loaded "
        f"{type(loader)}, expected {expected_loader_cls}"
    )

    # --- Alignment check with HF VideoProcessor.sample_frames ---
    processor = AutoVideoProcessor.from_pretrained(model_repo, trust_remote_code=True)

    fps_list = [1, 2, 30, 60]
    duration_list = [10, 60, 600]
    for fps, duration_secs in itertools.product(fps_list, duration_list):
        num_frames = fps * duration_secs
        video_bytes = create_long_gop_video(
            num_frames=num_frames,
            fps=fps,
            width=8,
            height=8,
        )

        _, vllm_meta = loader.load_bytes(video_bytes)  # type: ignore[attr-defined]

        hf_metadata = VideoMetadata(
            total_num_frames=vllm_meta["total_num_frames"],
            fps=vllm_meta["fps"],
            duration=vllm_meta["duration"],
        )
        hf_indices = processor.sample_frames(hf_metadata, **(hf_sample_kwargs or {}))
        vllm_indices = np.array(vllm_meta["frames_indices"])
        np.testing.assert_array_equal(
            hf_indices,
            vllm_indices,
            err_msg=(
                f"{model_repo!r} fps={fps} duration={duration_secs}s: "
                f"HF has {len(hf_indices)} indices "
                f"{hf_indices[:5].tolist()}..{hf_indices[-5:].tolist()}, "
                f"vLLM has {len(vllm_indices)} indices "
                f"{vllm_indices[:5].tolist()}..{vllm_indices[-5:].tolist()}"
            ),
        )


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
        # glm46v dynamic FPS (1800 frames @ 30fps = 60s)
        # 60s falls in (30, 300] → target_fps=1.0, extract_t = 60*1.0*2 = 120
        pytest.param(
            "glm46v",
            {"backend": "opencv"},
            120,
            id="glm46v-60s",
        ),
        pytest.param(
            "glm46v",
            {"backend": "pyav"},
            120,
            id="glm46v-pyav-60s",
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


# ============================================================================
# GLM-4.6V Dynamic FPS Threshold Tests
# ============================================================================


@pytest.mark.parametrize(
    "duration, original_fps, total_frames, temporal_patch_size, expected_extract_t",
    [
        # Short video ≤30s → target_fps=3.0
        # extract_t = 10 * 3.0 * 2 = 60
        pytest.param(10, 30, 300, 2, 60, id="short-10s"),
        # Exactly at boundary → target_fps=3.0
        # extract_t = 30 * 3.0 * 2 = 180
        pytest.param(30, 30, 900, 2, 180, id="boundary-30s"),
        # Medium video → target_fps=1.0
        # extract_t = 60 * 1.0 * 2 = 120
        pytest.param(60, 30, 1800, 2, 120, id="medium-60s"),
        # Medium boundary → target_fps=1.0
        # extract_t = 300 * 1.0 * 2 = 600
        pytest.param(300, 30, 9000, 2, 600, id="boundary-300s"),
        # Long video → target_fps=0.5
        # extract_t = 600 * 0.5 * 2 = 600
        pytest.param(600, 30, 18000, 2, 600, id="long-600s"),
        # Very long video, capped by _MAX_FRAME_COUNT_DYNAMIC=640
        # extract_t = min(2400 * 0.5 * 2, 640) = min(2400, 640) = 640
        pytest.param(2400, 30, 72000, 2, 640, id="long-capped-640"),
        # Duration exceeds _MAX_DURATION=2400
        # effective_duration = min(5000, 2400) = 2400, target_fps=0.5
        # extract_t = min(2400 * 0.5 * 2, 640) = 640
        pytest.param(5000, 30, 150000, 2, 640, id="exceeds-max-duration"),
        # temporal_patch_size=4
        # extract_t = 60 * 1.0 * 4 = 240
        pytest.param(60, 30, 1800, 4, 240, id="medium-patch-size-4"),
        # temporal_patch_size=1
        # extract_t = 60 * 1.0 * 1 = 60
        pytest.param(60, 30, 1800, 1, 60, id="medium-patch-size-1"),
    ],
)
def test_glm46v_dynamic_fps_thresholds(
    duration: int,
    original_fps: int,
    total_frames: int,
    temporal_patch_size: int,
    expected_extract_t: int,
):
    """Test GLM-4.6V dynamic FPS threshold selection and frame count."""
    source = VideoSourceMetadata(
        total_frames_num=total_frames,
        original_fps=original_fps,
        duration=duration,
    )
    target = VideoTargetMetadata(num_frames=-1, fps=-1, max_duration=-1)

    indices = GLM46VVideoBackend.compute_frames_index_to_sample(
        source, target, temporal_patch_size=temporal_patch_size
    )

    # Frame count should match expected (may be +1 from even padding)
    assert len(indices) in (expected_extract_t, expected_extract_t + 1), (
        f"Expected ~{expected_extract_t} frames, got {len(indices)}"
    )

    # Frame count must be even
    assert len(indices) % 2 == 0, f"Frame count must be even, got {len(indices)}"

    # All indices must be valid
    assert all(0 <= idx < total_frames for idx in indices), (
        f"Indices out of range [0, {total_frames})"
    )

    # Indices must be sorted and deduplicated
    assert indices == sorted(set(indices)), "Indices must be sorted and deduplicated"


def test_glm46v_even_frame_count_enforcement():
    """Test that GLM-4.6V always returns an even number of frames."""
    target = VideoTargetMetadata(num_frames=-1, fps=-1, max_duration=-1)
    # 5-second video at 30fps → 150 frames
    # extract_t = 5 * 3.0 * 2 = 30 (even, no padding needed)
    source_even = VideoSourceMetadata(total_frames_num=150, original_fps=30, duration=5)
    indices_even = GLM46VVideoBackend.compute_frames_index_to_sample(
        source_even, target
    )
    assert len(indices_even) % 2 == 0

    # 3-second video at 30fps → 90 frames
    # extract_t = 3 * 3.0 * 2 = 18 (even, no padding needed)
    source_even2 = VideoSourceMetadata(total_frames_num=90, original_fps=30, duration=3)
    indices_even2 = GLM46VVideoBackend.compute_frames_index_to_sample(
        source_even2, target
    )
    assert len(indices_even2) % 2 == 0


def test_glm46v_duration_estimation_from_fps():
    """Test GLM-4.6V handles missing duration by estimating from fps."""
    target = VideoTargetMetadata(num_frames=-1, fps=-1, max_duration=-1)
    # duration=0 → estimated from total_frames / fps
    # (89 / 30) + 1 ≈ 4s → target_fps=3.0, extract_t = 4 * 3.0 * 2 = 24
    source_no_duration = VideoSourceMetadata(
        total_frames_num=90, original_fps=30, duration=0
    )
    indices = GLM46VVideoBackend.compute_frames_index_to_sample(
        source_no_duration, target
    )

    assert len(indices) > 0
    assert len(indices) % 2 == 0
    assert all(0 <= idx < 90 for idx in indices)
