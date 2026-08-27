# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import subprocess
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
from vllm.models.minimax_m3.common.mm_preprocess import MiniMaxM3VideoBackend
from vllm.multimodal.video import (
    PYNVVIDEOCODEC_VIDEO_BACKEND,
    VIDEO_LOADER_REGISTRY,
    DynamicVideoBackend,
    Glm5NextVideoBackend,
    GLM46VVideoBackend,
    Molmo2VideoBackend,
    Qwen2VLVideoBackend,
    Qwen3VLVideoBackend,
    VideoBackend,
    VideoLoader,
    VideoSourceMetadata,
    VideoTargetMetadata,
    get_video_loader_backend_for_processor,
)
from vllm.multimodal.video_decoders import decode_video, resolve_video_backend_kwargs
from vllm.multimodal.video_decoders.pynvvideocodec import (
    PYNVVIDEOCODEC_DECODER_CACHE_SIZE,
    PyNvVideoCodecDecoderSlot,
    PyNvVideoCodecVideoBackendMixin,
    _pynv_decoder_pool,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.processor import get_video_processor_cls_name_from_config

from .utils import (
    create_edit_list_trimmed_video,
    create_long_gop_video,
    create_video_from_image,
)

pytestmark = pytest.mark.cpu_test

ASSETS_DIR = Path(__file__).parent / "assets"
assert ASSETS_DIR.exists()

NUM_FRAMES = 10
FAKE_OUTPUT_1 = np.random.rand(NUM_FRAMES, 1280, 720, 3)
FAKE_OUTPUT_2 = np.random.rand(NUM_FRAMES, 1280, 720, 3)


@contextmanager
def _fresh_decoder_pool():
    """Reset module-level decoder pool for isolated test runs."""
    pool = _pynv_decoder_pool
    old_slots = pool.slots
    old_active = pool.active
    old_cond = pool.cond
    old_max = pool.max_slots
    pool.slots = []
    pool.active = 0
    pool.cond = threading.Condition()
    pool.max_slots = None
    try:
        yield pool
    finally:
        pool.slots = old_slots
        pool.active = old_active
        pool.cond = old_cond
        pool.max_slots = old_max


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


def test_video_decoder_backends_are_lazy_imported():
    code = """
import sys
import vllm.multimodal.video  # noqa: F401

backend_modules = {
    f"vllm.multimodal.video_decoders.{backend}"
    for backend in ("opencv", "pyav", "torchcodec", "pynvvideocodec", "deepstream")
}
loaded = sorted(backend_modules & sys.modules.keys())
assert not loaded, loaded
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "backend",
    ["opencv", "pyav", "torchcodec", "pynvvideocodec", "deepstream"],
)
def test_decode_video_imports_only_selected_backend(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
):
    imports = []
    decoded = object()

    def fake_decoder(*args, **kwargs):
        return decoded

    class FakeBackendModule:
        pass

    setattr(FakeBackendModule, f"decode_{backend}", fake_decoder)

    def fake_import_module(name: str, package: str):
        imports.append((name, package))
        return FakeBackendModule

    monkeypatch.setattr(
        "vllm.multimodal.video_decoders.import_module", fake_import_module
    )
    result = decode_video(
        backend,
        loader_cls=None,
        data=b"",
        target=VideoTargetMetadata(-1, -1, 300),
        sampling_kwargs={},
        backend_kwargs={},
        frame_recovery=False,
    )

    assert result is decoded
    assert imports == [(f".{backend}", "vllm.multimodal.video_decoders")]


@pytest.mark.parametrize(
    ("backend", "kwargs", "expected_sampling", "expected_backend"),
    [
        (
            "torchcodec",
            {"min_frames": 4, "num_ffmpeg_threads": 2, "seek_mode": "approximate"},
            {"min_frames": 4},
            {"num_ffmpeg_threads": 2, "seek_mode": "approximate"},
        ),
        (
            "deepstream",
            {"max_frames": 16, "pool_size": 3, "timeout_sec": 10.0},
            {"max_frames": 16},
            {"pool_size": 3, "timeout_sec": 10.0},
        ),
    ],
)
def test_video_backend_kwargs_are_separated_from_sampling_kwargs(
    backend: str,
    kwargs: dict,
    expected_sampling: dict,
    expected_backend: dict,
):
    original_kwargs = dict(kwargs)
    sampling_kwargs, backend_kwargs = resolve_video_backend_kwargs(backend, kwargs)

    assert sampling_kwargs == expected_sampling
    assert backend_kwargs == expected_backend
    assert kwargs == original_kwargs


def test_video_backend_rejects_options_for_another_decoder():
    with pytest.raises(
        ValueError, match="num_ffmpeg_threads is not supported by the 'pyav' backend"
    ):
        resolve_video_backend_kwargs("pyav", {"num_ffmpeg_threads": 2})


@pytest.mark.parametrize(
    ("backend", "error"),
    [
        ("pyav", AssertionError),
        (PYNVVIDEOCODEC_VIDEO_BACKEND, ValueError),
    ],
)
def test_video_decoder_spec_validates_frame_recovery(
    backend: str, error: type[Exception]
):
    with pytest.raises(error, match="frame_recovery is not supported"):
        decode_video(
            backend,
            loader_cls=None,
            data=b"",
            target=VideoTargetMetadata(-1, -1, 300),
            sampling_kwargs={},
            backend_kwargs={},
            frame_recovery=True,
        )


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
        PyNvVideoCodecVideoBackendMixin,
        "_decode_to_pinned_host",
        classmethod(fake_decode),
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
        PyNvVideoCodecVideoBackendMixin,
        "_decode_to_pinned_host",
        classmethod(fake_decode),
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


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_pynvvideocodec_corrupted_videos_raise_value_error():
    valid_video = create_long_gop_video(num_frames=2, width=64, height=64)
    corrupted_video = (ASSETS_DIR / "corrupted.mp4").read_bytes()
    malformed_video = corrupted_video[:128]

    with _fresh_decoder_pool():
        loader = VIDEO_LOADER_REGISTRY.load(PYNVVIDEOCODEC_VIDEO_BACKEND)
        with pytest.raises(
            ValueError,
            match=r"^Invalid or unsupported video file\.$",
        ) as malformed_exc:
            loader.load_bytes(
                malformed_video,
                num_frames=1,
                hw_decoders=1,
            )

        assert malformed_exc.value.__cause__ is not None

        with pytest.raises(
            ValueError,
            match=r"^Invalid or unsupported video file\.$",
        ) as exc_info:
            loader.load_bytes(
                corrupted_video,
                num_frames=-1,
                hw_decoders=1,
            )

        assert exc_info.value.__cause__ is not None

        frames, _ = loader.load_bytes(
            valid_video,
            num_frames=1,
            hw_decoders=1,
        )
        assert frames.shape[0] == 1


@pytest.mark.parametrize("hw_decoders", [1, 3])
def test_pynvvideocodec_decoder_slots_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
    hw_decoders: int,
):
    class FakeSlot:
        pass

    create_count = 0
    with _fresh_decoder_pool():
        PyNvVideoCodecVideoBackendMixin._configure_decoder_slots(hw_decoders)

        def fake_create_slot(cls):
            nonlocal create_count
            create_count += 1
            return FakeSlot()

        monkeypatch.setattr(
            PyNvVideoCodecVideoBackendMixin,
            "_create_decoder_slot",
            classmethod(fake_create_slot),
        )

        borrowed = threading.Event()
        seen_slots = []

        with ExitStack() as stack:
            retained_slots = [
                stack.enter_context(
                    PyNvVideoCodecVideoBackendMixin._borrow_decoder_slot()
                )
                for _ in range(hw_decoders)
            ]

            def borrow_extra_slot():
                with (
                    PyNvVideoCodecVideoBackendMixin._borrow_decoder_slot()
                ) as extra_slot:
                    seen_slots.append(extra_slot)
                    borrowed.set()

            thread = threading.Thread(target=borrow_extra_slot)
            thread.start()
            assert not borrowed.wait(timeout=0.2)

        assert borrowed.wait(timeout=2.0)
        thread.join(timeout=2.0)
        assert not thread.is_alive()

        assert seen_slots[0] in retained_slots
        assert create_count == hw_decoders


def test_pynvvideocodec_decoder_slots_are_configured_once(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(_pynv_decoder_pool, "max_slots", None)

    PyNvVideoCodecVideoBackendMixin._configure_decoder_slots(2)
    PyNvVideoCodecVideoBackendMixin._configure_decoder_slots(2)

    with pytest.raises(RuntimeError, match="already configured as 2, got 3"):
        PyNvVideoCodecVideoBackendMixin._configure_decoder_slots(3)


def test_pynvvideocodec_failed_rebuild_invalidates_decoder_slot():
    events: list[tuple[str, str]] = []

    class FakeStream:
        cuda_stream = "cuda-stream"

    class FakeDecoder:
        poisoned = False

        def reconfigure_decoder(self, file_path: str):
            self.poisoned = True
            events.append(("reconfigure", file_path))
            raise RuntimeError("reconfigure failed")

    old_decoder = FakeDecoder()
    slot = PyNvVideoCodecDecoderSlot(FakeStream())
    slot.decoder = old_decoder
    slot.source_path = "valid.mp4"

    class FakeNvc:
        class OutputColorType:
            RGB = "rgb"

        @staticmethod
        def SimpleDecoder(file_path: str, **kwargs):
            events.append(("construct", file_path))
            assert slot.decoder is None
            assert slot.source_path is None
            raise RuntimeError("construct failed")

    pool = _pynv_decoder_pool
    old_slots = pool.slots
    old_active = pool.active
    old_cond = pool.cond
    old_max = pool.max_slots
    try:
        pool.slots = [slot]
        pool.active = 1
        pool.cond = threading.Condition()
        pool.max_slots = 1

        with (
            pytest.raises(RuntimeError, match="construct failed"),
            PyNvVideoCodecVideoBackendMixin._borrow_decoder_slot() as borrowed,
        ):
            assert borrowed is slot
            borrowed.get_decoder(
                "unsupported-8k.mp4",
                FakeNvc,
                device_index=0,
            )

        assert events == [
            ("reconfigure", "unsupported-8k.mp4"),
            ("construct", "unsupported-8k.mp4"),
        ]
        assert old_decoder.poisoned
        assert slot.decoder is None
        assert slot.source_path is None
        assert pool.slots == [slot]
    finally:
        pool.slots = old_slots
        pool.active = old_active
        pool.cond = old_cond
        pool.max_slots = old_max


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_pynvvideocodec_h200_recovers_after_unsupported_8k():
    import PyNvVideoCodec as nvc
    import torch

    if "H200" not in torch.cuda.get_device_name(0):
        pytest.skip("Requires H200 NVDEC resolution limits")

    valid_video = create_long_gop_video(num_frames=2, width=64, height=64)
    unsupported_video = (ASSETS_DIR / "unsupported_8k_h264.mp4").read_bytes()

    old_slots = _pynv_decoder_pool.slots
    old_active = _pynv_decoder_pool.active
    old_cond = _pynv_decoder_pool.cond
    old_max = _pynv_decoder_pool.max_slots
    try:
        _pynv_decoder_pool.slots = []
        _pynv_decoder_pool.active = 0
        _pynv_decoder_pool.cond = threading.Condition()
        _pynv_decoder_pool.max_slots = None

        loader = VIDEO_LOADER_REGISTRY.load(PYNVVIDEOCODEC_VIDEO_BACKEND)
        frames_before, _ = loader.load_bytes(
            valid_video,
            num_frames=1,
            hw_decoders=1,
        )

        with pytest.raises(Exception) as exc_info:
            loader.load_bytes(
                unsupported_video,
                num_frames=1,
                hw_decoders=1,
            )

        root_cause = exc_info.value
        while root_cause.__cause__ is not None:
            root_cause = root_cause.__cause__
        assert isinstance(root_cause, nvc.PyNvVCExceptionUnsupported)
        assert "MBCount not supported" in str(root_cause)

        frames_after, _ = loader.load_bytes(
            valid_video,
            num_frames=1,
            hw_decoders=1,
        )

        assert frames_after.shape == frames_before.shape
    finally:
        for slot in _pynv_decoder_pool.slots:
            slot.invalidate()
        _pynv_decoder_pool.slots = old_slots
        _pynv_decoder_pool.active = old_active
        _pynv_decoder_pool.cond = old_cond
        _pynv_decoder_pool.max_slots = old_max


def test_pynvvideocodec_cross_subclass_shares_single_pool():
    """Regression test for GHSA-j682-9xp5-rrf3.

    Multiple subclasses of PyNvVideoCodecVideoBackendMixin must share the
    same process-wide decoder slot limit rather than getting independent
    counters via ClassVar shadowing.
    """

    class MixinSubclassA(PyNvVideoCodecVideoBackendMixin):
        pass

    class MixinSubclassB(PyNvVideoCodecVideoBackendMixin):
        pass

    class FakeSlot:
        pass

    create_count = 0

    def fake_create_slot(cls):
        nonlocal create_count
        create_count += 1
        return FakeSlot()

    with _fresh_decoder_pool() as pool:
        pool.max_slots = 2

        orig_create = PyNvVideoCodecVideoBackendMixin._create_decoder_slot
        PyNvVideoCodecVideoBackendMixin._create_decoder_slot = classmethod(
            fake_create_slot
        )
        try:
            with ExitStack() as stack:
                stack.enter_context(MixinSubclassA._borrow_decoder_slot())
                stack.enter_context(MixinSubclassB._borrow_decoder_slot())
                assert pool.active == 2

                blocked = threading.Event()
                acquired = threading.Event()

                def try_borrow():
                    blocked.set()
                    with MixinSubclassB._borrow_decoder_slot():
                        acquired.set()

                t = threading.Thread(target=try_borrow)
                t.start()
                blocked.wait(timeout=2.0)
                assert not acquired.wait(timeout=0.3)

            assert acquired.wait(timeout=2.0)
            t.join(timeout=2.0)
            assert not t.is_alive()

            assert create_count == 2
            assert len(pool.slots) == 2
        finally:
            PyNvVideoCodecVideoBackendMixin._create_decoder_slot = orig_create


@pytest.mark.parametrize("hw_decoders", [0, -1, 1.5, True, "2"])
def test_pynvvideocodec_rejects_invalid_hw_decoders(hw_decoders: object):
    with pytest.raises(ValueError, match="hw_decoders must be a positive integer"):
        VideoBackend.load_bytes(
            b"fake video",
            backend=PYNVVIDEOCODEC_VIDEO_BACKEND,
            hw_decoders=hw_decoders,  # type: ignore[arg-type]
        )


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


def test_cosmos3_edge_uses_qwen3_vl_video_backend():
    backend = get_video_loader_backend_for_processor("Cosmos3EdgeVideoProcessor")

    assert backend == "qwen3_vl"
    assert isinstance(VIDEO_LOADER_REGISTRY.load(backend), Qwen3VLVideoBackend)


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
        pytest.param(
            "MiniMaxAI/MiniMax-M3",
            MiniMaxM3VideoBackend,
            None,
            id="minimax_m3_vl",
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


def test_video_backend_handles_edit_list_trimmed_video(
    monkeypatch: pytest.MonkeyPatch,
):
    """
    An mp4 edit list (e.g. from a lossless ``ffmpeg -ss ... -c copy`` cut)
    hides the decode lead-in: the header still counts every physical sample
    while sequential decode only yields the visible frames. Sampling over the
    header count used to collapse such videos to the few indices below the
    visible count (a single frame for the Qwen loaders) — the loader must
    resample over the true stream length instead.
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv")

        video_data, num_visible = create_edit_list_trimmed_video(
            num_frames=90, trim_start_frame=60
        )

        loader = VIDEO_LOADER_REGISTRY.load("opencv")
        for backend in ["opencv", "pyav"]:
            frames, metadata = loader.load_bytes(
                video_data, num_frames=-1, backend=backend
            )
            assert metadata["total_num_frames"] == num_visible, backend
            assert frames.shape[0] == num_visible, backend
            assert len(metadata["frames_indices"]) == num_visible, backend
            # The green channel encodes the source frame index: the visible
            # frames must be the trailing ones (60..89), not the lead-in.
            mean_green = frames[..., 1].reshape(frames.shape[0], -1).mean(axis=1)
            assert abs(mean_green[0] - 60) <= 5, backend
            assert abs(mean_green[-1] - 89) <= 5, backend

        # The Qwen samplers used to degenerate to a single frame here.
        qwen_frames, qwen_metadata = Qwen2VLVideoBackend.load_bytes(video_data)
        assert qwen_metadata["total_num_frames"] == num_visible
        assert qwen_frames.shape[0] >= 4


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

            def set(self, prop, value):
                # get_video_metadata probes the stream end with a seek;
                # keep the simulated frame counter aligned with rewinds.
                result = self._cap.set(prop, value)
                if prop == cv2.CAP_PROP_POS_FRAMES:
                    self._current_frame = int(value) - 1
                return result

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


# ============================================================================
# TorchCodec Backend Tests
# ============================================================================


def test_torchcodec_backend_loads_frames(
    dummy_video_path, monkeypatch: pytest.MonkeyPatch
):
    """Test that the torchcodec codec backend can load frames."""
    pytest.importorskip("torchcodec")
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv")

        with open(dummy_video_path, "rb") as f:
            video_data = f.read()

        loader = VIDEO_LOADER_REGISTRY.load("opencv")
        frames, metadata = loader.load_bytes(
            video_data, num_frames=8, backend="torchcodec"
        )

        assert frames.ndim == 4
        assert frames.shape[3] == 3  # RGB
        assert frames.shape[0] == 8
        assert frames.shape[0] == len(metadata["frames_indices"])
        assert metadata["video_backend"] == "torchcodec"
        assert "total_num_frames" in metadata
        assert "fps" in metadata
        assert "duration" in metadata


def test_torchcodec_dynamic_backend_loads_frames(
    dummy_video_path, monkeypatch: pytest.MonkeyPatch
):
    """Test that the torchcodec codec with dynamic sampling can load frames."""
    pytest.importorskip("torchcodec")
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv_dynamic")

        with open(dummy_video_path, "rb") as f:
            video_data = f.read()

        loader = VIDEO_LOADER_REGISTRY.load("opencv_dynamic")
        frames, metadata = loader.load_bytes(
            video_data, fps=2, max_duration=10, backend="torchcodec"
        )

        assert frames.ndim == 4
        assert frames.shape[3] == 3  # RGB
        assert frames.shape[0] > 0
        assert frames.shape[0] == len(metadata["frames_indices"])
        assert metadata["video_backend"] == "torchcodec_dynamic"


def test_torchcodec_backend_rejects_frame_recovery(dummy_video_path):
    """frame_recovery is OpenCV-only; torchcodec must reject it."""
    pytest.importorskip("torchcodec")
    with open(dummy_video_path, "rb") as f:
        video_data = f.read()

    loader = VIDEO_LOADER_REGISTRY.load("opencv")
    with pytest.raises(AssertionError):
        loader.load_bytes(
            video_data, num_frames=8, backend="torchcodec", frame_recovery=True
        )


def test_torchcodec_backend_returns_target_frames_not_keyframes():
    """Regression test: torchcodec must return the requested frames, not the
    GOP keyframe they seek back to.

    Mirrors ``test_pyav_backend_returns_target_frames_not_keyframes``: a long
    GOP (single keyframe at frame 0) with a per-frame green-channel marker.
    With ``seek_mode="exact"`` torchcodec resolves each index to the exact
    frame, so the returned markers must be distinct, ordered, and match the
    requested indices.
    """
    pytest.importorskip("torchcodec")
    num_frames = 50
    num_sampled = 4
    height, width = 64, 64

    video_bytes = create_long_gop_video(
        num_frames=num_frames, width=width, height=height
    )

    loader = VIDEO_LOADER_REGISTRY.load("opencv")
    frames, metadata = loader.load_bytes(
        video_bytes, num_frames=num_sampled, backend="torchcodec"
    )
    assert frames.shape == (num_sampled, height, width, 3)

    requested = list(metadata["frames_indices"])
    assert len(requested) == num_sampled

    actual = [int(f[height // 2, width // 2, 1]) for f in frames]

    assert len(set(actual)) == num_sampled, (
        f"torchcodec returned only {len(set(actual))} distinct frames for "
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
        # uniform sampling + torchcodec codec (same frame counts as opencv)
        pytest.param(
            "opencv",
            {"num_frames": 32, "backend": "torchcodec"},
            32,
            id="torchcodec-num_frames",
        ),
        pytest.param(
            "opencv", {"fps": 2, "backend": "torchcodec"}, 120, id="torchcodec-fps"
        ),
        pytest.param(
            "opencv",
            {"num_frames": 500, "fps": 2, "backend": "torchcodec"},
            120,
            id="torchcodec-num_frames_wins_fps",
        ),
        # dynamic sampling + torchcodec codec
        pytest.param(
            "opencv_dynamic",
            {"fps": 1, "max_duration": 60, "backend": "torchcodec"},
            60,
            id="torchcodec_dynamic-within_max_duration",
        ),
        pytest.param(
            "opencv_dynamic",
            {"fps": 2, "max_duration": 30, "backend": "torchcodec"},
            60,
            id="torchcodec_dynamic-exceeds_max_duration",
        ),
        # glm46v dynamic FPS + torchcodec codec
        pytest.param(
            "glm46v",
            {"backend": "torchcodec"},
            120,
            id="glm46v-torchcodec-60s",
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
    if kwargs.get("backend") == "torchcodec":
        pytest.importorskip("torchcodec")
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


def test_glm5next_backend_selected_for_processor():
    """Glm5NextVideoProcessor maps to the glm5next loader so only the
    sampled frames are decoded instead of the whole container. Both the
    borrowed-config spelling and the dedicated Glm5next class name (landing
    with the new checkpoint) must resolve."""
    for name in ("Glm5NextVideoProcessor", "Glm5nextVideoProcessor"):
        assert VIDEO_LOADER_REGISTRY.get_backend_for_video_processor(name) == "glm5next"


@pytest.mark.parametrize(
    ("total_frames", "original_fps", "duration", "fps", "max_frames"),
    [
        (900, 30.0, 30.0, -1, None),  # 30s at flat 2.0 raw fps -> 60 frames
        (3000, 30.0, 100.0, -1, None),
        (72000, 30.0, 2400.0, -1, None),  # 2048 cap
        (48, 2.0, 24.0, -1, None),
        (7, 30.0, 10.0, -1, None),  # short video -> uniform spread + dedup
        (300, 25.0, 0, -1, None),  # duration derived from frame count
        (900, 30.0, 30.0, 4, None),  # request fps override (raw fps)
        (900, 30.0, 30.0, -1, 16),  # request max_frames override
    ],
)
def test_glm5next_backend_indices_match_sampler(
    total_frames, original_fps, duration, fps, max_frames
):
    """The loader must select exactly the frames the processor's sampler
    would, with target.fps mapping onto the raw-fps override."""
    from vllm.transformers_utils.processors.glm5next import (
        glm_sample_frame_indices,
    )

    source = VideoSourceMetadata(
        total_frames_num=total_frames, original_fps=original_fps, duration=duration
    )
    target = VideoTargetMetadata(num_frames=-1, fps=fps, max_duration=-1)

    indices = Glm5NextVideoBackend.compute_frames_index_to_sample(
        source, target, max_frames=max_frames
    )

    assert indices == glm_sample_frame_indices(
        total_frames,
        original_fps,
        duration,
        target_fps=fps if fps > 0 else None,
        max_frame_count=max_frames,
    )
    assert len(indices) % 2 == 0
    assert indices == sorted(indices)  # pair padding may repeat the last frame
    assert all(0 <= idx < total_frames for idx in indices)


def test_glm5next_backend_metadata_contract():
    """create_hf_metadata reports the subset so the processor skips
    re-sampling (do_sample_frames=False) and keeps the original totals."""
    source = VideoSourceMetadata(total_frames_num=900, original_fps=30.0, duration=30.0)
    target = VideoTargetMetadata(num_frames=-1, fps=-1, max_duration=-1)
    indices = Glm5NextVideoBackend.compute_frames_index_to_sample(source, target)

    metadata = Glm5NextVideoBackend.create_hf_metadata(
        source, indices, video_backend="glm5next"
    )
    assert metadata["do_sample_frames"] is False
    assert metadata["frames_indices"] == indices
    assert metadata["total_num_frames"] == 900
    assert metadata["fps"] == 30.0
    assert metadata["duration"] == 30.0

    # A fully-selected source keeps do_sample_frames=True so the processor's
    # sampler takes over on the complete frame set.
    full = list(range(48))
    assert (
        Glm5NextVideoBackend.create_hf_metadata(
            VideoSourceMetadata(total_frames_num=48, original_fps=2.0, duration=24.0),
            full,
            video_backend="glm5next",
        )["do_sample_frames"]
        is True
    )


def _write_gray_video(tmp_path, total_frames, fps, size=(32, 32)):
    """Synthetic clip whose frame i is flat gray level i (near-lossless under
    mp4v), so a decoded frame's level maps back to its source index."""
    cv2 = pytest.importorskip("cv2")

    path = tmp_path / f"gray_{total_frames}_{fps}.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size, isColor=False
    )
    assert writer.isOpened()
    for i in range(total_frames):
        writer.write(np.full((*size, 1), i, dtype=np.uint8))
    writer.release()
    return path


class _CountingCap:
    """Proxy over a real capture that counts grab/seek decoding work."""

    def __init__(self, cap):
        self._cap = cap
        self.grabs = 0
        self.seeks = 0

    def get(self, prop):
        return self._cap.get(prop)

    def grab(self):
        self.grabs += 1
        return self._cap.grab()

    def set(self, prop, value):
        self.seeks += 1
        return self._cap.set(prop, value)

    def read(self):
        return self._cap.read()


@pytest.mark.parametrize("backend", ["opencv", "pyav", "torchcodec"])
def test_glm5next_backend_codec_parity(tmp_path, backend):
    """Every codec samples the same GLM indices and decodes the same
    frames; the seek reader, pyav seek-decode and torchcodec batched
    index-exact decode must agree."""
    if backend == "pyav":
        pytest.importorskip("av")
    elif backend == "torchcodec":
        pytest.importorskip("torchcodec")

    from vllm.transformers_utils.processors.glm5next import (
        glm_sample_frame_indices,
    )

    total_frames, fps = 120, 10
    path = _write_gray_video(tmp_path, total_frames, fps)
    # Dense default sampling (gap 5) and a sparse max_frames cap (gap 20).
    for max_frames in (None, 6):
        kwargs = {} if max_frames is None else {"max_frames": max_frames}
        expected = glm_sample_frame_indices(
            total_frames, float(fps), 12.0, max_frame_count=max_frames
        )

        frames, metadata = Glm5NextVideoBackend.load_bytes(
            path.read_bytes(), backend=backend, **kwargs
        )

        assert metadata["frames_indices"] == expected
        assert metadata["video_backend"].startswith(backend)
        assert len(frames) == len(expected)
        for i, idx in enumerate(expected):
            assert abs(round(float(np.asarray(frames[i]).mean())) - idx) <= 1


def test_glm5next_backend_decodes_only_sampled_frames(tmp_path):
    """End to end over a synthetic clip: load_bytes returns exactly the
    sampler's frame count, with the right frame content at each index."""
    pytest.importorskip("cv2")

    total_frames, fps = 60, 10
    path = _write_gray_video(tmp_path, total_frames, fps)

    from vllm.transformers_utils.processors.glm5next import (
        glm_sample_frame_indices,
    )

    expected = glm_sample_frame_indices(total_frames, float(fps), 6.0)

    frames, metadata = Glm5NextVideoBackend.load_bytes(path.read_bytes())

    assert len(frames) == len(expected)
    assert metadata["frames_indices"] == expected
    assert metadata["do_sample_frames"] is (len(expected) == total_frames)
    # Flat gray frames survive mp4v near-losslessly: each decoded frame's
    # level maps back to its source frame index.
    for frame, idx in zip(frames, expected):
        decoded_idx = round(float(np.asarray(frame).mean()))
        assert abs(decoded_idx - idx) <= 1


def test_glm5next_read_frames_seeks_past_large_gaps(tmp_path):
    """Sparse targets must not walk the container: the stock reader grabs
    every frame up to the last index; the GLM reader seeks instead."""
    cv2 = pytest.importorskip("cv2")

    total_frames, fps = 200, 10
    path = _write_gray_video(tmp_path, total_frames, fps)
    targets = [0, 80, 160, 190]  # gaps of 80/80/30 -> only 30 <= threshold

    stock = cv2.VideoCapture(str(path))
    _, stock_indices = VideoBackend.read_frames(stock, targets, total_frames)
    stock.release()
    assert stock_indices == targets

    cap = _CountingCap(cv2.VideoCapture(str(path)))
    frames, indices = Glm5NextVideoBackend.read_frames(cap, targets, total_frames)
    cap._cap.release()

    assert indices == targets == stock_indices
    for frame, idx in zip(frames, targets):
        assert abs(round(float(np.asarray(frame).mean())) - idx) <= 1
    # Walks only the sub-threshold 30-frame hop; the two 80-frame gaps are
    # seeks. The stock reader grabs all 190 preceding frames.
    assert cap.grabs <= 29
    assert cap.seeks == 3


def test_glm5next_read_frames_dense_walk_matches_stock(tmp_path):
    """Dense targets keep the sequential walk (seeking would be slower) and
    return the same frames as the stock reader."""
    cv2 = pytest.importorskip("cv2")

    total_frames, fps = 120, 10
    path = _write_gray_video(tmp_path, total_frames, fps)
    targets = list(range(0, total_frames, 15))  # gaps of 15 -> all walking

    stock = cv2.VideoCapture(str(path))
    stock_frames, stock_indices = VideoBackend.read_frames(stock, targets, total_frames)
    stock.release()

    cap = _CountingCap(cv2.VideoCapture(str(path)))
    frames, indices = Glm5NextVideoBackend.read_frames(cap, targets, total_frames)
    cap._cap.release()

    assert indices == stock_indices
    for frame, stock_frame, idx in zip(frames, stock_frames, targets):
        assert abs(float(np.asarray(frame).mean()) - float(stock_frame.mean())) <= 2.0
        assert abs(round(float(np.asarray(frame).mean())) - idx) <= 1
    # One initial seek, then pure walking -- no re-seek churn.
    assert cap.seeks == 1
