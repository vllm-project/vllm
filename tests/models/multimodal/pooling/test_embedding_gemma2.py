# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import inspect
import os
import textwrap

import pytest
import torch

from tests.models.registry import HF_EXAMPLE_MODELS

MODEL = os.environ.get("EG2_MODEL_PATH")
needs_model = pytest.mark.skipif(not MODEL, reason="EG2_MODEL_PATH not set")
TEXT_COS, MM_COS = 0.999, 0.999


@pytest.fixture
def needs_hf_embedding_gemma2():
    """Skip when the installed transformers has no `embedding_gemma2` module."""
    model_info = HF_EXAMPLE_MODELS.get_hf_info("EmbeddingGemma2Model")
    model_info.check_transformers_version(on_fail="skip")


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        torch.tensor(a, dtype=torch.float32),
        torch.tensor(b, dtype=torch.float32),
        dim=-1,
    )


def _ensure_synthetic_video(
    path: str, duration: float, fps: float, width: int, height: int
) -> str:
    if os.path.exists(path):
        os.remove(path)
    import cv2
    import numpy as np

    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", cv2.VideoWriter.fourcc)
    fourcc = fourcc_fn(*"mp4v")
    num_frames = int(duration * fps)
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    for i in range(num_frames):
        # Spatially varying color pattern that evolves dynamically over frames
        frame: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = ((x_coords * 255 // max(1, width)) + i * 17) % 256
        frame[..., 1] = ((y_coords * 255 // max(1, height)) + i * 29) % 256
        frame[..., 2] = (x_coords + y_coords + i * 13) % 256
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="module")
def st_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        MODEL, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16}
    )


@needs_model
def test_structure(vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    with vllm_runner(MODEL, runner="pooling", max_model_len=8192) as vm:

        def probe(model):
            lm = model.language_model
            win = [layer.self_attn.attn.impl.sliding_window for layer in lm.layers]
            return win, model.config.text_config.sliding_window

        win, cfg_w = vm.apply_model(probe)[0]
        assert (cfg_w, cfg_w) in win  # (512, 512) == W-1 with W = 513


@needs_model
@pytest.mark.parametrize(
    "prompt",
    [
        "short query",
        "task: search result | query: what is vllm",
        "word " * 700,  # > 512: crosses sliding window
        "word " * 1500,  # > 1024
    ],
)
def test_text_parity(vllm_runner, st_model, prompt):
    ref = st_model.encode([prompt], normalize_embeddings=True)
    with vllm_runner(MODEL, runner="pooling", max_model_len=8192) as vm:
        out = vm.embed([prompt])
    sim = _cos(out, ref).min().item()
    assert sim >= TEXT_COS, (
        f"Text cosine {sim} < {TEXT_COS} for prompt len {len(prompt)}"
    )


@needs_model
def test_mm_recipe(vllm_runner, st_model):
    from PIL import Image

    img = Image.new("RGB", (224, 224), color=(255, 0, 0))
    prompt = "Describe this image: <|image|>"

    # SentenceTransformer multimodal encode
    ref = st_model.encode([{"image": img, "text": prompt}], normalize_embeddings=True)
    with vllm_runner(MODEL, runner="pooling", max_model_len=8192) as vm:
        out = vm.embed([{"prompt": prompt, "multi_modal_data": {"image": img}}])
    sim = _cos(out, ref).min().item()
    assert sim >= MM_COS, f"Image cosine {sim} < {MM_COS}"


@needs_model
def test_natural_image_parity(vllm_runner, st_model):
    from PIL import Image

    img_path = "/projects/gemma4-vllm/images/cat.jpg"
    if not os.path.exists(img_path):
        pytest.skip(f"Test asset {img_path} not found")
    img = Image.open(img_path)
    prompt = "What is this image? <|image|>"
    ref = st_model.encode([{"image": img, "text": prompt}], normalize_embeddings=True)
    with vllm_runner(MODEL, runner="pooling", max_model_len=8192) as vm:
        out = vm.embed([{"prompt": prompt, "multi_modal_data": {"image": img}}])
    sim = _cos(out, ref).min().item()
    assert sim >= MM_COS, f"Natural image cosine {sim} < {MM_COS}"


@needs_model
def test_audio_parity(vllm_runner, st_model):
    import torchaudio

    audio_path = "/projects/gemma4-vllm/audio/tellMeAboutTheSun.wav"
    if not os.path.exists(audio_path):
        pytest.skip(f"Test asset {audio_path} not found")
    wav, sr = torchaudio.load(audio_path)
    mono_wav = wav.mean(dim=0).numpy()
    prompt = "Describe this audio: <|audio|>"
    ref = st_model.encode(
        [{"audio": audio_path, "text": prompt}], normalize_embeddings=True
    )
    with vllm_runner(MODEL, runner="pooling", max_model_len=8192) as vm:
        out = vm.embed(
            [{"prompt": prompt, "multi_modal_data": {"audio": (mono_wav, sr)}}]
        )
    sim = _cos(out, ref).min().item()
    assert sim >= MM_COS, f"Audio cosine {sim} < {MM_COS}"


@needs_model
def test_mixed_parity(vllm_runner, st_model):
    import torchaudio
    from PIL import Image

    img_path = "/projects/gemma4-vllm/images/cat.jpg"
    audio_path = "/projects/gemma4-vllm/audio/tellMeAboutTheSun.wav"
    if not os.path.exists(img_path) or not os.path.exists(audio_path):
        pytest.skip("Test assets for mixed parity not found")
    img = Image.open(img_path)
    wav, sr = torchaudio.load(audio_path)
    mono_wav = wav.mean(dim=0).numpy()
    prompt = "Describe image <|image|> and audio <|audio|>"
    ref = st_model.encode(
        [{"image": img, "audio": audio_path, "text": prompt}],
        normalize_embeddings=True,
    )
    with vllm_runner(MODEL, runner="pooling", max_model_len=8192) as vm:
        out = vm.embed(
            [
                {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": img,
                        "audio": (mono_wav, sr),
                    },
                }
            ]
        )
    sim = _cos(out, ref).min().item()
    assert sim >= MM_COS, f"Mixed cosine {sim} < {MM_COS}"


@needs_model
def test_video_parity_uniform_overflow(vllm_runner, st_model):
    from transformers.video_utils import load_video

    video_path = _ensure_synthetic_video(
        "/tmp/test_40s.mp4", duration=40.0, fps=10.0, width=160, height=120
    )
    frames, meta = load_video(video_path, backend="opencv")
    prompt = "Describe this video: <|video|>"
    ref = st_model.encode(
        [{"video": video_path, "text": prompt}], normalize_embeddings=True
    )
    with vllm_runner(MODEL, runner="pooling", max_model_len=8192) as vm:
        out = vm.embed(
            [{"prompt": prompt, "multi_modal_data": {"video": (frames, meta)}}]
        )
    sim = _cos(out, ref).min().item()
    assert sim >= MM_COS, f"40s video cosine {sim} < {MM_COS}"


@needs_model
def test_video_parity_portrait(vllm_runner, st_model):
    from transformers.video_utils import load_video

    video_path = _ensure_synthetic_video(
        "/tmp/test_portrait.mp4", duration=3.0, fps=10.0, width=120, height=160
    )
    frames, meta = load_video(video_path, backend="opencv")
    prompt = "Describe this video: <|video|>"
    ref = st_model.encode(
        [{"video": video_path, "text": prompt}], normalize_embeddings=True
    )
    with vllm_runner(MODEL, runner="pooling", max_model_len=8192) as vm:
        out = vm.embed(
            [{"prompt": prompt, "multi_modal_data": {"video": (frames, meta)}}]
        )
    sim = _cos(out, ref).min().item()
    assert sim >= MM_COS, f"Portrait video cosine {sim} < {MM_COS}"


@needs_model
def test_tower_skip(vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    def nparams(m):
        return sum(p.numel() for p in m.parameters()) / 1e6

    with vllm_runner(
        MODEL,
        runner="pooling",
        max_model_len=2048,
        limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
    ) as vm:
        p_text = vm.apply_model(nparams)[0]
        assert abs(p_text - 271.00) <= 0.05, (
            f"Expected 271.00M text params, got {p_text}M"
        )

    with vllm_runner(MODEL, runner="pooling", max_model_len=2048) as vm:
        p_full = vm.apply_model(nparams)[0]
        assert abs(p_full - 744.37) <= 0.05, (
            f"Expected 744.37M full params, got {p_full}M"
        )


def test_encoder_contract(needs_hf_embedding_gemma2):
    from vllm.model_executor.models.embedding_gemma2 import EmbeddingGemma2Model
    from vllm.model_executor.models.gemma4_mm import (
        Gemma4ForConditionalGeneration as G,
    )

    borrowed_methods = [
        "_parse_and_validate_multimodal_inputs",
        "_parse_and_validate_image_input",
        "_parse_and_validate_video_input",
        "_parse_and_validate_audio_input",
        "_process_image_input",
        "_process_video_input",
        "_process_audio_input",
        "embed_multimodal",
        "_encoder_chunk",
    ]

    for name in borrowed_methods:
        # Check method identity (L-F6 + T-F4)
        assert getattr(EmbeddingGemma2Model, name) is getattr(G, name)

        # AST checks: verify no zero-arg super() calls in borrowed methods
        func = getattr(EmbeddingGemma2Model, name)
        src = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "super"
            ):
                assert len(node.args) > 0, (
                    f"Zero-arg super() call found in borrowed method {name}"
                )

    # AST check: all self.<attr> non-callable accesses are assigned in __init__
    init_src = inspect.getsource(EmbeddingGemma2Model.__init__)
    for name in borrowed_methods:
        func = getattr(EmbeddingGemma2Model, name)
        src = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
            ):
                attr = node.attr
                assert hasattr(EmbeddingGemma2Model, attr) or attr in init_src, (
                    f"Attribute {attr} accessed on self in {name} is neither "
                    f"defined on class nor in __init__"
                )


@needs_model
def test_instance_attribute_access(vllm_runner):
    """Check self.<attr> accesses on a real model instance."""
    from vllm.model_executor.models.embedding_gemma2 import EmbeddingGemma2Model

    borrowed_methods = [
        "_parse_and_validate_multimodal_inputs",
        "_parse_and_validate_image_input",
        "_parse_and_validate_video_input",
        "_parse_and_validate_audio_input",
        "_process_image_input",
        "_process_video_input",
        "_process_audio_input",
        "embed_multimodal",
        "_encoder_chunk",
    ]
    attrs_accessed = set()
    for name in borrowed_methods:
        func = getattr(EmbeddingGemma2Model, name)
        src = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
            ):
                attrs_accessed.add(node.attr)

    def check_attrs(model):
        for attr in attrs_accessed:
            assert hasattr(model, attr), (
                f"Real instance of EmbeddingGemma2Model is missing attribute '{attr}' "
                f"accessed by borrowed method"
            )
        return True

    with vllm_runner(MODEL, runner="pooling", max_model_len=2048) as vm:
        res = vm.apply_model(check_attrs)
        assert res[0] is True


@needs_model
def test_video_loader_real_video(vllm_runner, st_model):
    """Decode a real MP4 file through EmbeddingGemma2VideoBackend.load_bytes

    and VideoMediaIO / MediaConnector with default num_frames=32.
    """
    from vllm.multimodal.media import MediaConnector
    from vllm.multimodal.media.video import VideoMediaIO
    from vllm.multimodal.video import VIDEO_LOADER_REGISTRY

    video_path = "/projects/gemma4-vllm/videos/test_intel.mp4"
    backend_cls = VIDEO_LOADER_REGISTRY.name2class["embedding_gemma2"]
    with open(video_path, "rb") as f:
        data = f.read()

    # Direct backend load
    frames, meta = backend_cls.load_bytes(data)

    # VideoMediaIO with default num_frames=32 (VideoMediaIO passes num_frames=32)
    video_io = VideoMediaIO(image_io=None, video_backend="embedding_gemma2")
    loaded = video_io.load_bytes(data)
    frames_io, meta_io = loaded.media
    assert len(frames_io) == len(frames)
    assert meta_io.get("do_sample_frames") is False

    # MediaConnector with video_processor="EmbeddingGemma2VideoProcessor"
    connector = MediaConnector(allowed_local_media_path="/projects/gemma4-vllm/videos")
    fetched = connector.fetch_video(
        f"file://{video_path}",
        video_processor="EmbeddingGemma2VideoProcessor",
    )
    frames_conn, meta_conn = fetched.media
    assert len(frames_conn) == 32
    assert meta_conn.get("do_sample_frames") is False

    prompt = "Describe this video: <|video|>"
    ref = st_model.encode(
        [{"video": video_path, "text": prompt}], normalize_embeddings=True
    )
    with vllm_runner(MODEL, runner="pooling", max_model_len=8192) as vm:
        out = vm.embed(
            [
                {
                    "prompt": prompt,
                    "multi_modal_data": {"video": (frames_conn, meta_conn)},
                }
            ]
        )
    sim = _cos(out, ref).min().item()
    assert sim >= MM_COS, f"Real video loader cosine {sim} < {MM_COS}"


@needs_model
def test_video_loader_duplicate_expansion_e2e(vllm_runner, st_model):
    """End-to-end duplicate expansion test with low-fps clip."""
    import numpy as np

    from vllm.multimodal.video import VIDEO_LOADER_REGISTRY

    backend_cls = VIDEO_LOADER_REGISTRY.name2class["embedding_gemma2"]
    # 4s clip at 0.5 fps = 2 unique frames
    lowfps_path = "/tmp/test_lowfps_05.mp4"
    _ensure_synthetic_video(lowfps_path, duration=4.0, fps=0.5, width=64, height=48)
    with open(lowfps_path, "rb") as f:
        data = f.read()
    # Default fps=1.0 -> 4 frames expanding 2 unique frames to [0, 0, 1, 1]
    frames, meta = backend_cls.load_bytes(data)
    assert frames.shape[0] == 4
    # Compare returned frames directly, not only indices
    assert np.array_equal(frames[0], frames[1])
    assert not np.array_equal(frames[0], frames[2])
    assert np.array_equal(frames[2], frames[3])

    prompt = "Describe this video: <|video|>"
    with vllm_runner(MODEL, runner="pooling", max_model_len=8192) as vm:
        out = vm.embed(
            [{"prompt": prompt, "multi_modal_data": {"video": (frames, meta)}}]
        )
    ref = st_model.encode(
        [{"video": lowfps_path, "text": prompt}], normalize_embeddings=True
    )
    sim = _cos(out, ref).min().item()
    assert sim >= MM_COS, f"Duplicate expansion cosine {sim} < {MM_COS}"
    assert len(out) == 1
    assert len(out[0]) == 768


def test_video_loader_indices(needs_hf_embedding_gemma2):
    from transformers.models.embedding_gemma2.video_processing_embedding_gemma2 import (
        EmbeddingGemma2VideoProcessor,
    )
    from transformers.video_utils import VideoMetadata

    from vllm.multimodal.video import (
        VIDEO_LOADER_REGISTRY,
        VideoSourceMetadata,
        VideoTargetMetadata,
    )

    backend_cls = VIDEO_LOADER_REGISTRY.name2class["embedding_gemma2"]
    hf_proc = EmbeddingGemma2VideoProcessor()

    # Case 1: clip shorter than 32s (e.g. 5s @ 24fps = 120 frames)
    src_5s = VideoSourceMetadata(total_frames_num=120, original_fps=24.0, duration=5.0)
    tgt = VideoTargetMetadata(num_frames=-1, fps=1.0, max_duration=300)
    vllm_indices = backend_cls.compute_frames_index_to_sample(src_5s, tgt)
    hf_meta_5s = VideoMetadata(fps=24.0, duration=5.0, total_num_frames=120)
    hf_indices_5s = hf_proc.sample_frames(
        hf_meta_5s, fps=1.0, max_frames=32, overflow_strategy="uniform"
    ).tolist()
    assert vllm_indices == hf_indices_5s

    # Case 2: clip shorter than 1s (e.g. 0.5s @ 24fps = 12 frames)
    src_sub1s = VideoSourceMetadata(
        total_frames_num=12, original_fps=24.0, duration=0.5
    )
    vllm_sub1s = backend_cls.compute_frames_index_to_sample(src_sub1s, tgt)
    hf_meta_sub1s = VideoMetadata(fps=24.0, duration=0.5, total_num_frames=12)
    hf_indices_sub1s = hf_proc.sample_frames(
        hf_meta_sub1s, fps=1.0, max_frames=32, overflow_strategy="uniform"
    ).tolist()
    assert vllm_sub1s == hf_indices_sub1s

    # Case 3: clip longer than 32s (e.g. 40s @ 10fps = 400 frames)
    # -> uniform overflow cap 32 frames
    src_40s = VideoSourceMetadata(
        total_frames_num=400, original_fps=10.0, duration=40.0
    )
    vllm_40s = backend_cls.compute_frames_index_to_sample(src_40s, tgt)
    hf_meta_40s = VideoMetadata(fps=10.0, duration=40.0, total_num_frames=400)
    hf_indices_40s = hf_proc.sample_frames(
        hf_meta_40s, fps=1.0, max_frames=32, overflow_strategy="uniform"
    ).tolist()
    assert vllm_40s == hf_indices_40s
    assert len(vllm_40s) == 32

    # Case 4: default target fps (target.fps <= 0 defaults to 1.0)
    tgt_default_fps = VideoTargetMetadata(num_frames=-1, fps=-1.0, max_duration=300)
    vllm_def = backend_cls.compute_frames_index_to_sample(src_5s, tgt_default_fps)
    assert vllm_def == hf_indices_5s

    # Case 5 (NEW-2): 60 fps / 300 s (18000 frames) vs HF sample_frames
    src_300s = VideoSourceMetadata(
        total_frames_num=18000, original_fps=60.0, duration=300.0
    )
    v_300s = backend_cls.compute_frames_index_to_sample(
        src_300s, VideoTargetMetadata(num_frames=-1, fps=60.0, max_duration=300)
    )
    h_300s = hf_proc.sample_frames(
        VideoMetadata(fps=60.0, duration=300.0, total_num_frames=18000),
        fps=60.0,
        max_frames=32,
        overflow_strategy="uniform",
    ).tolist()
    assert v_300s == h_300s, f"60fps/300s mismatch: {v_300s[-3:]} vs {h_300s[-3:]}"

    # Case 6 (NEW-2): 1 fps / 4 h (432000 frames @ 30 native fps) vs HF sample_frames
    src_4h = VideoSourceMetadata(
        total_frames_num=432000, original_fps=30.0, duration=14400.0
    )
    v_4h = backend_cls.compute_frames_index_to_sample(
        src_4h, VideoTargetMetadata(num_frames=-1, fps=1.0, max_duration=300)
    )
    h_4h = hf_proc.sample_frames(
        VideoMetadata(fps=30.0, duration=14400.0, total_num_frames=432000),
        fps=1.0,
        max_frames=32,
        overflow_strategy="uniform",
    ).tolist()
    assert v_4h == h_4h, f"1fps/4h mismatch: {v_4h[-2:]} vs {h_4h[-2:]}"


def test_video_loader_duplicate_expansion(needs_hf_embedding_gemma2):
    from transformers.models.embedding_gemma2.video_processing_embedding_gemma2 import (
        EmbeddingGemma2VideoProcessor,
    )
    from transformers.video_utils import VideoMetadata

    from vllm.multimodal.video import (
        VIDEO_LOADER_REGISTRY,
        VideoSourceMetadata,
        VideoTargetMetadata,
    )

    backend_cls = VIDEO_LOADER_REGISTRY.name2class["embedding_gemma2"]
    hf_proc = EmbeddingGemma2VideoProcessor()

    # 4s clip @ 0.5fps = 2 frames total. Requested target fps = 1.0 -> 4 frames.
    # Source has fewer frames than requested sampled frames, requiring duplicates.
    src_lowfps = VideoSourceMetadata(total_frames_num=2, original_fps=0.5, duration=4.0)
    tgt = VideoTargetMetadata(num_frames=-1, fps=1.0, max_duration=300)
    vllm_indices = backend_cls.compute_frames_index_to_sample(src_lowfps, tgt)
    hf_meta = VideoMetadata(fps=0.5, duration=4.0, total_num_frames=2)
    hf_indices = hf_proc.sample_frames(
        hf_meta, fps=1.0, max_frames=32, overflow_strategy="uniform"
    ).tolist()
    assert vllm_indices == hf_indices
    assert len(vllm_indices) == 4
    assert vllm_indices == [0, 0, 1, 1]


def test_video_media_io_embedding_gemma2():
    """Verify VideoMediaIO with default num_frames=32 loads video bytes for

    EmbeddingGemma2 without error.
    """
    import os
    from unittest.mock import MagicMock

    from vllm.multimodal.media import MediaConnector
    from vllm.multimodal.media.video import VideoMediaIO

    video_path = "/projects/gemma4-vllm/videos/test_intel.mp4"
    if not os.path.exists(video_path):
        return

    with open(video_path, "rb") as f:
        data = f.read()

    # VideoMediaIO defaults to num_frames=32
    video_io = VideoMediaIO(image_io=MagicMock(), video_backend="embedding_gemma2")
    loaded = video_io.load_bytes(data)
    frames, meta = loaded.media
    assert len(frames) == 32
    assert meta.get("do_sample_frames") is False

    # MediaConnector path
    connector = MediaConnector(allowed_local_media_path="/projects/gemma4-vllm/videos")
    fetched = connector.fetch_video(
        f"file://{video_path}",
        video_processor="EmbeddingGemma2VideoProcessor",
    )
    frames_conn, meta_conn = fetched.media
    assert len(frames_conn) == 32
    assert meta_conn.get("do_sample_frames") is False


def test_config_capping_and_explicit_limits(caplog):
    """Verify EmbeddingGemma2ModelConfig capping and explicit user overrides."""
    import logging
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.model_executor.models.config import (
        EmbeddingGemma2ModelConfig,
        Gemma4Config,
    )
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    # (a) Defaults: uncapped max_model_len=262144 -> capped to 8192,
    # and derived scheduler limits (>= 262144) lowered to 8192.
    cfg_default = SimpleNamespace(
        attention_config=SimpleNamespace(backend=None),
        model_config=SimpleNamespace(
            model="dummy", max_model_len=262144, original_max_model_len=None
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=262144,
            max_num_encoder_input_tokens=262144,
            encoder_cache_size=262144,
            max_num_seqs=256,
            verify_max_model_len=lambda x: None,
        ),
    )
    with patch.object(Gemma4Config, "verify_and_update_config"):
        EmbeddingGemma2ModelConfig.verify_and_update_config(cfg_default)

    assert cfg_default.attention_config.backend == AttentionBackendEnum.TRITON_ATTN
    assert cfg_default.model_config.max_model_len == 8192
    assert cfg_default.scheduler_config.max_num_batched_tokens == 8192
    assert cfg_default.scheduler_config.max_num_encoder_input_tokens == 8192
    assert cfg_default.scheduler_config.encoder_cache_size == 8192

    # (b) Reachable explicit limits: explicit max_model_len=8192 and explicit
    # max_num_batched_tokens=32768 stays 32768.
    # Note: omitting max_model_len while passing max_num_batched_tokens < 262144
    # is rejected earlier by SchedulerConfig validation (32768 < 262144) before
    # this hook runs.
    cfg_explicit_batched = SimpleNamespace(
        attention_config=SimpleNamespace(backend=None),
        model_config=SimpleNamespace(
            model="dummy", max_model_len=8192, original_max_model_len=8192
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=32768,
            max_num_encoder_input_tokens=32768,
            encoder_cache_size=32768,
            max_num_seqs=256,
            verify_max_model_len=lambda x: None,
        ),
    )
    with patch.object(Gemma4Config, "verify_and_update_config"):
        EmbeddingGemma2ModelConfig.verify_and_update_config(cfg_explicit_batched)

    assert cfg_explicit_batched.model_config.max_model_len == 8192
    assert cfg_explicit_batched.scheduler_config.max_num_batched_tokens == 32768
    assert cfg_explicit_batched.scheduler_config.max_num_encoder_input_tokens == 32768
    assert cfg_explicit_batched.scheduler_config.encoder_cache_size == 32768

    # (c) Explicit value > uncapped_len (e.g. 300000) with uncapped model len is lowered
    # to 8192 and logs a warning because it exceeds uncapped_len.
    cfg_explicit_large = SimpleNamespace(
        attention_config=SimpleNamespace(backend=None),
        model_config=SimpleNamespace(
            model="dummy", max_model_len=262144, original_max_model_len=None
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=300000,
            max_num_encoder_input_tokens=300000,
            encoder_cache_size=300000,
            max_num_seqs=256,
            verify_max_model_len=lambda x: None,
        ),
    )
    with (
        caplog.at_level(logging.WARNING),
        patch.object(Gemma4Config, "verify_and_update_config"),
    ):
        EmbeddingGemma2ModelConfig.verify_and_update_config(cfg_explicit_large)

    assert cfg_explicit_large.model_config.max_model_len == 8192
    assert cfg_explicit_large.scheduler_config.max_num_batched_tokens == 8192
    assert cfg_explicit_large.scheduler_config.max_num_encoder_input_tokens == 8192
    assert cfg_explicit_large.scheduler_config.encoder_cache_size == 8192
    assert any(
        "lowering explicitly set scheduler_config.max_num_batched_tokens"
        " from 300000 to 8192" in rec.message
        for rec in caplog.records
    )

    # (d) Explicit max_model_len=16384 is respected and not capped to 8192
    cfg_explicit_model_len = SimpleNamespace(
        attention_config=SimpleNamespace(backend=None),
        model_config=SimpleNamespace(
            model="dummy", max_model_len=16384, original_max_model_len=16384
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=16384,
            max_num_encoder_input_tokens=16384,
            encoder_cache_size=16384,
            max_num_seqs=256,
            verify_max_model_len=lambda x: None,
        ),
    )
    with patch.object(Gemma4Config, "verify_and_update_config"):
        EmbeddingGemma2ModelConfig.verify_and_update_config(cfg_explicit_model_len)

    assert cfg_explicit_model_len.model_config.max_model_len == 16384
    assert cfg_explicit_model_len.scheduler_config.max_num_batched_tokens == 16384
