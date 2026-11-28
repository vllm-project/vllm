# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the LLaVA-OneVision-2 codec-video marker mechanism.

The codec video backend cannot be exercised end-to-end in CI because it
relies on the model's `trust_remote_code` package and real video bytes.
These tests cover the pure-Python marker plumbing that wraps a video *path*
for vLLM's `MultiModalDataParser`:

* `prepare_codec_video_input` must encode a per-path hash into its
  dummy ndarray so distinct codec videos do not collide on `mm_hash`
  (otherwise `EncoderCacheManager` would skip the encoder for every video
  after the first).
* `_extract_codec_video_paths` must round-trip the marker back to the
  original path(s) after the parser strips the metadata dict, and must return
  `None` for non-codec inputs.
"""

import types

import numpy as np
import pytest
from PIL import Image

from vllm.config.scheduler import SchedulerConfig
from vllm.model_executor.models.llava_onevision2 import (
    _CODEC_VIDEO_MARKER,
    LlavaOnevision2VideoBackend,
    _extract_codec_video_paths,
    _frame_video_to_pil_and_timestamps,
    _validate_video_source,
    prepare_codec_video_input,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.multimodal.video import VideoSourceMetadata, VideoTargetMetadata
from vllm.v1.core.encoder_cache_manager import compute_mm_encoder_budget

from ...utils import build_model_context


def _model_config(local: str = "", domains=None):
    # Minimal stand-in for vLLM's ModelConfig: the validator only reads
    # ``allowed_local_media_path`` and ``allowed_media_domains``.
    return types.SimpleNamespace(
        allowed_local_media_path=local,
        allowed_media_domains=domains,
    )


def test_prepare_codec_video_input_shape_and_marker():
    dummy, meta = prepare_codec_video_input("/data/foo.mp4")

    # 4-D ndarray satisfies MultiModalDataParser's video shape check.
    assert isinstance(dummy, np.ndarray)
    assert dummy.shape == (1, 1, 16, 3)
    assert dummy.dtype == np.uint8

    # Metadata carries the exact path under the codec marker key.
    assert meta == {_CODEC_VIDEO_MARKER: "/data/foo.mp4"}


def test_prepare_codec_video_input_is_deterministic():
    # Same path must yield identical dummy bytes (stable mm_hash).
    a, _ = prepare_codec_video_input("/data/foo.mp4")
    b, _ = prepare_codec_video_input("/data/foo.mp4")
    assert a.tobytes() == b.tobytes()


def test_prepare_codec_video_input_distinct_paths_distinct_bytes():
    # Distinct paths must yield distinct dummy bytes so the parser-visible
    # ndarray (the only part reaching MultiModalHasher) varies per video.
    paths = ["/data/a.mp4", "/data/b.mp4", "/data/c.mp4", "/data/a_.mp4"]
    payloads = {prepare_codec_video_input(p)[0].tobytes() for p in paths}
    assert len(payloads) == len(paths)


def test_extract_codec_video_paths_parser_list_shape():
    # Parser yields list-of-(ndarray, metadata-dict) for tuple inputs.
    items = [prepare_codec_video_input(p) for p in ("/x/1.mp4", "/x/2.mp4")]
    assert _extract_codec_video_paths(items) == ["/x/1.mp4", "/x/2.mp4"]


def test_extract_codec_video_paths_single_raw_tuple():
    # Single raw (ndarray, dict) tuple (pre-parser path) is also accepted.
    item = prepare_codec_video_input("/x/solo.mp4")
    assert _extract_codec_video_paths(item) == ["/x/solo.mp4"]


def test_extract_codec_video_paths_non_codec_returns_none():
    # Plain decoded-frame inputs (ndarray / list of ndarray) are not codec
    # markers and must fall through to the frame backend.
    plain = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    assert _extract_codec_video_paths(plain) is None
    assert _extract_codec_video_paths([plain, plain]) is None
    assert _extract_codec_video_paths([]) is None
    # Tuple without the marker key is ignored.
    assert _extract_codec_video_paths((plain, {"fps": 2.0})) is None


def test_extract_codec_video_paths_mixed_batch_returns_none():
    # If any item in the batch lacks the marker, the whole batch is treated
    # as non-codec (the backend does not mix codec and frame videos).
    codec = prepare_codec_video_input("/x/1.mp4")
    plain = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    assert _extract_codec_video_paths([codec, plain]) is None


# ---------------------------------------------------------------------------
# Media access controls (_validate_video_source)
#
# The codec backend keeps the raw path string alive past vLLM's
# MultiModalDataParser and hands it to the trust-remote-code codec module,
# which opens it directly (cv2/ffmpeg) outside vLLM's MediaConnector. So the
# codec backend is restricted to *local files* confined to
# --allowed-local-media-path; remote http(s)/data URLs are rejected (they must
# use the frame backend, which rides the connector). The validator also returns
# the *resolved* path so the codec module opens exactly what was validated
# (no symlink-retarget / validate-vs-open gap).
# ---------------------------------------------------------------------------


def test_validate_http_url_rejected():
    # Codec backend is local-only: remote URLs bypass the connector's domain
    # and redirect controls, so they are rejected regardless of allowlist.
    with pytest.raises(ValueError):
        _validate_video_source("http://example.com/v.mp4", _model_config())


def test_validate_https_url_rejected_even_when_host_allowlisted():
    with pytest.raises(ValueError):
        _validate_video_source(
            "https://good.com/v.mp4",
            _model_config(domains=["good.com"]),
        )


def test_validate_data_url_rejected():
    # data: URLs are a remote/inline source for the frame backend, not codec.
    with pytest.raises(ValueError):
        _validate_video_source("data:video/mp4;base64,AAAA", _model_config())


def test_validate_local_file_blocked_without_allowed_path():
    # Local file access is opt-in: without --allowed-local-media-path the
    # bare path is rejected.
    with pytest.raises(ValueError):
        _validate_video_source("/data/v.mp4", _model_config())


def test_validate_local_file_allowed_inside_allowed_dir(tmp_path):
    f = tmp_path / "v.mp4"
    f.touch()
    assert _validate_video_source(str(f), _model_config(local=str(tmp_path))) == str(f)


def test_validate_local_file_traversal_blocked():
    # Path traversal escaping the allowed root is rejected after resolution.
    with pytest.raises(ValueError):
        _validate_video_source(
            "/tmp/ov2/../../etc/passwd",
            _model_config(local="/tmp/ov2"),
        )


def test_validate_file_scheme_allowed_inside_allowed_dir(tmp_path):
    f = tmp_path / "v.mp4"
    f.touch()
    assert _validate_video_source(
        f.as_uri(), _model_config(local=str(tmp_path))
    ) == str(f)


def test_validate_file_scheme_percent_encoded_traversal_blocked():
    # ``%2e%2e`` decodes to ``..``; the confinement check URL-decodes first
    # (url2pathname) so the path cannot stay literally under the allowed root.
    with pytest.raises(ValueError):
        _validate_video_source(
            "file:///tmp/ov2/%2e%2e/%2e%2e/etc/passwd",
            _model_config(local="/tmp/ov2"),
        )


def test_validate_unsupported_scheme_blocked():
    with pytest.raises(ValueError):
        _validate_video_source("ftp://example.com/v.mp4", _model_config())


def test_validate_returns_resolved_path_through_symlink(tmp_path):
    # The validator resolves symlinks and returns the *real* path, so the codec
    # module opens exactly what was validated (closes the validate-vs-open gap).
    real = tmp_path / "real.mp4"
    real.touch()
    link = tmp_path / "link.mp4"
    link.symlink_to(real)
    assert _validate_video_source(str(link), _model_config(local=str(tmp_path))) == str(
        real
    )


def test_validate_relative_bare_path_blocked():
    # Bare (scheme=="") paths come only from the codec backend and must be
    # absolute: resolving a relative path against an ambiguous CWD before the
    # confinement check is brittle/unsafe, so it is rejected outright.
    with pytest.raises(ValueError):
        _validate_video_source("ov2/v.mp4", _model_config(local="/tmp/ov2"))


# ---------------------------------------------------------------------------
# Frame backend marker -> PIL + timestamps (_frame_video_to_pil_and_timestamps)
#
# Non-codec videos reach _apply_hf_processor_main as a ``(frames_ndarray, metadata)``
# tuple -- produced by the registered ``LlavaOnevision2VideoBackend`` for real
# ``video_url`` inputs, or by the dummy-inputs builder during profiling
# (``video_needs_metadata=True``). The helper materialises PIL frames and
# per-frame timestamps (``frame_index / fps``), padding the frame count up to
# the even temporal-merge boundary.
# ---------------------------------------------------------------------------


def test_frame_video_to_pil_and_timestamps_basic():
    frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    metadata = {"fps": 2.0, "frames_indices": [0, 4, 8, 12]}
    pil_frames, timestamps = _frame_video_to_pil_and_timestamps((frames, metadata))

    assert len(pil_frames) == 4
    assert all(f.size == (8, 8) for f in pil_frames)
    # timestamps = frame_index / fps
    assert timestamps == [0.0, 2.0, 4.0, 6.0]


def test_frame_video_to_pil_and_timestamps_even_pads_odd_frame_count():
    # Odd frame count -> last frame repeated to satisfy temporal merge=2.
    frames = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    metadata = {"fps": 1.0, "frames_indices": [0, 1, 2]}
    pil_frames, timestamps = _frame_video_to_pil_and_timestamps((frames, metadata))

    assert len(pil_frames) == 4
    assert len(timestamps) == 4
    # The padded frame reuses the final index/timestamp.
    assert timestamps == [0.0, 1.0, 2.0, 2.0]


def test_frame_video_to_pil_and_timestamps_defaults_when_metadata_sparse():
    # Missing frames_indices -> sequential range; missing/zero fps -> default.
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    pil_frames, timestamps = _frame_video_to_pil_and_timestamps((frames, {}))

    assert len(pil_frames) == 2
    # Default fps is 1.0, indices fall back to range(T).
    assert timestamps == [0.0, 1.0]


def test_frame_video_to_pil_and_timestamps_rejects_non_tuple():
    # Bare arrays (no metadata) must be rejected: the frame backend requires
    # the (frames, metadata) tuple produced by the registered loader.
    plain = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        _frame_video_to_pil_and_timestamps(plain)


# ---------------------------------------------------------------------------
# LlavaOnevision2VideoBackend.compute_frames_index_to_sample honors the caller
# supplied VideoTargetMetadata (passed via --media-io-kwargs) so benchmarks can
# override the conservative defaults (fps=1.0, max_frames=32). Unset target
# fields (sentinel <= 0) fall back to those OV2 hf-chat reference constants.
# ---------------------------------------------------------------------------


def _src(total_frames: int, fps: float) -> VideoSourceMetadata:
    return VideoSourceMetadata(
        total_frames_num=total_frames,
        original_fps=fps,
        duration=total_frames / fps if fps > 0 else 0.0,
    )


def test_backend_defaults_cap_at_32_frames():
    # 300 frames @ 1fps source, target unset -> capped at default max_frames=32.
    src = _src(300, 1.0)
    target = VideoTargetMetadata(num_frames=-1, fps=-1, max_duration=300.0)
    idx = LlavaOnevision2VideoBackend.compute_frames_index_to_sample(src, target)

    assert len(idx) == 32
    assert idx[0] == 0
    assert idx[-1] == 299
    assert len(idx) % 2 == 0


def test_backend_target_num_frames_overrides_default_cap():
    # VSI-Bench parity: target.num_frames=128 must lift the 32-frame cap.
    src = _src(300, 1.0)
    target = VideoTargetMetadata(num_frames=128, fps=-1, max_duration=300.0)
    idx = LlavaOnevision2VideoBackend.compute_frames_index_to_sample(src, target)

    assert len(idx) == 128
    assert idx[0] == 0
    assert idx[-1] == 299


def test_backend_target_fps_controls_sampling_when_below_cap():
    # 60 frames @ 30fps (2s) with target fps=1 -> ~2 frames (even-padded).
    src = _src(60, 30.0)
    target = VideoTargetMetadata(num_frames=128, fps=1.0, max_duration=300.0)
    idx = LlavaOnevision2VideoBackend.compute_frames_index_to_sample(src, target)

    # fps-derived nframes (2) is below the 128 cap, so fps wins.
    assert len(idx) <= 8
    assert len(idx) % 2 == 0


@pytest.mark.parametrize("max_model_len", [8192, 20000, 32768, 65536])
def test_dummy_video_matches_video_budget(max_model_len: int):
    ctx = build_model_context(
        "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct",
        model_config_kwargs={"max_model_len": max_model_len},
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    budget = processor.info.get_mm_max_tokens_per_item(max_model_len, {"video": 1})
    assert budget is not None
    dummy = processor.get_dummy_mm_inputs({"video": 1})
    num_embeds = sum(p.get_num_embeds() for p in dummy["mm_placeholders"]["video"])

    assert num_embeds == budget["video"]


_LARGE_IMAGES = [(2048, 2048), (2560, 1440), (4000, 1000), (2100, 1904)]


@pytest.mark.parametrize(
    ("mm_processor_kwargs", "image_sizes"),
    [
        (None, _LARGE_IMAGES),
        ({"max_pixels": 1003520}, _LARGE_IMAGES),
        ({"min_pixels": 1605632, "max_pixels": 1605632}, [(560, 560), (540, 960)]),
        ({"min_pixels": 196, "max_pixels": 196}, [(28, 28), (2, 393)]),
        ({"max_pixels": 2574656}, [(22988, 116), (116, 22988)]),
        ({"max_pixels": 50176}, [(5600, 28), (28, 5600)]),
    ],
)
def test_image_budget_covers_largest_images(
    mm_processor_kwargs: dict | None, image_sizes: list[tuple[int, int]]
):
    ctx = build_model_context(
        "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct",
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"image": 1, "video": 0},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    max_tokens = processor.info.get_mm_max_tokens_per_item(8192, {"image": 1})
    assert max_tokens is not None
    budget = max_tokens["image"]
    dummy = processor.get_dummy_mm_inputs({"image": 1})
    assert sum(p.get_num_embeds() for p in dummy["mm_placeholders"]["image"]) == budget

    prompt = processor.info.get_tokenizer().encode(
        "<|vision_start|><|image_pad|><|vision_end|>"
    )
    for width, height in image_sizes:
        inputs = ProcessorInputs(
            prompt=prompt,
            mm_data_items=processor.info.parse_mm_data(
                {"image": Image.new("RGB", (width, height))}
            ),
        )
        real = processor.apply(inputs, TimingContext(enabled=False))
        num_embeds = sum(p.get_num_embeds() for p in real["mm_placeholders"]["image"])
        assert num_embeds <= budget


@pytest.mark.parametrize(
    ("max_model_len", "num_frames", "width", "height"),
    [(20000, 4, 2520, 1400), (65536, 12, 3840, 2160)],
)
def test_video_budget_covers_videos_that_fit(
    max_model_len: int, num_frames: int, width: int, height: int
):
    ctx = build_model_context(
        "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct",
        model_config_kwargs={"max_model_len": max_model_len},
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    max_tokens = processor.info.get_mm_max_tokens_per_item(max_model_len, {"video": 1})
    assert max_tokens is not None

    video = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    metadata = {
        "fps": 1.0,
        "duration": float(num_frames),
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
        "video_backend": "llava_onevision2",
        "do_sample_frames": False,
    }
    inputs = ProcessorInputs(
        prompt=processor.info.get_tokenizer().encode(
            "<|vision_start|><|video_pad|><|vision_end|>"
        ),
        mm_data_items=processor.info.parse_mm_data({"video": [(video, metadata)]}),
    )
    real = processor.apply(inputs, TimingContext(enabled=False))
    num_embeds = sum(p.get_num_embeds() for p in real["mm_placeholders"]["video"])

    assert len(real["prompt_token_ids"]) <= max_model_len
    assert num_embeds <= max_tokens["video"]


@pytest.mark.parametrize(
    ("max_model_len", "mm_processor_kwargs"),
    [
        (4096, None),
        (8192, None),
        (4096, {"min_pixels": 1646400, "max_pixels": 1646400}),
    ],
)
def test_video_budget_fits_unchunked_batch(
    max_model_len: int, mm_processor_kwargs: dict | None
):
    ctx = build_model_context(
        "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct",
        mm_processor_kwargs=mm_processor_kwargs,
        model_config_kwargs={"max_model_len": max_model_len},
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    max_tokens = processor.info.get_mm_max_tokens_per_item(max_model_len, {"video": 1})
    assert max_tokens is not None
    scheduler_config = SchedulerConfig(
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        max_num_batched_tokens=max_model_len,
        disable_chunked_mm_input=True,
    )

    assert compute_mm_encoder_budget(
        scheduler_config, {"video": max_tokens["video"]}
    ) == (max_model_len, max_model_len)
    dummy = processor.get_dummy_mm_inputs({"video": 1})
    assert (
        sum(p.get_num_embeds() for p in dummy["mm_placeholders"]["video"])
        == max_tokens["video"]
    )
