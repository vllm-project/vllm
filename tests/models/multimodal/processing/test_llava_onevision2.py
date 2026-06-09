# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the LLaVA-OneVision-2 codec-video marker mechanism.

The codec video backend cannot be exercised end-to-end in CI because it
relies on the model's ``trust_remote_code`` package and real video bytes.
These tests cover the pure-Python marker plumbing that wraps a video *path*
for vLLM's ``MultiModalDataParser``:

* :func:`prepare_codec_video_input` must encode a per-path hash into its
  dummy ndarray so distinct codec videos do not collide on ``mm_hash``
  (otherwise ``EncoderCacheManager`` would skip the encoder for every video
  after the first).
* :func:`_extract_codec_video_paths` must round-trip the marker back to the
  original path(s) after the parser strips the metadata dict, and must return
  ``None`` for non-codec inputs.
"""

import types

import numpy as np
import pytest

from vllm.model_executor.models.llava_onevision2 import (
    _CODEC_VIDEO_MARKER,
    _extract_codec_video_paths,
    _extract_video_paths,
    _validate_video_source,
    prepare_codec_video_input,
)


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
# Both video backends keep the raw path/URL string alive past vLLM's
# MultiModalDataParser, so the model re-applies the gates from
# vllm/multimodal/media/connector.py before the string reaches
# qwen_vl_utils.fetch_video / the codec module. These tests pin that the
# self-contained validator matches MediaConnector's behaviour for
# --allowed-media-domains and --allowed-local-media-path.
# ---------------------------------------------------------------------------


def test_validate_http_allowed_when_no_domain_allowlist():
    # No --allowed-media-domains configured: any http(s) host is permitted
    # (matches MediaConnector, which only filters when an allowlist is set).
    _validate_video_source("http://example.com/v.mp4", _model_config())


def test_validate_http_blocked_when_host_not_in_allowlist():
    with pytest.raises(ValueError):
        _validate_video_source(
            "http://evil.com/v.mp4",
            _model_config(domains=["good.com"]),
        )


def test_validate_http_allowed_when_host_in_allowlist():
    _validate_video_source(
        "http://good.com/v.mp4",
        _model_config(domains=["good.com"]),
    )


def test_validate_local_file_blocked_without_allowed_path():
    # Local file access is opt-in: without --allowed-local-media-path the
    # bare path is rejected.
    with pytest.raises(ValueError):
        _validate_video_source("/data/v.mp4", _model_config())


def test_validate_local_file_allowed_inside_allowed_dir():
    _validate_video_source(
        "/tmp/ov2/v.mp4",
        _model_config(local="/tmp/ov2"),
    )


def test_validate_local_file_traversal_blocked():
    # Path traversal escaping the allowed root is rejected after resolution.
    with pytest.raises(ValueError):
        _validate_video_source(
            "/tmp/ov2/../../etc/passwd",
            _model_config(local="/tmp/ov2"),
        )


def test_validate_file_scheme_allowed_inside_allowed_dir():
    _validate_video_source(
        "file:///tmp/ov2/v.mp4",
        _model_config(local="/tmp/ov2"),
    )


def test_validate_data_url_allowed():
    # data: URLs carry inline bytes and need no filesystem/network access.
    _validate_video_source("data:video/mp4;base64,AAAA", _model_config())


def test_validate_unsupported_scheme_blocked():
    with pytest.raises(ValueError):
        _validate_video_source("ftp://example.com/v.mp4", _model_config())


# ---------------------------------------------------------------------------
# Validation engages for *all* path-based backends, including native.
#
# _call_hf_processor validates the result of _extract_video_paths before the
# backend dispatch, so the gate fires whenever a raw path reaches the frame OR
# native backend. These tests pin the precondition: string payloads are
# extracted to paths (so _validate_video_sources runs), while pre-decoded
# inputs return None (validation correctly skipped). This guards against
# silently disabling native-backend validation, which would re-open the
# video_backend="native" SSRF / local-file-read bypass.
# ---------------------------------------------------------------------------


def test_extract_video_paths_returns_paths_for_string_payloads():
    # Single, flat-list, and nested-list path payloads must all yield the raw
    # path strings that _validate_video_sources then checks.
    assert _extract_video_paths("/x/solo.mp4") == ["/x/solo.mp4"]
    assert _extract_video_paths(["/x/1.mp4", "/x/2.mp4"]) == [
        "/x/1.mp4",
        "/x/2.mp4",
    ]
    assert _extract_video_paths([["/x/1.mp4"], ["/x/2.mp4"]]) == [
        "/x/1.mp4",
        "/x/2.mp4",
    ]


def test_extract_video_paths_returns_none_for_predecoded_inputs():
    # Pre-decoded frames (ndarray / list of ndarray) carry no path, so
    # validation is skipped (nothing to fetch from a URL/filesystem).
    plain = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    assert _extract_video_paths(plain) is None
    assert _extract_video_paths([plain, plain]) is None
    assert _extract_video_paths(None) is None
    assert _extract_video_paths([]) is None
