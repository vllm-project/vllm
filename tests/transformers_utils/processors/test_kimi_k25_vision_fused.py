# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for Kimi fused vision image loading."""

from pathlib import Path

import pytest
from PIL import Image

from vllm.transformers_utils.processors.kimi_k25_vision_fused import (
    _ensure_media_type,
    _to_pil,
)

_UNRESOLVED_MEDIA = (
    "Local image paths and URLs must be resolved by "
    "MediaConnector before they reach the processor."
)


def test_pil_image_is_returned_as_rgb():
    image = Image.new("L", (2, 2), color=3)
    loaded = _to_pil(image)
    assert loaded.mode == "RGB"
    assert loaded.size == (2, 2)


def test_path_url_and_file_url_strings_are_rejected(tmp_path: Path):
    image_path = tmp_path / "pixel.png"
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(image_path)
    for raw in (str(image_path), f"file://{image_path}", "http://example.test/a.png"):
        with pytest.raises(ValueError, match=_UNRESOLVED_MEDIA):
            _to_pil(raw)


def test_missing_and_non_image_paths_raise_the_same_error(tmp_path: Path):
    missing = tmp_path / "missing.png"
    text_file = tmp_path / "notes.txt"
    text_file.write_text("not an image", encoding="utf-8")
    for raw in (str(missing), str(text_file)):
        with pytest.raises(ValueError, match=_UNRESOLVED_MEDIA):
            _to_pil(raw)


def test_video_chunk_path_frame_is_rejected(tmp_path: Path):
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (2, 2), color=(4, 5, 6)).save(image_path)
    with pytest.raises(ValueError, match=_UNRESOLVED_MEDIA):
        _ensure_media_type({"type": "video_chunk", "video_chunk": [str(image_path)]})
