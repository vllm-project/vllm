# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image

from tests.multimodal.utils import create_video_from_image
from vllm import envs
from vllm.multimodal.media.base import MediaDecodedBudget

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.fixture(scope="module")
def decode_frames():
    path = (
        Path(__file__).resolve().parents[4] / "vllm/models/dots3_note/common/video.py"
    )
    spec = importlib.util.spec_from_file_location("dots3_note_video_budget", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._decode_frames


def _video_bytes(tmp_path: Path) -> bytes:
    image_path = tmp_path / "frame.png"
    video_path = tmp_path / "video.mp4"
    Image.new("RGB", (64, 64), color="red").save(image_path)
    create_video_from_image(str(image_path), str(video_path), num_frames=8)
    return video_path.read_bytes()


def test_secondary_video_decode_rejects_when_request_budget_is_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decode_frames
):
    frame_bytes = 64 * 64 * 3
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", 11 * frame_bytes)
    budget = MediaDecodedBudget()
    budget.charge(8 * frame_bytes)

    with pytest.raises(ValueError, match="VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES"):
        decode_frames(_video_bytes(tmp_path), 4096, 4096, 85, budget)

    assert budget.used == 8 * frame_bytes


def test_secondary_video_decode_rejects_without_request_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decode_frames
):
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", 3 * 64 * 64 * 3)

    with pytest.raises(ValueError, match="VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES"):
        decode_frames(_video_bytes(tmp_path), 4096, 4096, 85)


def test_secondary_video_decode_accepts_frames_within_request_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decode_frames
):
    frame_bytes = 64 * 64 * 3
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", 16 * frame_bytes)
    budget = MediaDecodedBudget()
    budget.charge(8 * frame_bytes)

    frames, duration = decode_frames(_video_bytes(tmp_path), 4096, 4096, 85, budget)

    assert 4 <= len(frames) <= 8
    assert duration > 0
    assert budget.used == (8 + len(frames)) * frame_bytes


def test_secondary_video_decode_checks_frame_pixel_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decode_frames
):
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 64 * 64 - 1)

    with pytest.raises(ValueError, match="VLLM_MAX_IMAGE_PIXELS"):
        decode_frames(_video_bytes(tmp_path), 4096, 4096, 85)
