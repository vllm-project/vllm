# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CustomMMDataset."""

import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.benchmarks.datasets import CustomMMDataset


@pytest.fixture(scope="session")
def tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("gpt2")


def _dataset(tmp_path: Path, rows: list[dict]) -> CustomMMDataset:
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in rows))
    return CustomMMDataset(dataset_path=str(jsonl), random_seed=0)


@pytest.mark.benchmark
def test_video_files_produces_video_url(
    tokenizer: PreTrainedTokenizerBase, tmp_path: Path
):
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"")
    ds = _dataset(tmp_path, [{"prompt": "p", "video_files": [str(vid)]}])
    [sample] = ds.sample(tokenizer=tokenizer, num_requests=1, output_len=32)
    assert sample.multi_modal_data["type"] == "video_url"
    assert sample.multi_modal_data["video_url"]["url"] == f"file://{vid}"


@pytest.mark.benchmark
def test_image_files_produces_image_url(
    tokenizer: PreTrainedTokenizerBase, tmp_path: Path
):
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"")
    ds = _dataset(tmp_path, [{"prompt": "p", "image_files": [str(img)]}])
    [sample] = ds.sample(tokenizer=tokenizer, num_requests=1, output_len=32)
    assert sample.multi_modal_data["type"] == "image_url"
    assert sample.multi_modal_data["image_url"]["url"] == f"file://{img}"


@pytest.mark.benchmark
def test_image_files_takes_priority_over_video_files(
    tokenizer: PreTrainedTokenizerBase, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """When both keys are present image_files wins and a warning is emitted."""
    vid, img = tmp_path / "clip.mp4", tmp_path / "frame.jpg"
    vid.write_bytes(b"")
    img.write_bytes(b"")
    ds = _dataset(
        tmp_path,
        [{"prompt": "p", "video_files": [str(vid)], "image_files": [str(img)]}],
    )
    with caplog.at_level("WARNING"):
        [sample] = ds.sample(tokenizer=tokenizer, num_requests=1, output_len=32)
    assert any("Only images will be used" in r.message for r in caplog.records)
    assert sample.multi_modal_data["type"] == "image_url"


@pytest.mark.benchmark
def test_multiple_files_warns_and_uses_first(
    tokenizer: PreTrainedTokenizerBase, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Multiple files in a list: warning emitted and only the first is used."""
    vid1, vid2 = tmp_path / "a.mp4", tmp_path / "b.mp4"
    vid1.write_bytes(b"")
    vid2.write_bytes(b"")
    ds = _dataset(tmp_path, [{"prompt": "p", "video_files": [str(vid1), str(vid2)]}])
    with caplog.at_level("WARNING"):
        [sample] = ds.sample(tokenizer=tokenizer, num_requests=1, output_len=32)
    assert any("Only the first will be used" in r.message for r in caplog.records)
    assert sample.multi_modal_data["video_url"]["url"] == f"file://{vid1}"
