# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import struct
from pathlib import Path

import pybase64 as base64
import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.benchmarks.datasets import CustomMMDataset, SampleRequest
from vllm.benchmarks.datasets.datasets import process_audio


def _make_wav_bytes(num_samples: int = 800, sample_rate: int = 16000) -> bytes:
    """Create a minimal valid WAV file (PCM 16-bit mono) in memory."""
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,  # chunk size
        1,  # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    # Silent audio (all zeros)
    samples = b"\x00\x00" * num_samples
    return header + samples


@pytest.fixture(scope="session")
def hf_tokenizer() -> PreTrainedTokenizerBase:
    """Use a small, commonly available tokenizer."""
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    """Create a temporary WAV file for testing."""
    wav_path = tmp_path / "test.wav"
    wav_path.write_bytes(_make_wav_bytes())
    return wav_path


@pytest.fixture
def audio_jsonl(tmp_path: Path) -> Path:
    """Create a temporary JSONL file with audio_files entries."""
    wav_dir = tmp_path / "audio"
    wav_dir.mkdir()

    wav_paths = []
    for i in range(3):
        p = wav_dir / f"clip_{i}.wav"
        p.write_bytes(_make_wav_bytes(num_samples=800 + i * 100))
        wav_paths.append(str(p))

    jsonl_path = tmp_path / "dataset.jsonl"
    lines = []
    for wav_path in wav_paths:
        lines.append(
            json.dumps(
                {
                    "prompt": f"Transcribe this audio clip: {wav_path}",
                    "audio_files": [wav_path],
                    "output_tokens": 64,
                }
            )
        )
    jsonl_path.write_text("\n".join(lines) + "\n")
    return jsonl_path


# ---------------------------------------------------------------------------
# process_audio tests
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_process_audio_from_file_path(wav_file: Path):
    """Test process_audio with a local file path string."""
    result = process_audio(str(wav_file))

    assert isinstance(result, dict)
    assert result["type"] == "input_audio"
    assert "input_audio" in result
    assert result["input_audio"]["format"] == "wav"

    # Verify base64 data decodes back to the original bytes
    decoded = base64.b64decode(result["input_audio"]["data"])
    assert decoded == wav_file.read_bytes()


@pytest.mark.benchmark
def test_process_audio_from_bytes_dict():
    """Test process_audio with a dict containing raw bytes."""
    raw = _make_wav_bytes()
    result = process_audio({"bytes": raw})

    assert result["type"] == "input_audio"
    assert result["input_audio"]["format"] == "wav"  # default

    decoded = base64.b64decode(result["input_audio"]["data"])
    assert decoded == raw


@pytest.mark.benchmark
def test_process_audio_from_bytes_dict_custom_format():
    """Test process_audio with explicit format in bytes dict."""
    raw = _make_wav_bytes()
    result = process_audio({"bytes": raw, "format": "mp3"})

    assert result["input_audio"]["format"] == "mp3"


@pytest.mark.benchmark
def test_process_audio_format_from_extension(tmp_path: Path):
    """Test that process_audio infers format from file extension."""
    for ext, expected_fmt in [("wav", "wav"), ("mp3", "mp3"), ("flac", "flac")]:
        p = tmp_path / f"test.{ext}"
        p.write_bytes(_make_wav_bytes())  # content doesn't matter for this test
        result = process_audio(str(p))
        assert result["input_audio"]["format"] == expected_fmt


@pytest.mark.benchmark
def test_process_audio_invalid_input():
    """Test that process_audio raises ValueError for unsupported types."""
    with pytest.raises(ValueError, match="Invalid audio input"):
        process_audio(12345)

    with pytest.raises(ValueError, match="Invalid audio input"):
        process_audio(["not", "a", "valid", "input"])


# ---------------------------------------------------------------------------
# CustomMMDataset with audio_files tests
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_custom_mm_dataset_audio_only(
    audio_jsonl: Path, hf_tokenizer: PreTrainedTokenizerBase
):
    """Test CustomMMDataset loading JSONL with audio_files (no images)."""
    ds = CustomMMDataset(dataset_path=str(audio_jsonl))
    ds.load_data()
    assert len(ds.data) == 3

    samples = ds.sample(tokenizer=hf_tokenizer, num_requests=3, output_len=64)
    assert len(samples) == 3

    for sample in samples:
        assert isinstance(sample, SampleRequest)
        assert sample.multi_modal_data is not None
        assert isinstance(sample.multi_modal_data, dict)
        assert sample.multi_modal_data["type"] == "input_audio"
        assert "data" in sample.multi_modal_data["input_audio"]
        assert sample.multi_modal_data["input_audio"]["format"] == "wav"
        assert sample.expected_output_len == 64
        assert sample.prompt_len > 0


@pytest.mark.benchmark
def test_custom_mm_dataset_mixed_image_and_audio(
    tmp_path: Path, hf_tokenizer: PreTrainedTokenizerBase
):
    """Test CustomMMDataset with both image_files and audio_files."""
    # Create a WAV file
    wav_path = tmp_path / "clip.wav"
    wav_path.write_bytes(_make_wav_bytes())

    # Create a tiny JPEG file
    img_path = tmp_path / "img.jpg"
    # Minimal valid JPEG (1x1 white pixel)
    img_path.write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
        b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
        b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05"
        b"\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06"
        b'\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br'
        b"\x82\t\n\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZcde"
        b"fghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95"
        b"\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2"
        b"\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8"
        b"\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4"
        b"\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa"
        b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00T\xdb\xa8\xa0\x02\x80\x0f\xff\xd9"
    )

    jsonl_path = tmp_path / "mixed.jsonl"
    entry = {
        "prompt": "Describe the image and audio",
        "image_files": [str(img_path)],
        "audio_files": [str(wav_path)],
        "output_tokens": 128,
    }
    jsonl_path.write_text(json.dumps(entry) + "\n")

    ds = CustomMMDataset(dataset_path=str(jsonl_path))
    ds.load_data()

    samples = ds.sample(tokenizer=hf_tokenizer, num_requests=1, output_len=128)
    assert len(samples) == 1

    sample = samples[0]
    # With both image and audio, mm_data should be a list
    assert isinstance(sample.multi_modal_data, list)
    assert len(sample.multi_modal_data) == 2

    types = {item["type"] for item in sample.multi_modal_data}
    assert types == {"image_url", "input_audio"}


@pytest.mark.benchmark
def test_custom_mm_dataset_no_media_raises(
    tmp_path: Path, hf_tokenizer: PreTrainedTokenizerBase
):
    """Test that a JSONL row with neither image_files nor audio_files raises."""
    jsonl_path = tmp_path / "empty.jsonl"
    jsonl_path.write_text(json.dumps({"prompt": "no media"}) + "\n")

    ds = CustomMMDataset(dataset_path=str(jsonl_path))
    ds.load_data()

    with pytest.raises(ValueError, match="neither 'image_files' nor 'audio_files'"):
        ds.sample(tokenizer=hf_tokenizer, num_requests=1, output_len=64)


@pytest.mark.benchmark
def test_custom_mm_dataset_multiple_audio_files(
    tmp_path: Path, hf_tokenizer: PreTrainedTokenizerBase
):
    """Test that multiple audio_files per row are all included."""
    wav_paths = []
    for i in range(3):
        p = tmp_path / f"clip_{i}.wav"
        p.write_bytes(_make_wav_bytes(num_samples=800 + i * 100))
        wav_paths.append(str(p))

    jsonl_path = tmp_path / "multi_audio.jsonl"
    entry = {
        "prompt": "Transcribe all clips",
        "audio_files": wav_paths,
        "output_tokens": 64,
    }
    jsonl_path.write_text(json.dumps(entry) + "\n")

    ds = CustomMMDataset(dataset_path=str(jsonl_path))
    ds.load_data()

    samples = ds.sample(tokenizer=hf_tokenizer, num_requests=1, output_len=64)
    assert len(samples) == 1

    sample = samples[0]
    # Multiple audio files → list of content dicts
    assert isinstance(sample.multi_modal_data, list)
    assert len(sample.multi_modal_data) == 3
    for item in sample.multi_modal_data:
        assert item["type"] == "input_audio"


@pytest.mark.benchmark
def test_custom_mm_dataset_oversampling(
    audio_jsonl: Path, hf_tokenizer: PreTrainedTokenizerBase
):
    """Test that requesting more samples than data triggers oversampling."""
    ds = CustomMMDataset(dataset_path=str(audio_jsonl))
    ds.load_data()
    assert len(ds.data) == 3

    samples = ds.sample(tokenizer=hf_tokenizer, num_requests=6, output_len=64)
    assert len(samples) == 6
