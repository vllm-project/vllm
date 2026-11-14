# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from io import BytesIO
from typing import Any, cast

import pybase64 as base64
import pytest
import soundfile as sf
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.benchmarks.datasets import RandomMultiModalDataset, SampleRequest


@pytest.fixture(scope="session")
def hf_tokenizer() -> PreTrainedTokenizerBase:
    """Use a small, commonly available tokenizer."""
    return AutoTokenizer.from_pretrained("openai-community/gpt2")


@pytest.fixture
def audio_dataset() -> RandomMultiModalDataset:
    """Create a RandomMultiModalDataset instance for testing."""
    return RandomMultiModalDataset(random_seed=42)


@pytest.mark.benchmark
def test_map_config_to_modality_audio(audio_dataset: RandomMultiModalDataset):
    """A leading 0 marks a bucket config tuple as audio, checked before the
    existing image/video num_frames branches."""
    assert audio_dataset.map_config_to_modality((0, 3, 1)) == "audio"
    assert audio_dataset.map_config_to_modality((0, 10, 2)) == "audio"

    # Image/video classification is unaffected by the new audio branch.
    assert audio_dataset.map_config_to_modality((256, 256, 1)) == "image"
    assert audio_dataset.map_config_to_modality((256, 256, 8)) == "video"


@pytest.mark.benchmark
def test_generate_synthetic_audio_different_seeds():
    """Test that different seeds produce different audio content."""
    dataset1 = RandomMultiModalDataset(random_seed=123)
    dataset2 = RandomMultiModalDataset(random_seed=456)

    audio1 = dataset1.generate_synthetic_audio(duration=2, num_channels=1)
    audio2 = dataset2.generate_synthetic_audio(duration=2, num_channels=1)

    assert audio1["bytes"] != audio2["bytes"]


@pytest.mark.benchmark
@pytest.mark.parametrize("sample_rate", [8000, 16000, 44100, 48000])
def test_generate_synthetic_audio_sample_rate(
    audio_dataset: RandomMultiModalDataset, sample_rate: int
):
    """`sample_rate` actually changes the WAV's encoded rate, and duration
    stays exact at any rate (num_samples = sample_rate * duration)."""
    audio = audio_dataset.generate_synthetic_audio(
        duration=3, num_channels=1, sample_rate=sample_rate
    )
    data, decoded_sample_rate = sf.read(BytesIO(audio["bytes"]))

    assert decoded_sample_rate == sample_rate
    assert len(data) / decoded_sample_rate == pytest.approx(3, abs=0.01)


@pytest.mark.benchmark
@pytest.mark.parametrize("duration,num_channels", [(3, 1), (10, 2)])
def test_generate_mm_item_audio(
    audio_dataset: RandomMultiModalDataset, duration: int, num_channels: int
):
    """generate_mm_item() on an audio bucket returns an `input_audio` content
    item whose WAV payload decodes to the requested duration and channel
    count."""
    audio_config = (0, duration, num_channels)
    result = audio_dataset.generate_mm_item(audio_config)

    # Check the result structure matches the OpenAI chat-completions format.
    assert isinstance(result, dict)
    assert result["type"] == "input_audio"
    assert "input_audio" in result
    assert result["input_audio"]["format"] == "wav"

    # Decode the WAV payload and verify duration/channel count.
    wav_bytes = base64.b64decode(result["input_audio"]["data"])
    assert len(wav_bytes) > 0

    data, sample_rate = sf.read(BytesIO(wav_bytes))
    decoded_channels = 1 if data.ndim == 1 else data.shape[1]
    decoded_duration = len(data) / sample_rate

    assert decoded_channels == num_channels
    assert decoded_duration == pytest.approx(duration, abs=0.01)


@pytest.mark.benchmark
def test_sample_mixed_image_and_audio_buckets_respects_audio_limit(
    audio_dataset: RandomMultiModalDataset, hf_tokenizer: PreTrainedTokenizerBase
):
    """sample() with mixed image and audio buckets respects the per-prompt
    audio limit, the same way it already does for image/video limits."""
    bucket_config = {
        (64, 64, 1): 0.5,  # Images
        (0, 3, 1): 0.5,  # 3s mono audio
    }

    # Only one audio item allowed per prompt.
    limit_mm_per_prompt = {"image": 5, "video": 0, "audio": 1}

    samples = audio_dataset.sample(
        tokenizer=hf_tokenizer,
        num_requests=5,
        base_items_per_request=3,
        num_mm_items_range_ratio=0.0,
        limit_mm_per_prompt=limit_mm_per_prompt,
        bucket_config=bucket_config,
        input_len=20,
        output_len=5,
    )

    assert len(samples) == 5

    audio_count = 0
    image_count = 0
    for sample in samples:
        assert isinstance(sample, SampleRequest)
        mm_data = cast(list[dict[str, Any]], sample.multi_modal_data)

        per_request_audio = sum(1 for item in mm_data if item["type"] == "input_audio")
        assert per_request_audio <= 1  # Respects the audio limit

        for item in mm_data:
            if item["type"] == "input_audio":
                audio_count += 1
                assert item["input_audio"]["format"] == "wav"
            elif item["type"] == "image_url":
                image_count += 1

    # Should have some of both due to 0.5/0.5 probability.
    assert audio_count > 0
    assert image_count > 0


@pytest.mark.benchmark
def test_sample_audio_only_buckets(
    audio_dataset: RandomMultiModalDataset, hf_tokenizer: PreTrainedTokenizerBase
):
    """Test sampling with only audio buckets."""
    bucket_config = {
        (0, 5, 1): 1.0,  # Only audio
    }

    limit_mm_per_prompt = {"image": 0, "video": 0, "audio": 2}

    samples = audio_dataset.sample(
        tokenizer=hf_tokenizer,
        num_requests=3,
        base_items_per_request=1,
        num_mm_items_range_ratio=0.0,
        limit_mm_per_prompt=limit_mm_per_prompt,
        bucket_config=bucket_config,
        input_len=20,
        output_len=5,
    )

    assert len(samples) == 3

    for sample in samples:
        mm_data = cast(list[dict[str, Any]], sample.multi_modal_data)
        assert len(mm_data) == 1

        item = mm_data[0]
        assert item["type"] == "input_audio"
        assert item["input_audio"]["format"] == "wav"
