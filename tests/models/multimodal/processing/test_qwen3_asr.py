# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-ASR preprocessing must preserve its HF processor's audio inputs."""

import numpy as np
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_asr import Qwen3ASRProcessor

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.inputs import batched_tensors_equal
from vllm.transformers_utils.processor import cached_processor_from_config

from .test_common import _assert_inputs_equal

_AUDIO_PROMPT = "<|audio_start|><|audio_pad|><|audio_end|>"


@pytest.fixture(params=["Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-0.6B-hf"])
def asr_processor(request):
    config = ModelConfig(
        request.param,
        dtype="float32",
        max_model_len=4096,
        limit_mm_per_prompt={"audio": 2},
        mm_processor_cache_gb=0,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(config)
    native = Qwen3ASRProcessor.from_pretrained(request.param)
    return processor, native


@pytest.mark.parametrize(
    "samples", [321, 15999, 16000, 16001, 16159, 16160, 16161, 32001, 480001]
)
@pytest.mark.parametrize("num_audios", [1, 2])
def test_audio_processing_matches_hf(asr_processor, samples, num_audios):
    """Each clip must match HF independently of other clips in the request."""
    processor, native = asr_processor
    rng = np.random.RandomState(42)
    audios = [
        rng.randn(samples + index * 160).astype(np.float32)
        for index in range(num_audios)
    ]
    expected = [
        native(text=_AUDIO_PROMPT, audio=audio, return_tensors="pt") for audio in audios
    ]
    actual = processor(
        _AUDIO_PROMPT * num_audios,
        mm_items=processor.info.parse_mm_data({"audio": audios}),
        hf_processor_mm_kwargs={},
    )

    expected_ids = [
        token for item in expected for token in item["input_ids"][0].tolist()
    ]
    assert actual["prompt_token_ids"] == expected_ids
    masks = [item["input_features_mask"][0] for item in expected]
    expected_features = torch.cat(
        [
            item["input_features"][0, :, mask.bool()]
            for item, mask in zip(expected, masks)
        ],
        dim=1,
    )
    actual_features = actual["mm_kwargs"].get_data()
    torch.testing.assert_close(
        actual_features["input_audio_features"], expected_features, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_features["audio_feature_lengths"],
        torch.stack([mask.sum() for mask in masks]),
    )


@pytest.mark.parametrize("configured", [False, True])
def test_nested_audio_kwargs(asr_processor, configured):
    """Dummy-text defaults must not conflict with explicit audio truncation."""
    processor, native = asr_processor
    audio = np.random.RandomState(42).randn(32001).astype(np.float32)
    kwargs = {"audio_kwargs": {"truncation": True, "max_length": 16000}}
    expected = native(text=_AUDIO_PROMPT, audio=audio, return_tensors="pt", **kwargs)
    if configured:
        processor.info.ctx.get_mm_config().mm_processor_kwargs = kwargs
    actual = processor(
        _AUDIO_PROMPT,
        mm_items=processor.info.parse_mm_data({"audio": audio}),
        hf_processor_mm_kwargs={} if configured else kwargs,
    )

    assert kwargs == {"audio_kwargs": {"truncation": True, "max_length": 16000}}
    assert actual["prompt_token_ids"] == expected["input_ids"][0].tolist()
    mask = expected["input_features_mask"][0]
    actual_features = actual["mm_kwargs"].get_data()
    torch.testing.assert_close(
        actual_features["input_audio_features"],
        expected["input_features"][0, :, mask.bool()],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_features["audio_feature_lengths"], mask.sum().reshape(1)
    )


@pytest.mark.parametrize("cached_indices", [[], [0], [1], [0, 1]])
def test_audio_processing_cache_is_batch_independent(asr_processor, cached_indices):
    """Partial cache hits must not change features, masks, or prompt expansion."""
    processor, _ = asr_processor
    config = processor.info.ctx.model_config
    config.get_multimodal_config().mm_processor_cache_gb = 1
    cache = MultiModalProcessorOnlyCache(config)
    rng = np.random.RandomState(42)
    audios = [rng.randn(samples).astype(np.float32) for samples in (16001, 32000)]
    for index in cached_indices:
        processor(
            _AUDIO_PROMPT,
            mm_items=processor.info.parse_mm_data({"audio": audios[index]}),
            hf_processor_mm_kwargs={},
            cache=cache,
        )

    for batch in (audios, audios[::-1]):
        kwargs = dict(
            mm_items=processor.info.parse_mm_data({"audio": batch}),
            hf_processor_mm_kwargs={},
        )
        _assert_inputs_equal(
            processor(_AUDIO_PROMPT * 2, **kwargs),
            processor(_AUDIO_PROMPT * 2, **kwargs, cache=cache),
        )


def test_text_only_processing(asr_processor):
    """Empty media input must not invoke the audio feature extractor."""
    processor, native = asr_processor
    prompt = "Transcribe the audio."
    actual = processor(
        prompt,
        mm_items=processor.info.parse_mm_data({}),
        hf_processor_mm_kwargs={},
    )
    assert actual["prompt_token_ids"] == native.tokenizer.encode(prompt)
    assert actual["mm_kwargs"].get_data() == {}


@pytest.mark.parametrize("hf_data", [{}, {"text": ""}, {"audio": None}, {"audio": []}])
def test_hf_processor_without_audio(asr_processor, hf_data):
    """The media hook must not require audio or call HF for absent waveforms."""
    processor, _ = asr_processor
    assert not processor._call_hf_processor(hf_data, {})


@pytest.mark.parametrize("hf_data", [{}, {"text": ""}])
def test_postprocessing_without_audio_features(asr_processor, hf_data):
    """Non-audio output must pass through even when HF input is nonempty."""
    processor, _ = asr_processor
    processed = BatchFeature({"input_ids": torch.tensor([[1, 2]])})
    actual = processor._postprocess_hf_mm_data(hf_data, {}, processed)
    assert actual is processed
    assert set(actual) == {"input_ids"}
    torch.testing.assert_close(actual["input_ids"], torch.tensor([[1, 2]]))


def test_chat_template_processing(asr_processor):
    """The generic processor loader must still render audio chat prompts."""
    processor, native = asr_processor
    chat_processor = cached_processor_from_config(processor.info.ctx.model_config)
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": "unused"}]},
    ]
    prompt = chat_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    expected_prompt = native.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    assert prompt == expected_prompt
    audio = np.random.RandomState(42).randn(16001).astype(np.float32)
    expected = native(text=expected_prompt, audio=audio, return_tensors="pt")
    actual = processor(
        prompt,
        mm_items=processor.info.parse_mm_data({"audio": audio}),
        hf_processor_mm_kwargs={},
    )
    assert actual["prompt_token_ids"] == expected["input_ids"][0].tolist()


def test_precomputed_audio_features(asr_processor):
    """Precomputed features must bypass waveform extraction unchanged."""
    processor, _ = asr_processor
    processor.info.ctx.get_mm_config().enable_mm_embeds = True
    audio = np.random.RandomState(42).randn(16001).astype(np.float32)
    expected = processor(
        _AUDIO_PROMPT,
        mm_items=processor.info.parse_mm_data({"audio": audio}),
        hf_processor_mm_kwargs={},
    )
    features = expected["mm_kwargs"].get_data()
    actual = processor(
        _AUDIO_PROMPT,
        mm_items=processor.info.parse_mm_data({"audio": features}),
        hf_processor_mm_kwargs={},
    )
    assert actual["prompt_token_ids"] == expected["prompt_token_ids"]
    assert actual["mm_placeholders"] == expected["mm_placeholders"]
    assert batched_tensors_equal(actual["mm_kwargs"].get_data(), features)
