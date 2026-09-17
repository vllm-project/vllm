# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.model_executor.models.funaudiochat import (
    FunAudioChatDummyInputsBuilder,
    FunAudioChatMultiModalProcessor,
    FunAudioChatProcessingInfo,
)
from vllm.transformers_utils.configs.funaudiochat import FunAudioChatConfig


class RecordingFeatureExtractor:
    sampling_rate = 16000
    n_fft = 400
    hop_length = 160

    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] | None = None

    def __call__(self, wavs, **kwargs):
        self.last_kwargs = kwargs
        n_frames = [max(1, int(w.shape[0]) // self.hop_length) for w in wavs]
        max_frames = max(n_frames)
        features = torch.zeros((len(wavs), 128, max_frames))
        mask = torch.zeros((len(wavs), max_frames), dtype=torch.long)
        for i, frames in enumerate(n_frames):
            mask[i, :frames] = 1
        return {"input_features": features, "attention_mask": mask}


class MockSpeechTokenizer:
    pad_token = "<|audio_pad|>"

    def __call__(self, speech_strs, **kwargs):
        token_len = max(1, len(self.pad_token))
        lengths = [len(s) // token_len for s in speech_strs]
        max_len = max(lengths)
        pad_to = kwargs.get("pad_to_multiple_of")
        if pad_to:
            max_len = ((max_len + pad_to - 1) // pad_to) * pad_to
        ids = torch.zeros((len(speech_strs), max_len), dtype=torch.long)
        mask = torch.zeros((len(speech_strs), max_len), dtype=torch.long)
        for i, n in enumerate(lengths):
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask}


class MockTokenizer:
    def encode(self, text, **kwargs):
        return [1, 2, 3]


@pytest.fixture
def funaudiochat_processor(monkeypatch):
    feature_extractor = RecordingFeatureExtractor()
    speech_tokenizer = MockSpeechTokenizer()
    tokenizer = MockTokenizer()

    monkeypatch.setattr(
        "vllm.model_executor.models.funaudiochat.WhisperFeatureExtractor.from_pretrained",
        lambda *args, **kwargs: feature_extractor,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.funaudiochat.TokenizersBackend.from_pretrained",
        lambda *args, **kwargs: speech_tokenizer,
    )

    ctx = MagicMock()
    ctx.get_hf_config.return_value = FunAudioChatConfig()
    ctx.model_config.hf_config = FunAudioChatConfig()
    ctx.model_config.model = "funaudiochat"
    ctx.model_config.revision = None
    ctx.model_config.tokenizer_revision = None
    mm_config = MagicMock()
    mm_config.enable_mm_embeds = False
    mm_config.allow_missing_mm_embeddings = False
    ctx.model_config.multimodal_config = mm_config
    ctx.model_config.get_multimodal_config.return_value = mm_config
    ctx.tokenizer = tokenizer
    ctx.get_tokenizer.return_value = tokenizer

    info = FunAudioChatProcessingInfo(ctx)
    processor = FunAudioChatMultiModalProcessor(
        info, FunAudioChatDummyInputsBuilder(info)
    )
    return processor, feature_extractor


def test_max_audio_samples_match_profiled_300s_budget(funaudiochat_processor):
    processor, _ = funaudiochat_processor
    sampling_rate = 16000

    assert processor.info.get_max_speech_frames() == 7500
    assert processor.info.get_max_audio_samples(sampling_rate) == 4_800_000
    assert processor.info.get_max_audio_samples(sampling_rate) / sampling_rate == 300.0


def test_over_budget_audio_is_rejected(funaudiochat_processor):
    processor, feature_extractor = funaudiochat_processor
    sampling_rate = 16000
    over_budget = np.zeros(
        processor.info.get_max_audio_samples(sampling_rate) + 1, dtype=np.float32
    )
    mm_items = processor.info.parse_mm_data({"audio": [over_budget]}, validate=False)

    with pytest.raises(ValueError, match="too long for FunAudioChat"):
        processor._apply_hf_processor_main(mm_items, {})

    assert feature_extractor.last_kwargs is None


def test_profiled_max_audio_is_accepted(funaudiochat_processor):
    processor, feature_extractor = funaudiochat_processor
    sampling_rate = 16000
    max_samples = processor.info.get_max_audio_samples(sampling_rate)
    audio = np.zeros(max_samples, dtype=np.float32)
    mm_items = processor.info.parse_mm_data({"audio": [audio]}, validate=False)

    processed = processor._apply_hf_processor_main(mm_items, {})

    assert processed["speech_attention_mask"].sum().item() == 7500
    assert feature_extractor.last_kwargs is not None
    assert feature_extractor.last_kwargs.get("padding") is True
    assert feature_extractor.last_kwargs.get("padding") != "max_length"


def test_short_audio_does_not_pad_features_to_max_length(funaudiochat_processor):
    processor, feature_extractor = funaudiochat_processor
    sampling_rate = 16000
    one_second = np.zeros(sampling_rate, dtype=np.float32)
    mm_items = processor.info.parse_mm_data(
        {"audio": [one_second, one_second]}, validate=False
    )

    processed = processor._apply_hf_processor_main(mm_items, {})

    assert feature_extractor.last_kwargs is not None
    assert feature_extractor.last_kwargs.get("padding") is True
    # hop_length=160 → 100 frames per 1s clip, not Whisper's 30000-frame max.
    assert processed["input_features"].shape[-1] == 100
    assert processed["input_features"].nbytes < 1_000_000
    assert processed["speech_attention_mask"].sum(dim=-1).tolist() == [25, 25]
