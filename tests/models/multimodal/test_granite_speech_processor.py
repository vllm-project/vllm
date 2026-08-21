# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import torch

from vllm.model_executor.models.granite_speech import (
    GraniteSpeechMultiModalProcessor,
)
from vllm.multimodal.processing import BaseMultiModalProcessor


class _FakeAudioProcessor:
    def _get_num_audio_features(self, audio_lengths: list[int]) -> list[int]:
        return [length // 100 for length in audio_lengths]


class _FakeProcessor:
    audio_processor = _FakeAudioProcessor()


class _FakeInfo:
    def get_hf_processor(self, **kwargs):
        return _FakeProcessor()


def test_granite_speech_uses_audio_lengths_for_embed_sizes(monkeypatch):
    def fake_call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        return {"input_ids": torch.tensor([[1, 2, 100352, 3]])}

    monkeypatch.setattr(
        BaseMultiModalProcessor,
        "_call_hf_processor",
        fake_call_hf_processor,
    )

    processor = GraniteSpeechMultiModalProcessor.__new__(
        GraniteSpeechMultiModalProcessor
    )
    processor.info = _FakeInfo()

    outputs = processor._call_hf_processor(
        prompt="<|audio|>transcribe",
        mm_data={"audios": [np.zeros(32000), np.zeros(48000)]},
        mm_kwargs={},
        tok_kwargs={},
    )

    assert outputs["audio_embed_sizes"].tolist() == [320, 480]


def test_granite_speech_lets_vllm_apply_prompt_updates():
    processor = GraniteSpeechMultiModalProcessor.__new__(
        GraniteSpeechMultiModalProcessor
    )

    assert not processor._hf_processor_applies_updates(
        prompt_text="<|audio|>transcribe",
        mm_items=SimpleNamespace(),
        hf_processor_mm_kwargs={},
        tokenization_kwargs={},
    )
