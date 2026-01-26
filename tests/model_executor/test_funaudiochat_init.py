# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch.nn as nn

from vllm.model_executor.models.interfaces import LMMissingLayer, TowerMissingLayer

pytestmark = pytest.mark.cpu_test


class _DummyTower(nn.Module):
    def __init__(self, *_: object, **__: object) -> None:
        super().__init__()


class _DummyLanguageModel(nn.Module):
    def __init__(self, *_: object, **__: object) -> None:
        super().__init__()

    def make_empty_intermediate_tensors(self, *_: object, **__: object):
        return None


class _DummyMultimodalConfig:
    def __init__(self, *, limit_mm_per_prompt: dict[str, int], mm_encoder_only: bool):
        self._limit_mm_per_prompt = limit_mm_per_prompt
        self.mm_encoder_only = mm_encoder_only

    def get_limit_per_prompt(self, modality: str) -> int:
        return int(self._limit_mm_per_prompt.get(modality, 1))


def _patch_funaudiochat(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm.model_executor.models.funaudiochat as funaudiochat

    monkeypatch.setattr(funaudiochat, "FunAudioChatAudioEncoder", _DummyTower)
    monkeypatch.setattr(funaudiochat, "FunAudioChatDiscreteEncoder", _DummyTower)

    def _fake_init_vllm_registered_model(**_: object):
        return _DummyLanguageModel()

    monkeypatch.setattr(
        funaudiochat, "init_vllm_registered_model", _fake_init_vllm_registered_model
    )


def _make_vllm_config(*, limit_audio: int, mm_encoder_only: bool):
    multimodal_config = _DummyMultimodalConfig(
        limit_mm_per_prompt={"audio": limit_audio},
        mm_encoder_only=mm_encoder_only,
    )
    hf_config = SimpleNamespace(audio_config=object(), text_config=object())
    model_config = SimpleNamespace(
        hf_config=hf_config,
        multimodal_config=multimodal_config,
    )
    return SimpleNamespace(model_config=model_config, quant_config=None)


def test_funaudiochat_audio_disabled_uses_tower_missing_layer(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_funaudiochat(monkeypatch)

    from vllm.model_executor.models.funaudiochat import (
        FunAudioChatForConditionalGeneration,
    )

    model = FunAudioChatForConditionalGeneration(
        vllm_config=_make_vllm_config(limit_audio=0, mm_encoder_only=False)
    )

    assert isinstance(model.continuous_audio_tower, TowerMissingLayer)
    assert isinstance(model.audio_tower, TowerMissingLayer)
    assert isinstance(model.language_model, _DummyLanguageModel)


def test_funaudiochat_mm_encoder_only_uses_lm_missing_layer(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_funaudiochat(monkeypatch)

    from vllm.model_executor.models.funaudiochat import (
        FunAudioChatForConditionalGeneration,
    )

    model = FunAudioChatForConditionalGeneration(
        vllm_config=_make_vllm_config(limit_audio=1, mm_encoder_only=True)
    )

    assert isinstance(model.language_model, LMMissingLayer)
