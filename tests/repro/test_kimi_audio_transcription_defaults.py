# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.entrypoints.openai.translations.speech_to_text import _is_kimi_audio_asr_model


def test_is_kimi_audio_asr_model_true() -> None:
    hf_cfg = SimpleNamespace(architectures=["MoonshotKimiaForCausalLM"])
    assert _is_kimi_audio_asr_model(hf_cfg)


def test_is_kimi_audio_asr_model_false() -> None:
    hf_cfg = SimpleNamespace(architectures=["OtherModel"])
    assert not _is_kimi_audio_asr_model(hf_cfg)


def test_is_kimi_audio_asr_model_missing_architectures() -> None:
    hf_cfg = SimpleNamespace()
    assert not _is_kimi_audio_asr_model(hf_cfg)
