# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

from vllm.model_executor.models import funaudiochat


def test_funaudiochat_processing_info_forwards_configured_revisions(monkeypatch):
    feature_extractor = object()
    speech_tokenizer = object()
    feature_loader = Mock(return_value=feature_extractor)
    speech_loader = Mock(return_value=speech_tokenizer)
    monkeypatch.setattr(
        funaudiochat.WhisperFeatureExtractor,
        "from_pretrained",
        feature_loader,
    )
    monkeypatch.setattr(
        funaudiochat.TokenizersBackend,
        "from_pretrained",
        speech_loader,
    )

    info = funaudiochat.FunAudioChatProcessingInfo(
        SimpleNamespace(
            model_config=SimpleNamespace(
                model="example/funaudiochat",
                revision="model-revision",
                tokenizer_revision="tokenizer-revision",
            )
        )
    )

    assert info.feature_extractor is feature_extractor
    assert info.speech_tokenizer is speech_tokenizer
    feature_loader.assert_called_once_with(
        "example/funaudiochat",
        revision="model-revision",
    )
    speech_loader.assert_called_once_with(
        "example/funaudiochat",
        subfolder="speech_tokenizer",
        revision="tokenizer-revision",
    )
