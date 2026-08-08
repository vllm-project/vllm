# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

from vllm.model_executor.models import funaudiochat


def test_funaudiochat_processing_info_forwards_configured_revisions(monkeypatch):
    feature_loader = Mock()
    speech_loader = Mock()
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

    info.get_feature_extractor()
    info.get_speech_tokenizer()
    feature_loader.assert_called_once_with(
        "example/funaudiochat",
        revision="model-revision",
    )
    speech_loader.assert_called_once_with(
        "example/funaudiochat",
        subfolder="speech_tokenizer",
        revision="tokenizer-revision",
    )
