# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``MistralCommonFeatureExtractor.fetch_audio``.

``transformers>=5.10`` adds a ``ProcessorMixin.prepare_inputs_layout`` helper
that calls ``self.feature_extractor.fetch_audio(...)`` unconditionally. The
duck-typed :class:`MistralCommonFeatureExtractor` previously did not implement
that method, so loading any voxtral model under transformers 5.10.x raised
``AttributeError: 'MistralCommonFeatureExtractor' object has no attribute
'fetch_audio'``. These tests pin the new ``fetch_audio`` method to the same
contract as ``transformers.SequenceFeatureExtractor.fetch_audio``.
"""

import numpy as np
import pytest
import torch

from vllm.tokenizers.mistral import MistralTokenizer
from vllm.transformers_utils.processors.voxtral import (
    MistralCommonFeatureExtractor,
)


@pytest.fixture(scope="module")
def feature_extractor() -> MistralCommonFeatureExtractor:
    tokenizer = MistralTokenizer.from_pretrained("mistralai/Voxtral-Mini-3B-2507")
    return MistralCommonFeatureExtractor(tokenizer.instruct.audio_encoder)


@pytest.mark.parametrize(
    "audio",
    [
        np.zeros(1024, dtype=np.float32),
        torch.zeros(1024),
        [0.0, 1.0, 2.0],
    ],
    ids=["numpy_array", "torch_tensor", "list_of_floats"],
)
def test_fetch_audio_passes_through(
    feature_extractor: MistralCommonFeatureExtractor, audio
):
    result = feature_extractor.fetch_audio(audio)
    assert result is audio


def test_fetch_audio_recurses_over_list_of_arrays(
    feature_extractor: MistralCommonFeatureExtractor,
):
    a = np.zeros(8, dtype=np.float32)
    b = np.ones(8, dtype=np.float32)
    result = feature_extractor.fetch_audio([a, b])
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] is a
    assert result[1] is b


def test_fetch_audio_uses_self_sampling_rate_when_none(
    monkeypatch, feature_extractor: MistralCommonFeatureExtractor
):
    """If ``sampling_rate`` is None, ``self.sampling_rate`` must be used.

    Verified indirectly via the recursion path: when we pass a list of arrays
    without sampling_rate, recursive calls receive the resolved rate.
    """
    captured: list[int | None] = []
    original = feature_extractor.fetch_audio

    def spy(audio, sampling_rate=None):
        captured.append(sampling_rate)
        return original(audio, sampling_rate=sampling_rate)

    monkeypatch.setattr(feature_extractor, "fetch_audio", spy)
    feature_extractor.fetch_audio([np.zeros(4, dtype=np.float32)])
    # Top-level call has sampling_rate=None; inner recursive call sees the
    # resolved rate from self.sampling_rate.
    assert captured[0] is None
    assert captured[1] == 16000


def test_fetch_audio_explicit_sampling_rate_propagates(
    monkeypatch, feature_extractor: MistralCommonFeatureExtractor
):
    captured: list[int | None] = []
    original = feature_extractor.fetch_audio

    def spy(audio, sampling_rate=None):
        captured.append(sampling_rate)
        return original(audio, sampling_rate=sampling_rate)

    monkeypatch.setattr(feature_extractor, "fetch_audio", spy)
    feature_extractor.fetch_audio([np.zeros(4, dtype=np.float32)], sampling_rate=8000)
    assert captured[0] == 8000
    assert captured[1] == 8000


def test_fetch_audio_rejects_unsupported_type(
    feature_extractor: MistralCommonFeatureExtractor,
):
    with pytest.raises(TypeError, match="only a numpy array"):
        feature_extractor.fetch_audio(42)  # type: ignore[arg-type]


def test_fetch_audio_str_delegates_to_load_audio(
    monkeypatch, feature_extractor: MistralCommonFeatureExtractor
):
    """A str input must round-trip through ``transformers.audio_utils.load_audio``.

    We monkey-patch ``load_audio`` so the test stays offline (no real URL/path
    fetched) and still asserts the delegation contract.
    """
    sentinel = np.array([0.5, -0.5], dtype=np.float32)
    received: dict[str, object] = {}

    def fake_load_audio(path, sampling_rate=None):
        received["path"] = path
        received["sampling_rate"] = sampling_rate
        return sentinel

    import transformers.audio_utils

    monkeypatch.setattr(transformers.audio_utils, "load_audio", fake_load_audio)

    result = feature_extractor.fetch_audio("/tmp/fake.wav")
    assert result is sentinel
    assert received["path"] == "/tmp/fake.wav"
    assert received["sampling_rate"] == 16000
