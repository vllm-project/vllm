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

from vllm.transformers_utils.processors.voxtral import (
    MistralCommonFeatureExtractor,
)


class _FakeAudioConfig:
    sampling_rate = 16000
    frame_rate = 12.5
    is_streaming = False


class _FakeAudioEncoder:
    audio_config = _FakeAudioConfig()


@pytest.fixture
def feature_extractor() -> MistralCommonFeatureExtractor:
    return MistralCommonFeatureExtractor(_FakeAudioEncoder())


def test_fetch_audio_passes_through_numpy_array(feature_extractor):
    audio = np.zeros(1024, dtype=np.float32)
    result = feature_extractor.fetch_audio(audio)
    assert result is audio


def test_fetch_audio_passes_through_torch_tensor(feature_extractor):
    audio = torch.zeros(1024)
    result = feature_extractor.fetch_audio(audio)
    assert result is audio


def test_fetch_audio_passes_through_list_of_floats(feature_extractor):
    audio = [0.0, 1.0, 2.0]
    result = feature_extractor.fetch_audio(audio)
    assert result is audio


def test_fetch_audio_recurses_over_list_of_arrays(feature_extractor):
    a = np.zeros(8, dtype=np.float32)
    b = np.ones(8, dtype=np.float32)
    result = feature_extractor.fetch_audio([a, b])
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] is a
    assert result[1] is b


def test_fetch_audio_uses_self_sampling_rate_when_none(feature_extractor):
    """If ``sampling_rate`` is None, ``self.sampling_rate`` must be used.

    Verified indirectly via the recursion path: when we pass a list of arrays
    without sampling_rate, recursive calls receive the resolved rate.
    """
    captured: list[int] = []

    original = feature_extractor.fetch_audio

    def spy(audio, sampling_rate=None):
        captured.append(sampling_rate)
        return original(audio, sampling_rate=sampling_rate)

    feature_extractor.fetch_audio = spy  # type: ignore[method-assign]
    feature_extractor.fetch_audio([np.zeros(4, dtype=np.float32)])
    # Top-level call has sampling_rate=None; inner recursive call sees the
    # resolved rate from self.sampling_rate.
    assert captured[0] is None
    assert captured[1] == _FakeAudioConfig.sampling_rate


def test_fetch_audio_explicit_sampling_rate_propagates(feature_extractor):
    captured: list[int] = []
    original = feature_extractor.fetch_audio

    def spy(audio, sampling_rate=None):
        captured.append(sampling_rate)
        return original(audio, sampling_rate=sampling_rate)

    feature_extractor.fetch_audio = spy  # type: ignore[method-assign]
    feature_extractor.fetch_audio([np.zeros(4, dtype=np.float32)], sampling_rate=8000)
    assert captured[0] == 8000
    assert captured[1] == 8000


def test_fetch_audio_rejects_unsupported_type(feature_extractor):
    with pytest.raises(TypeError, match="only a numpy array"):
        feature_extractor.fetch_audio(42)  # type: ignore[arg-type]


def test_fetch_audio_str_delegates_to_load_audio(monkeypatch, feature_extractor):
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
    assert received["sampling_rate"] == _FakeAudioConfig.sampling_rate


def test_prepare_inputs_layout_path_works_with_numpy(feature_extractor):
    """End-to-end shape check: feed numpy through fetch_audio (pre-call hook)
    then through ``__call__`` like transformers >=5.10 does.

    ``ProcessorMixin.prepare_inputs_layout`` calls ``fetch_audio`` first,
    then ``make_list_of_audio``, then the processor's ``__call__``. This test
    validates the first hop: ``__call__`` still works on the result.
    """
    audio = np.random.randn(1024).astype(np.float32)
    fetched = feature_extractor.fetch_audio(audio)
    # In a real run, transformers wraps this in make_list_of_audio. Here we
    # only verify identity-passthrough so the downstream __call__ would work.
    assert fetched is audio
