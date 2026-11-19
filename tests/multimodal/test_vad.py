# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm.config.speech_to_text import SpeechToTextConfig, VADConfig
from vllm.multimodal.vad import VAD, split_audio_with_vad


@pytest.mark.cpu_test
@pytest.mark.parametrize("max_audio_clip_s", [30, None])
def test_split_audio_with_vad_returns_no_chunks_without_speech(max_audio_clip_s):
    pytest.importorskip("silero_vad")
    sample_rate = 16000
    silence = np.zeros(sample_rate * 5, dtype=np.float32)
    asr_config = SpeechToTextConfig(max_audio_clip_s=max_audio_clip_s)

    chunks = split_audio_with_vad(
        duration=5.0,
        asr_config=asr_config,
        vad=VAD(),
        vad_config=VADConfig(enabled=True),
        audio_data=silence,
        sample_rate=sample_rate,
    )

    assert chunks == []


@pytest.mark.cpu_test
@pytest.mark.parametrize("max_audio_clip_s", [30, None])
def test_split_audio_without_vad_keeps_silent_audio(max_audio_clip_s):
    sample_rate = 16000
    silence = np.zeros(sample_rate * 5, dtype=np.float32)
    asr_config = SpeechToTextConfig(max_audio_clip_s=max_audio_clip_s)

    chunks = split_audio_with_vad(
        duration=5.0,
        asr_config=asr_config,
        vad=None,
        vad_config=VADConfig(enabled=False),
        audio_data=silence,
        sample_rate=sample_rate,
    )

    assert len(chunks) == 1
    np.testing.assert_array_equal(chunks[0], silence)


def _fake_vad(speech_spans_s: list[tuple[float, float]], sample_rate: int):
    vad = MagicMock(spec=VAD)
    vad.get_speech_timestamps.return_value = [
        {"start": int(start * sample_rate), "end": int(end * sample_rate)}
        for start, end in speech_spans_s
    ]
    return vad


def _join_spans(
    audio: np.ndarray, spans_s: list[tuple[float, float]], sample_rate: int
) -> np.ndarray:
    return np.concatenate(
        [
            audio[int(start * sample_rate) : int(end * sample_rate)]
            for start, end in spans_s
        ]
    )


@pytest.mark.cpu_test
def test_split_audio_with_vad_splits_and_packs_speech():
    """VAD spans longer than max_audio_clip_s are RMS split, and the resulting
    pieces are packed into chunks of up to max_audio_clip_s with the
    non-speech removed."""
    sample_rate = 16000
    audio = np.random.default_rng(0).standard_normal(100 * sample_rate)
    audio = audio.astype(np.float32)
    # The 20-60 s span is RMS split at its quietest 0.1 s window between 29 s
    # and 30 s into the span, i.e. at 49.5 s.
    audio[int(49.5 * sample_rate) : int(49.6 * sample_rate)] = 0.0
    speech_spans_s = [(2.0, 5.0), (10.0, 14.0), (20.0, 60.0), (70.0, 80.0)]

    chunks = split_audio_with_vad(
        duration=100.0,
        asr_config=SpeechToTextConfig(
            max_audio_clip_s=30,
            overlap_chunk_second=1,
            min_energy_split_window_size=1600,
        ),
        vad=_fake_vad(speech_spans_s, sample_rate),
        vad_config=VADConfig(enabled=True),
        audio_data=audio,
        sample_rate=sample_rate,
    )

    # Pieces of 3, 4, 29.5, 10.5 and 10 s: 3 + 4 fit, adding 29.5 doesn't;
    # 29.5 + 10.5 doesn't; the RMS remainder 10.5 + 10 does.
    expected_chunks_s = [
        [(2.0, 5.0), (10.0, 14.0)],
        [(20.0, 49.5)],
        [(49.5, 60.0), (70.0, 80.0)],
    ]
    assert len(chunks) == len(expected_chunks_s)
    for chunk, spans_s in zip(chunks, expected_chunks_s):
        np.testing.assert_array_equal(chunk, _join_spans(audio, spans_s, sample_rate))


@pytest.mark.cpu_test
def test_split_audio_with_vad_chunk_count_is_bounded_by_speech():
    """Many short speech segments, e.g. from extreme VAD parameters, must not
    fan out into one engine request each."""
    sample_rate = 16000
    duration_s = 600
    audio = np.zeros(duration_s * sample_rate, dtype=np.float32)
    # 0.1 s of speech every 0.3 s: 2000 segments, 200 s of speech in total.
    speech_spans_s = [(t * 0.3, t * 0.3 + 0.1) for t in range(2000)]

    chunks = split_audio_with_vad(
        duration=float(duration_s),
        asr_config=SpeechToTextConfig(max_audio_clip_s=30),
        vad=_fake_vad(speech_spans_s, sample_rate),
        vad_config=VADConfig(enabled=True),
        audio_data=audio,
        sample_rate=sample_rate,
    )

    assert len(chunks) == math.ceil(200 / 30)
    assert all(chunk.shape[-1] <= 30 * sample_rate for chunk in chunks)


@pytest.mark.cpu_test
def test_split_audio_without_vad_covers_the_whole_audio():
    sample_rate = 16000
    audio = np.random.default_rng(0).standard_normal(65 * sample_rate)
    audio = audio.astype(np.float32)

    chunks = split_audio_with_vad(
        duration=65.0,
        asr_config=SpeechToTextConfig(max_audio_clip_s=30),
        vad=None,
        vad_config=VADConfig(enabled=False),
        audio_data=audio,
        sample_rate=sample_rate,
    )

    assert len(chunks) == 3
    np.testing.assert_array_equal(np.concatenate(chunks), audio)


@pytest.mark.cpu_test
def test_split_audio_with_vad_only_starts_at_first_speech():
    sample_rate = 16000
    audio = np.random.default_rng(0).standard_normal(30 * sample_rate)
    audio = audio.astype(np.float32)
    speech_spans_s = [(5.0, 10.0), (20.0, 25.0)]

    chunks = split_audio_with_vad(
        duration=30.0,
        asr_config=SpeechToTextConfig(max_audio_clip_s=None),
        vad=_fake_vad(speech_spans_s, sample_rate),
        vad_config=VADConfig(enabled=True),
        audio_data=audio,
        sample_rate=sample_rate,
    )

    assert len(chunks) == 1
    np.testing.assert_array_equal(
        chunks[0], _join_spans(audio, speech_spans_s, sample_rate)
    )
