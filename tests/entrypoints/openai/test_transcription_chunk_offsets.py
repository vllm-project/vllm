# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.entrypoints.openai.speech_to_text.speech_to_text import OpenAISpeechToText


def _make_speech_to_text_for_chunking() -> OpenAISpeechToText:
    stt = OpenAISpeechToText.__new__(OpenAISpeechToText)
    stt.asr_config = SimpleNamespace(
        max_audio_clip_s=3,
        overlap_chunk_second=1,
        min_energy_split_window_size=1,
    )
    return stt


def test_split_audio_with_start_offsets_tracks_true_boundaries() -> None:
    stt = _make_speech_to_text_for_chunking()
    sample_rate = 10

    # Force split points inside overlap windows so they differ from fixed
    # chunk_size multiples and expose timestamp drift if offsets are ignored.
    audio = np.ones(70, dtype=np.float32)
    audio[23] = 0.0
    audio[47] = 0.0

    chunks, starts = stt._split_audio_with_start_offsets(audio, sample_rate)

    assert starts == pytest.approx([0.0, 2.3, 4.7])
    assert [len(chunk) for chunk in chunks] == [23, 24, 23]


def test_split_audio_backward_compat_returns_chunks_only() -> None:
    stt = _make_speech_to_text_for_chunking()
    sample_rate = 10
    audio = np.ones(70, dtype=np.float32)
    audio[23] = 0.0
    audio[47] = 0.0

    chunks = stt._split_audio(audio, sample_rate)

    assert [len(chunk) for chunk in chunks] == [23, 24, 23]


def test_find_split_point_fallbacks_to_search_start_when_window_too_small() -> None:
    stt = _make_speech_to_text_for_chunking()
    # Make the min-energy window larger than the overlap search segment so
    # no loop iteration runs in _find_split_point.
    stt.asr_config.min_energy_split_window_size = 50

    wav = np.ones(100, dtype=np.float32)
    start_idx = 70
    end_idx = 75

    split_point = stt._find_split_point(wav, start_idx, end_idx)

    assert split_point == start_idx
