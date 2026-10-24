# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from vllm.model_executor.models.qwen3_asr_realtime import Qwen3ASRRealtimeBuffer


def _buffer(n_window: int) -> Qwen3ASRRealtimeBuffer:
    return Qwen3ASRRealtimeBuffer(
        sampling_rate=16000,
        segment_duration_s=5.0,
        n_window=n_window,
        hop_length=160,
    )


def test_flush_prepends_one_tower_chunk_for_both_windows():
    segment = np.arange(80000, dtype=np.float32)
    extra = np.arange(1600, dtype=np.float32) + 3.0
    for n_window in (50, 100):
        buffer = _buffer(n_window)
        buffer.write_audio(np.concatenate((segment, extra)))
        got = buffer.read_audio()
        assert got is not None
        assert got.shape == (80000,)
        assert np.array_equal(got, segment)
        flushed = buffer.flush()
        assert flushed is not None
        assert flushed.shape == (17600,)
        assert np.array_equal(flushed[:16000], segment[-16000:])
        assert np.array_equal(flushed[16000:], extra)


def test_empty_flush_stays_none_when_a_carry_is_held():
    buffer = _buffer(100)
    buffer.write_audio(np.ones(80000, dtype=np.float32))
    assert buffer.read_audio() is not None
    assert buffer.flush() is None


def test_later_read_audio_is_not_prefixed_with_the_carry():
    buffer = _buffer(100)
    first = np.full(80000, 1.0, dtype=np.float32)
    second = np.full(80000, 2.0, dtype=np.float32)
    buffer.write_audio(np.concatenate((first, second)))
    assert np.all(buffer.read_audio() == 1.0)
    assert np.all(buffer.read_audio() == 2.0)
    assert buffer.flush() is None


def test_only_the_latest_yielded_segment_is_carried():
    buffer = _buffer(100)
    first = np.arange(80000, dtype=np.float32)
    second = np.arange(80000, dtype=np.float32) + 1000.0
    tail = np.arange(100, dtype=np.float32)
    buffer.write_audio(np.concatenate((first, second, tail)))
    buffer.read_audio()
    buffer.read_audio()
    flushed = buffer.flush()
    assert flushed is not None
    assert np.array_equal(flushed[:16000], second[-16000:])
    assert np.array_equal(flushed[16000:], tail)


def test_flush_without_a_prior_segment_returns_the_tail_only():
    buffer = _buffer(50)
    tail = np.arange(100, dtype=np.float32)
    buffer.write_audio(tail)
    flushed = buffer.flush()
    assert flushed is not None
    assert np.array_equal(flushed, tail)
