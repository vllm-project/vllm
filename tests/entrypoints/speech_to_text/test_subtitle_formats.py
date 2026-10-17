# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SRT/VTT subtitle rendering of speech-to-text segments."""

from types import SimpleNamespace

from vllm.entrypoints.speech_to_text.base.utils import (
    format_segments_as_srt,
    format_segments_as_vtt,
)


def _segment(start: float, end: float, text: str) -> SimpleNamespace:
    return SimpleNamespace(start=start, end=end, text=text)


def test_srt_basic():
    segments = [
        _segment(0.0, 2.5, " Hello, this is a transcription"),
        _segment(2.5, 5.42, "of the audio file."),
    ]
    assert format_segments_as_srt(segments) == (
        "1\n"
        "00:00:00,000 --> 00:00:02,500\n"
        "Hello, this is a transcription\n"
        "\n"
        "2\n"
        "00:00:02,500 --> 00:00:05,420\n"
        "of the audio file.\n"
    )


def test_vtt_basic():
    segments = [
        _segment(0.0, 2.5, "Hello, this is a transcription"),
        _segment(2.5, 5.42, "of the audio file."),
    ]
    assert format_segments_as_vtt(segments) == (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:02.500\n"
        "Hello, this is a transcription\n"
        "\n"
        "00:00:02.500 --> 00:00:05.420\n"
        "of the audio file.\n"
    )


def test_timestamp_rollover_to_hours():
    segments = [_segment(3671.5, 7322.25, "long audio")]
    assert format_segments_as_srt(segments) == (
        "1\n01:01:11,500 --> 02:02:02,250\nlong audio\n"
    )
    assert format_segments_as_vtt(segments) == (
        "WEBVTT\n\n01:01:11.500 --> 02:02:02.250\nlong audio\n"
    )


def test_empty_text_segments_are_skipped_and_renumbered():
    segments = [
        _segment(0.0, 1.0, "hello"),
        _segment(1.0, 2.0, "   "),
        _segment(2.0, 3.0, "world"),
    ]
    assert format_segments_as_srt(segments) == (
        "1\n00:00:00,000 --> 00:00:01,000\nhello\n"
        "\n"
        "2\n00:00:02,000 --> 00:00:03,000\nworld\n"
    )


def test_no_segments():
    assert format_segments_as_srt([]) == ""
    assert format_segments_as_vtt([]) == "WEBVTT\n"


def test_zero_length_segment():
    segments = [_segment(0.0, 0.0, "x")]
    assert format_segments_as_srt(segments) == ("1\n00:00:00,000 --> 00:00:00,000\nx\n")
