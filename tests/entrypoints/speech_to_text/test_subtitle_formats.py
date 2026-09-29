# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.entrypoints.speech_to_text.base.subtitle_formats import (
    _format_timestamp,
    segments_to_srt,
    segments_to_vtt,
)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00:00,000"),
        # float truncation would give ,539
        (11.54, "00:00:11,540"),
        # rounding up must carry into seconds, never emit ,1000
        (0.9995, "00:00:01,000"),
        (59.9996, "00:01:00,000"),
        (3599.9996, "01:00:00,000"),
        (3661.2, "01:01:01,200"),
    ],
)
def test_format_timestamp(seconds: float, expected: str):
    assert _format_timestamp(seconds, ",") == expected
    assert _format_timestamp(seconds, ".") == expected.replace(",", ".")


def test_segments_to_srt_and_vtt():
    segments = [
        SimpleNamespace(start=0.0, end=11.54, text=" Mary had a little lamb "),
        SimpleNamespace(start=11.54, end=17.66, text="its fleece was white"),
    ]
    assert segments_to_srt(segments) == (
        "1\n00:00:00,000 --> 00:00:11,540\nMary had a little lamb\n\n"
        "2\n00:00:11,540 --> 00:00:17,660\nits fleece was white\n"
    )
    assert segments_to_vtt(segments) == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:11.540\nMary had a little lamb\n\n"
        "00:00:11.540 --> 00:00:17.660\nits fleece was white\n"
    )
    assert segments_to_srt([]) == ""
    assert segments_to_vtt([]) == "WEBVTT\n"


def test_cue_text_sanitised():
    segments = [SimpleNamespace(start=0.0, end=1.0, text=" R&D <3 a --> b ")]
    assert segments_to_srt(segments).splitlines()[2] == "R&D <3 a -> b"
    assert segments_to_vtt(segments).splitlines()[3] == "R&amp;D &lt;3 a -> b"
