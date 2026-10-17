# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models.moss_transcribe_diarize import (
    MossTranscribeDiarizeForConditionalGeneration,
)


def test_parse_diarized_transcript_preserves_moss_segments():
    segments = MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(
        "[0.48][S01]Welcome[1.66][12.26][S02]Ready[13.81]"
    )

    assert [
        (segment.start, segment.end, segment.speaker, segment.text)
        for segment in segments
    ] == [
        (0.48, 1.66, "S01", "Welcome"),
        (12.26, 13.81, "S02", "Ready"),
    ]


def test_parse_diarized_transcript_preserves_overlapping_segments():
    segments = MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(
        "[0][S01]First speaker[2][1][S02]Second speaker[3]"
    )

    assert [
        (segment.start, segment.end, segment.speaker, segment.text)
        for segment in segments
    ] == [
        (0.0, 2.0, "S01", "First speaker"),
        (1.0, 3.0, "S02", "Second speaker"),
    ]


def test_parse_diarized_transcript_preserves_numeric_text_markers():
    segments = MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(
        "[0][S01]The [2024] report is ready.[4]"
    )

    assert [segment.text for segment in segments] == ["The [2024] report is ready."]


def test_parse_diarized_transcript_ignores_whitespace_between_segments():
    segments = MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(
        "[0][S01]Hello[1]\n [2][S02]Hi[3]"
    )

    assert [(segment.start, segment.end, segment.text) for segment in segments] == [
        (0.0, 1.0, "Hello"),
        (2.0, 3.0, "Hi"),
    ]


def test_parse_diarized_transcript_ignores_noise_before_a_segment():
    segments = MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(
        "noise [bad][0.1][S01]Hello[0.9]"
    )

    assert [(segment.start, segment.end, segment.text) for segment in segments] == [
        (0.1, 0.9, "Hello"),
    ]


def test_parse_diarized_transcript_preserves_timestamps_before_the_end():
    segments = MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(
        "[2][S01]The earlier timestamp is [1] not the end[3]"
    )

    assert [segment.text for segment in segments] == [
        "The earlier timestamp is [1] not the end",
    ]


def test_parse_diarized_transcript_skips_empty_segments():
    segments = MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(
        "[0][S01][1][2][S02]Complete[3]"
    )

    assert [(segment.speaker, segment.text) for segment in segments] == [
        ("S02", "Complete"),
    ]


def test_parse_diarized_transcript_fails_closed_for_incomplete_output():
    segments = MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(
        "[0][S01]Complete[1][2][S02]Incomplete"
    )

    assert segments == []


def test_parse_diarized_transcript_fails_closed_for_trailing_text():
    segments = MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(
        "[0][S01]Complete[1] trailing text"
    )

    assert segments == []


def test_parse_diarized_transcript_preserves_overlong_timestamp_markers():
    marker = f"[{'1' * 33}]"
    segments = MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(
        f"[0][S01]Value {marker}[1]"
    )

    assert [segment.text for segment in segments] == [f"Value {marker}"]
