# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models.moss_transcribe_diarize import (
    MossTranscribeDiarizeForConditionalGeneration,
)


def test_parse_diarized_transcript_preserves_moss_segments():
    segments = MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(
        "[0.48][S01]Hello.[2.80][S02]Hi, how are you?[6.10]"
    )

    assert [
        (segment.start, segment.end, segment.speaker, segment.text)
        for segment in segments
    ] == [
        (0.48, 2.8, "S01", "Hello."),
        (2.8, 6.1, "S02", "Hi, how are you?"),
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
