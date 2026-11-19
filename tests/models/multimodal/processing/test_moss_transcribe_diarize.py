# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

import numpy as np
import pytest

from vllm.config.speech_to_text import SpeechToTextConfig, SpeechToTextParams
from vllm.model_executor.models.interfaces import SupportsTranscription
from vllm.model_executor.models.moss_transcribe_diarize import (
    DEFAULT_MOSS_TRANSCRIBE_DIARIZE_PROMPT,
    MossTranscribeDiarizeForConditionalGeneration,
)


def _question(request_prompt: str = "", hotwords: str | None = None) -> str:
    params = SpeechToTextParams(
        audio=np.zeros(1600, dtype=np.float32),
        stt_config=SpeechToTextConfig(),
        model_config=None,
        hotwords=hotwords,
        request_prompt=request_prompt,
    )
    prompt_data = MossTranscribeDiarizeForConditionalGeneration.get_generation_prompt(
        params
    )
    assert isinstance(prompt_data, dict)
    prompt = prompt_data.get("prompt")
    assert isinstance(prompt, str)
    return prompt.split("<|audio_end|>\n", 1)[1].split("<|im_end|>", 1)[0]


@pytest.mark.parametrize(
    ("request_prompt", "hotwords", "expected"),
    [
        ("", None, DEFAULT_MOSS_TRANSCRIBE_DIARIZE_PROMPT),
        ("", "  ", DEFAULT_MOSS_TRANSCRIBE_DIARIZE_PROMPT),
        (
            "",
            "魔搭, 达摩院",
            DEFAULT_MOSS_TRANSCRIBE_DIARIZE_PROMPT + "热词提示：魔搭, 达摩院",
        ),
        ("转录为文本。", " vLLM ", "转录为文本。热词提示：vLLM"),
    ],
    ids=[
        "default",
        "blank-hotwords",
        "hotwords",
        "custom-prompt-hotwords",
    ],
)
def test_generation_prompt_appends_hotwords(request_prompt, hotwords, expected):
    assert _question(request_prompt, hotwords) == expected


def test_parse_diarized_transcript_preserves_moss_segments():
    segments = MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(
        "[0.48][S01]Welcome[1.66][12.26][S02]Ready[13.81]", None
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
        "[0][S01]First speaker[2][1][S02]Second speaker[3]", None
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
        "[0][S01]The [2024] report is ready.[4]", None
    )

    assert [segment.text for segment in segments] == ["The [2024] report is ready."]


def test_parse_diarized_transcript_ignores_whitespace_between_segments():
    segments = MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(
        "[0][S01]Hello[1]\n [2][S02]Hi[3]", None
    )

    assert [(segment.start, segment.end, segment.text) for segment in segments] == [
        (0.0, 1.0, "Hello"),
        (2.0, 3.0, "Hi"),
    ]


def test_parse_diarized_transcript_ignores_noise_before_a_segment():
    segments = MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(
        "noise [bad][0.1][S01]Hello[0.9]", None
    )

    assert [(segment.start, segment.end, segment.text) for segment in segments] == [
        (0.1, 0.9, "Hello"),
    ]


def test_parse_diarized_transcript_preserves_timestamps_before_the_end():
    segments = MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(
        "[2][S01]The earlier timestamp is [1] not the end[3]", None
    )

    assert [segment.text for segment in segments] == [
        "The earlier timestamp is [1] not the end",
    ]


def test_parse_diarized_transcript_skips_empty_segments():
    segments = MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(
        "[0][S01][1][2][S02]Complete[3]", None
    )

    assert [(segment.speaker, segment.text) for segment in segments] == [
        ("S02", "Complete"),
    ]


def test_parse_diarized_transcript_fails_closed_for_incomplete_output():
    segments = MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(
        "[0][S01]Complete[1][2][S02]Incomplete", None
    )

    assert segments == []


def test_parse_diarized_transcript_fails_closed_for_trailing_text():
    segments = MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(
        "[0][S01]Complete[1] trailing text", None
    )

    assert segments == []


def test_parse_diarized_transcript_preserves_overlong_timestamp_markers():
    marker = f"[{'1' * 33}]"
    segments = MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(
        f"[0][S01]Value {marker}[1]", None
    )

    assert [segment.text for segment in segments] == [f"Value {marker}"]


def _streaming_parser():
    model_cls = cast(
        type[SupportsTranscription], MossTranscribeDiarizeForConditionalGeneration
    )
    return model_cls.get_streaming_diarized_parser_cls()(model_cls, None)


def test_streaming_parser_sends_a_segment_once_the_next_one_starts():
    parser = _streaming_parser()
    deltas = ["[0.48][S01]Good ", "morning[1.66][12", ".26][S0", "2]Ready[13.81]"]

    sent = [
        [(s.speaker, s.text) for s in parser.process_delta(delta, finished=False)]
        for delta in deltas
    ]

    assert sent == [[], [], [], [("S01", "Good morning")]]
    assert [(s.speaker, s.text) for s in parser.process_delta("", finished=True)] == [
        ("S02", "Ready")
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "[0.48][S01]Welcome[1.66][12.26][S02]Ready[13.81]",
            [(0.48, 1.66, "S01", "Welcome"), (12.26, 13.81, "S02", "Ready")],
        ),
        (
            "[0][S01]First speaker[2][1][S02]Second speaker[3]",
            [(0, 2, "S01", "First speaker"), (1, 3, "S02", "Second speaker")],
        ),
        (
            "[0][S01]The [2024] report is ready.[4]",
            [(0, 4, "S01", "The [2024] report is ready.")],
        ),
        (
            "[0][S01]Hello[1]\n [2][S02]Hi[3]",
            [(0, 1, "S01", "Hello"), (2, 3, "S02", "Hi")],
        ),
        ("noise [bad][0.1][S01]Hello[0.9]", [(0.1, 0.9, "S01", "Hello")]),
        ("[0][S01][1][2][S02]Complete[3]", [(2, 3, "S02", "Complete")]),
        ("[0][S01]a[1..2][S02]b[1]", [(0, 1, "S01", "a[1..2][S02]b")]),
        (
            "[0][S01]A[2]\x1c[S02]B[3]",
            [(0, 3, "S01", "A[2]\x1c[S02]B")],
        ),
        (
            f"[0][S01]Value [{'1' * 33}][1]",
            [(0, 1, "S01", f"Value [{'1' * 33}]")],
        ),
    ],
)
@pytest.mark.parametrize("chunk_size", [1, 7, 1000])
def test_streaming_parser_output_does_not_depend_on_delta_boundaries(
    text, expected, chunk_size
):
    """Both parsing paths must match explicit semantics, not each other's output."""
    parser = _streaming_parser()

    segments = [
        s
        for offset in range(0, len(text), chunk_size)
        for s in parser.process_delta(
            text[offset : offset + chunk_size], finished=False
        )
    ]
    segments += parser.process_delta("", finished=True)

    assert [(s.start, s.end, s.speaker, s.text) for s in segments] == expected
    non_streaming = (
        MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(
            text, None
        )
    )
    assert [(s.start, s.end, s.speaker, s.text) for s in non_streaming] == expected


@pytest.mark.parametrize("speaker_marker", ["[S02]", "[Sx]"])
def test_streaming_parser_preserves_whitespace_until_header_is_resolved(speaker_marker):
    whitespace = "\t\u2003\n " * 512
    parser = _streaming_parser()
    text = "[0][S01]A[1][2]" + whitespace + speaker_marker
    sent = [s for char in text for s in parser.process_delta(char, finished=False)]
    assert [(s.start, s.end, s.speaker, s.text) for s in sent] == (
        [(0, 1, "S01", "A")] if speaker_marker == "[S02]" else []
    )
    sent.extend(parser.process_delta("B[3]", finished=True))
    expected = (
        [(0, 1, "S01", "A"), (2, 3, "S02", "B")]
        if speaker_marker == "[S02]"
        else [(0, 3, "S01", "A[1][2]" + whitespace + "[Sx]B")]
    )
    assert [(s.start, s.end, s.speaker, s.text) for s in sent] == expected


@pytest.mark.parametrize(
    ("text", "expected_prefix"),
    [
        ("noise only", []),
        ("[0][S01][1]", []),
        ("[2][S01]Reversed[1]", []),
        ("[0][S01]A[1] trailing text", []),
        ("[0][S01]A[1][S", []),
        ("[0][S01]Complete[1][2][S02]Incomplete", [(0, 1, "S01", "Complete")]),
    ],
)
@pytest.mark.parametrize("chunk_size", [1, 1000])
def test_streaming_parser_rejects_invalid_transcripts_with_explicit_prefix(
    text, expected_prefix, chunk_size
):
    parser = _streaming_parser()
    segments = []
    with pytest.raises(ValueError, match="valid diarized transcript"):
        for offset in range(0, len(text), chunk_size):
            segments.extend(
                parser.process_delta(text[offset : offset + chunk_size], finished=False)
            )
        segments.extend(parser.process_delta("", finished=True))
    assert [(s.start, s.end, s.speaker, s.text) for s in segments] == expected_prefix
    assert (
        MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(
            text, None
        )
        == []
    )


def test_streaming_parser_fails_closed_after_sending_complete_segments():
    parser = _streaming_parser()

    sent = parser.process_delta("[0][S01]Complete[1][2][S02]Incomplete", finished=False)

    assert [s.text for s in sent] == ["Complete"]
    with pytest.raises(ValueError):
        list(parser.process_delta("", finished=True))


@pytest.mark.parametrize("coalesced", [False, True])
def test_streaming_parser_preserves_complete_prefix_before_malformed_segment(coalesced):
    parser = _streaming_parser()
    transcript = "[0][S01]Complete[1][2][S02]No timestamp[3][S03]Later[4]"
    deltas = [transcript] if coalesced else list(transcript)

    segments = []
    with pytest.raises(ValueError, match="valid diarized transcript"):
        for delta in deltas:
            segments.extend(parser.process_delta(delta, finished=False))

    assert [segment.text for segment in segments] == ["Complete"]
