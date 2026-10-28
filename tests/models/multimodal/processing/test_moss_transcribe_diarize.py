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


def _streaming_parser():
    model_cls = cast(
        type[SupportsTranscription], MossTranscribeDiarizeForConditionalGeneration
    )
    return model_cls.get_streaming_diarized_parser_cls()(model_cls)


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
    "text",
    [
        "[0.48][S01]Welcome[1.66][12.26][S02]Ready[13.81]",
        "[0][S01]First speaker[2][1][S02]Second speaker[3]",
        "[0][S01]The [2024] report is ready.[4]",
        "[0][S01]Hello[1]\n [2][S02]Hi[3]",
        "noise [bad][0.1][S01]Hello[0.9]",
        "[0][S01][1][2][S02]Complete[3]",
        "[0][S01]a[1..2][S02]b[1]",
    ],
)
def test_streaming_parser_output_does_not_depend_on_delta_boundaries(text):
    """Streaming one character at a time gives the non-streaming segments."""
    parser = _streaming_parser()

    segments = [s for char in text for s in parser.process_delta(char, finished=False)]
    segments += parser.process_delta("", finished=True)

    assert segments == (
        MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(text)
    )


def test_streaming_parser_fails_closed_after_sending_complete_segments():
    parser = _streaming_parser()

    sent = parser.process_delta("[0][S01]Complete[1][2][S02]Incomplete", finished=False)

    assert [s.text for s in sent] == ["Complete"]
    with pytest.raises(ValueError):
        parser.process_delta("", finished=True)
