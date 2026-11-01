# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import numpy as np
import pytest
from transformers import AutoProcessor

from vllm.config.speech_to_text import SpeechToTextConfig, SpeechToTextParams
from vllm.model_executor.models.vibevoice_asr import (
    VibeVoiceAsrForConditionalGeneration as VibeVoiceAsr,
)

MODEL_ID = "microsoft/VibeVoice-ASR-HF"
TRANSCRIPT = (
    '[{"Start":0.0,"End":1.5,"Speaker":0,"Content":"Hello there."},'
    '{"Start":1.62,"End":3,"Speaker":1,"Content":"Hi!"}]'
)


def test_parse_diarized_transcript_maps_segments():
    segments = VibeVoiceAsr.parse_diarized_transcript(TRANSCRIPT)

    assert [
        (segment.start, segment.end, segment.speaker, segment.text)
        for segment in segments
    ] == [
        (0.0, 1.5, "0", "Hello there."),
        (1.62, 3.0, "1", "Hi!"),
    ]


def test_parse_diarized_transcript_strips_generated_assistant_header():
    """Without a generation prompt, the model emits the assistant header."""
    segments = VibeVoiceAsr.parse_diarized_transcript(f"assistant\n{TRANSCRIPT}\n")

    assert [segment.text for segment in segments] == ["Hello there.", "Hi!"]


def test_parse_diarized_transcript_keeps_complete_segments_of_truncated_output():
    segments = VibeVoiceAsr.parse_diarized_transcript(TRANSCRIPT[:-20])

    assert [segment.text for segment in segments] == ["Hello there."]


def test_parse_diarized_transcript_fails_closed_for_non_json_output():
    assert VibeVoiceAsr.parse_diarized_transcript("Hello there.") == []


def test_post_process_output_joins_segment_contents():
    assert VibeVoiceAsr.post_process_output(TRANSCRIPT) == "Hello there. Hi!"


def test_post_process_output_keeps_unparsable_output():
    assert VibeVoiceAsr.post_process_output(" Hello there.\n") == "Hello there."


@pytest.mark.parametrize("context", [None, "About VibeVoice"])
def test_generation_prompt_matches_hf_chat_template(context: str | None):
    sample_rate = 24_000
    stt_params = SpeechToTextParams(
        audio=np.zeros(int(1.5 * sample_rate), dtype=np.float32),
        stt_config=SpeechToTextConfig(sample_rate=sample_rate),
        model_config=MagicMock(),
        request_prompt=context or "",
    )

    content = [{"type": "audio", "path": "unused.wav"}]
    if context:
        content.append({"type": "text", "text": context})
    hf_prompt = AutoProcessor.from_pretrained(MODEL_ID).apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False
    )
    expected = hf_prompt.replace("<|AUDIO_DURATION|>", "1.50")

    prompt = VibeVoiceAsr.get_generation_prompt(stt_params)["prompt"]

    assert prompt == expected + "<|im_start|>assistant\n"
