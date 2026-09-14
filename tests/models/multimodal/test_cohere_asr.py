# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import SpeechToTextConfig
from vllm.model_executor.models.cohere_asr import CohereAsrForConditionalGeneration


@pytest.mark.parametrize(
    ("audio_duration_s", "expected_tokens"),
    # window_stride=0.01s @ 16kHz -> 160-sample hop; subsampling_factor=8.
    # frames = floor(duration * 16000 / 160); tokens = ceil(frames / 8).
    [(1.0, 13), (10.0, 125), (30.0, 375)],
)
def test_get_num_audio_tokens_streaming_estimate(audio_duration_s, expected_tokens):
    """The duration-based estimate must convert ``window_stride`` (seconds) to a
    sample hop and divide by the encoder subsampling factor. Values are pinned to
    concrete numbers so a rounding regression is actually caught."""
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            preprocessor={"window_stride": 0.01, "sample_rate": 16000},
            encoder={"subsampling_factor": 8},
        )
    )
    stt_config = SpeechToTextConfig(sample_rate=16000)

    got = CohereAsrForConditionalGeneration.get_num_audio_tokens(
        audio_duration_s, stt_config, model_config
    )

    assert got == expected_tokens
