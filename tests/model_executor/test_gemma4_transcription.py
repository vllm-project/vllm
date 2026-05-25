# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

import numpy as np

from vllm.config.model import ModelConfig
from vllm.config.speech_to_text import SpeechToTextConfig, SpeechToTextParams
from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration


def _make_stt_params(
    *,
    language: str | None = "en",
    task_type: str = "transcribe",
    to_language: str | None = None,
) -> SpeechToTextParams:
    return SpeechToTextParams(
        audio=np.zeros(1600, dtype=np.float32),
        stt_config=SpeechToTextConfig(sample_rate=16000),
        model_config=cast(ModelConfig, object()),
        language=language,
        task_type=task_type,
        to_language=to_language,
    )


def test_gemma4_transcription_prompt_uses_audio_token():
    prompt = Gemma4ForConditionalGeneration.get_generation_prompt(_make_stt_params())

    assert prompt["prompt"] == (
        "<bos><|turn>user\n"
        "Transcribe this audio into English: <|audio|><turn|>\n"
        "<|turn>model\n"
    )
    assert prompt["multi_modal_data"]["audio"][1] == 16000


def test_gemma4_translation_prompt_includes_source_and_target_language():
    prompt = Gemma4ForConditionalGeneration.get_generation_prompt(
        _make_stt_params(task_type="translate", language="it", to_language="en")
    )

    assert (
        "Translate this audio from Italian into English: <|audio|>" in prompt["prompt"]
    )
