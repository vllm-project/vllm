# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``response_prefix`` must survive the request -> ``SpeechToTextParams`` hop.

Most of this feature is plumbing: if ``build_stt_params`` drops the field the
transcription/translation endpoints silently ignore ``response_prefix`` with no
error. These guard that wiring for both request types without loading a model.
"""

import io

import numpy as np
from fastapi import UploadFile

from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
)
from vllm.entrypoints.speech_to_text.translation.protocol import TranslationRequest


def _empty_upload() -> UploadFile:
    return UploadFile(file=io.BytesIO(b""), filename="audio.wav")


def _build(req):
    return req.build_stt_params(
        audio=np.zeros(1, dtype=np.float32),
        stt_config=None,
        model_config=None,
        task_type="transcribe",
    )


def test_transcription_passes_response_prefix() -> None:
    params = _build(TranscriptionRequest(file=_empty_upload(), response_prefix="prior"))
    assert params.response_prefix == "prior"


def test_translation_passes_response_prefix() -> None:
    params = _build(TranslationRequest(file=_empty_upload(), response_prefix="prior"))
    assert params.response_prefix == "prior"


def test_response_prefix_defaults_to_empty() -> None:
    params = _build(TranscriptionRequest(file=_empty_upload()))
    assert params.response_prefix == ""
