# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transcription already rejects a string ``file``. Translation did not."""

import pytest
from fastapi import HTTPException

from vllm.entrypoints.speech_to_text.transcription.protocol import TranscriptionRequest
from vllm.entrypoints.speech_to_text.translation.protocol import TranslationRequest


@pytest.mark.parametrize(
    "request_cls",
    [TranscriptionRequest, TranslationRequest],
)
def test_audio_request_rejects_string_file(request_cls):
    with pytest.raises(HTTPException) as exc_info:
        request_cls.model_validate({"file": "not-a-file.wav"})
    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "Expected 'file' to be a file-like object, not 'str'."
