# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionResponseVerbose,
)
from vllm.entrypoints.speech_to_text.translation.protocol import (
    TranslationResponseVerbose,
)


@pytest.mark.parametrize(
    "response_cls",
    [TranscriptionResponseVerbose, TranslationResponseVerbose],
)
def test_verbose_response_serializes_duration_as_number(response_cls):
    response = response_cls(
        duration=11.0,
        language="en",
        text="hello",
    )

    duration = response.model_dump(mode="json")["duration"]
    assert duration == 11.0
    assert isinstance(duration, float)
