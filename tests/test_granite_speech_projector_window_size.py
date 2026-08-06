# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.granite_speech import (
    GraniteSpeechMultiModalProcessingInfo,
)

pytestmark = pytest.mark.skip_global_cleanup


class _Context:
    def get_hf_processor(self, **kwargs):
        window_size = int(kwargs.get("projector_window_size", 15))
        return SimpleNamespace(
            audio_processor=SimpleNamespace(projector_window_size=window_size)
        )


def test_granite_speech_request_projector_window_cannot_exceed_server_limit():
    info = GraniteSpeechMultiModalProcessingInfo(_Context())

    with pytest.raises(ValueError, match="may not exceed"):
        info.get_hf_processor(projector_window_size=16)


def test_granite_speech_request_projector_window_can_reduce_server_limit():
    info = GraniteSpeechMultiModalProcessingInfo(_Context())

    processor = info.get_hf_processor(projector_window_size=10)

    assert processor.audio_processor.projector_window_size == 10
