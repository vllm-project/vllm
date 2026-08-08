# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.speech_to_text.realtime.protocol import (
    RealtimeSessionConfig,
    SessionUpdate,
    TranscriptionDelta,
    TranscriptionDone,
)

pytestmark = pytest.mark.skip_global_cleanup


def test_session_update_accepts_optional_language_and_prompt() -> None:
    event = SessionUpdate(
        type="session.update",
        model="Qwen/Qwen3-ASR-1.7B",
        language="en",
        prompt="Santander Rewards",
    )

    assert event.model == "Qwen/Qwen3-ASR-1.7B"
    assert event.language == "en"
    assert event.prompt == "Santander Rewards"


def test_session_update_model_only_payload_still_valid() -> None:
    event = SessionUpdate(type="session.update", model="model-name")

    assert event.model == "model-name"
    assert event.language is None
    assert event.prompt is None


def test_realtime_session_config_defaults_are_noop() -> None:
    config = RealtimeSessionConfig()

    assert config.language is None
    assert config.prompt is None


def test_transcription_delta_carries_optional_language() -> None:
    assert TranscriptionDelta(delta="hi").language is None

    event = TranscriptionDelta(delta="hi", language="en")
    assert event.delta == "hi"
    assert event.language == "en"


def test_transcription_done_carries_optional_language() -> None:
    assert TranscriptionDone(text="hi").language is None

    event = TranscriptionDone(text="hi", language="zh")
    assert event.text == "hi"
    assert event.language == "zh"
