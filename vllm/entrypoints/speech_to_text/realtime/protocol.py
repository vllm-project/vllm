# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Annotated, Literal

from pydantic import Field

from vllm.entrypoints.serve.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.utils import random_uuid

# Client -> Server Events


class InputAudioBufferAppend(OpenAIBaseModel):
    """Append audio chunk to buffer."""

    type: Literal["input_audio_buffer.append"] = "input_audio_buffer.append"
    audio: str  # base64-encoded PCM16 @ 16kHz


class InputAudioBufferCommit(OpenAIBaseModel):
    """Process accumulated audio buffer."""

    type: Literal["input_audio_buffer.commit"] = "input_audio_buffer.commit"
    final: bool = False


class ServerVad(OpenAIBaseModel):
    """End a turn after a fixed silence, judged by Silero VAD."""

    type: Literal["server_vad"]
    threshold: float = Field(default=0.5, gt=0.15, le=1.0)
    """Speech probability at which a 32 ms frame counts as speech."""
    prefix_padding_ms: int = Field(default=300, ge=0)
    """How far ``audio_start_ms`` reaches back before the detected speech."""
    silence_duration_ms: int = Field(default=500, ge=0)
    """Silence that ends the turn."""


class SemanticVad(OpenAIBaseModel):
    """End a turn when Smart Turn v3.2 judges the speech before a pause complete."""

    type: Literal["semantic_vad"]
    eagerness: Literal["low", "medium", "high", "auto"] = "auto"
    """How soon a pause is judged: ``high`` ends turns soonest, ``low`` waits
    longest; ``auto`` is ``medium``."""


TurnDetection = Annotated[ServerVad | SemanticVad, Field(discriminator="type")]


# Server -> Client Events
class SessionUpdate(OpenAIBaseModel):
    """Configure session parameters."""

    type: Literal["session.update"] = "session.update"
    model: str | None = None
    turn_detection: TurnDetection | None = None
    """Send ``input_audio_buffer.speech_started`` / ``speech_stopped`` events.
    ``null`` (the default) turns detection off."""


class SessionCreated(OpenAIBaseModel):
    """Connection established notification."""

    type: Literal["session.created"] = "session.created"
    id: str = Field(default_factory=lambda: f"sess-{random_uuid()}")
    created: int = Field(default_factory=lambda: int(time.time()))


class TranscriptionDelta(OpenAIBaseModel):
    """Incremental transcription text."""

    type: Literal["transcription.delta"] = "transcription.delta"
    delta: str  # Incremental text


class TranscriptionDone(OpenAIBaseModel):
    """Final transcription with usage stats."""

    type: Literal["transcription.done"] = "transcription.done"
    text: str  # Complete transcription
    usage: UsageInfo | None = None


class InputAudioBufferSpeechStarted(OpenAIBaseModel):
    """Turn detection heard the start of a turn."""

    type: Literal["input_audio_buffer.speech_started"] = (
        "input_audio_buffer.speech_started"
    )
    audio_start_ms: int
    """Where the turn starts, in ms from the first audio of the session,
    including ``prefix_padding_ms``."""


class InputAudioBufferSpeechStopped(OpenAIBaseModel):
    """Turn detection heard the end of a turn."""

    type: Literal["input_audio_buffer.speech_stopped"] = (
        "input_audio_buffer.speech_stopped"
    )
    audio_end_ms: int
    """Where the turn was ended, in ms from the first audio of the session,
    including the trailing silence."""


class ErrorEvent(OpenAIBaseModel):
    """Error notification."""

    type: Literal["error"] = "error"
    error: str
    code: str | None = None
