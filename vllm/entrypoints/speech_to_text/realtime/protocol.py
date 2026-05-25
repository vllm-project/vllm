# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Literal

from pydantic import Field

from vllm.entrypoints.openai.engine.protocol import (
    OpenAIBaseModel,
    UsageInfo,
)
from vllm.utils import random_uuid

# Client -> Server Events


class InputAudioBufferAppend(OpenAIBaseModel):
    """Append audio chunk to buffer"""

    type: Literal["input_audio_buffer.append"] = "input_audio_buffer.append"
    audio: str  # base64-encoded PCM16 @ 16kHz


class InputAudioBufferCommit(OpenAIBaseModel):
    """Process accumulated audio buffer"""

    type: Literal["input_audio_buffer.commit"] = "input_audio_buffer.commit"
    final: bool = False


# Server -> Client Events
class SessionUpdate(OpenAIBaseModel):
    """Configure session parameters"""

    type: Literal["session.update"] = "session.update"
    model: str | None = None


class SessionCreated(OpenAIBaseModel):
    """Connection established notification"""

    type: Literal["session.created"] = "session.created"
    id: str = Field(default_factory=lambda: f"sess-{random_uuid()}")
    created: int = Field(default_factory=lambda: int(time.time()))


class TranscriptionDelta(OpenAIBaseModel):
    """Incremental transcription text"""

    type: Literal["transcription.delta"] = "transcription.delta"
    delta: str  # Incremental text


class TranscriptionDone(OpenAIBaseModel):
    """Final transcription with usage stats"""

    type: Literal["transcription.done"] = "transcription.done"
    text: str  # Complete transcription
    usage: UsageInfo | None = None


class ErrorEvent(OpenAIBaseModel):
    """Error notification"""

    type: Literal["error"] = "error"
    error: str
    code: str | None = None
