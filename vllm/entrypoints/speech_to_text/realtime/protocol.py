# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field

from vllm.entrypoints.openai.engine.protocol import (
    OpenAIBaseModel,
    UsageInfo,
)
from vllm.utils import random_uuid

_DEFAULT_SEGMENT_DURATION_S = 2.0


@dataclass
class RealtimeSessionConfig:
    """Session-level configuration for realtime transcription.

    Passed through the stack from session.update to buffer_realtime_audio
    so that model-specific prompt construction can incorporate user-supplied
    language, prompt context, and streaming behaviour tuning.

    Fields set to ``None`` defer to the model-level defaults in
    ``qwen3_asr_realtime.py``.
    """

    language: str | None = field(default=None)
    prompt: str | None = field(default=None)
    segment_duration_s: float = field(default=_DEFAULT_SEGMENT_DURATION_S)
    rollback_tokens: int | None = field(default=None)
    unfixed_chunks: int | None = field(default=None)
    max_prefix_tokens: int | None = field(default=None)
    max_audio_s: float | None = field(default=None)
    realtime_max_tokens: int | None = field(default=None)


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
    """Configure session parameters.

    Aligns with the OpenAI Realtime API session.update contract:
    - model: model identifier
    - language: ISO-639-1 language code (e.g. "en") for prompt prefill
    - prompt: free-text context hint for the transcription model
    - segment_duration_s: audio segment length in seconds
    - rollback_tokens: tokens dropped from end of prefix for re-decision
    - unfixed_chunks: initial segments that skip prefix rollback
    - max_prefix_tokens: hard cap on prefix length (tokens)
    - max_audio_s: max accumulated audio before trimming oldest
    - realtime_max_tokens: max generation tokens per segment
    """

    type: Literal["session.update"] = "session.update"
    model: str | None = None
    language: str | None = None
    prompt: str | None = None
    segment_duration_s: float | None = None
    rollback_tokens: int | None = None
    unfixed_chunks: int | None = None
    max_prefix_tokens: int | None = None
    max_audio_s: float | None = None
    realtime_max_tokens: int | None = None


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
