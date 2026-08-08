# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Literal

from pydantic import ConfigDict, Field

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


class SessionUpdate(OpenAIBaseModel):
    """Configure session parameters"""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["session.update"] = "session.update"
    model: str | None = None
    timestamp_granularities: list[Literal["segment"]] = Field(
        alias="timestamp_granularities[]", default_factory=list
    )
    """Timestamp granularities to populate on transcription events.

    Only `segment` is supported here; `word` timestamps are available on
    `POST /v1/audio/transcriptions` but not on this endpoint.
    """


# Server -> Client Events
class SessionCreated(OpenAIBaseModel):
    """Connection established notification"""

    type: Literal["session.created"] = "session.created"
    id: str = Field(default_factory=lambda: f"sess-{random_uuid()}")
    created: int = Field(default_factory=lambda: int(time.time()))


class SessionUpdated(OpenAIBaseModel):
    """Acknowledgement echoing the configuration that took effect."""

    type: Literal["session.updated"] = "session.updated"
    model: str | None = None
    timestamp_granularities: list[str] = Field(default_factory=list)


class TranscriptionSegmentTimestamp(OpenAIBaseModel):
    """A transcribed emission group and its end time within the utterance.

    One entry per boundary marker emitted by the model; words the model chose
    to emit together share one entry. End only: the model marks where a group
    of audio ends, not where it starts. Granularity is one audio frame (80 ms
    for Voxtral realtime). The clock restarts on every non-final
    `input_audio_buffer.commit`.
    """

    text: str
    """The transcribed text of the emission group."""

    end: float
    """End time of the group in seconds, relative to the current utterance."""


class TranscriptionDelta(OpenAIBaseModel):
    """Incremental transcription text"""

    type: Literal["transcription.delta"] = "transcription.delta"
    delta: str  # Incremental text
    segments: list[TranscriptionSegmentTimestamp] | None = None
    """Segments closed by this delta, when segment timestamps are enabled.

    Because the boundary marker itself decodes to no text, the segments
    describing a piece of text arrive on the delta *after* the one that
    carried it, and that delta's `delta` is the empty string.
    """


class TranscriptionDone(OpenAIBaseModel):
    """Final transcription with usage stats"""

    type: Literal["transcription.done"] = "transcription.done"
    text: str  # Complete transcription
    usage: UsageInfo | None = None
    segments: list[TranscriptionSegmentTimestamp] | None = None
    """Every segment of the utterance, repeating what the deltas already sent."""


class ErrorEvent(OpenAIBaseModel):
    """Error notification"""

    type: Literal["error"] = "error"
    error: str
    code: str | None = None
