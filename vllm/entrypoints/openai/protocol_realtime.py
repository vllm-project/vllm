# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OpenAI Realtime WebSocket API Protocol Models

This module defines Pydantic models for all client and server events
in the OpenAI Realtime API protocol.

Reference: https://platform.openai.com/docs/guides/realtime-conversations
"""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from vllm.utils import random_uuid


def generate_event_id() -> str:
    """Generate a unique event ID."""
    return f"event_{random_uuid()}"


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return f"sess_{random_uuid()}"


def generate_conversation_id() -> str:
    """Generate a unique conversation ID."""
    return f"conv_{random_uuid()}"


def generate_item_id() -> str:
    """Generate a unique item ID."""
    return f"item_{random_uuid()}"


def generate_response_id() -> str:
    """Generate a unique response ID."""
    return f"resp_{random_uuid()}"


# =============================================================================
# Base Models
# =============================================================================


class RealtimeBaseEvent(BaseModel):
    """Base class for all Realtime API events."""

    event_id: str = Field(default_factory=generate_event_id)
    type: str


class RealtimeError(BaseModel):
    """Error details for error events."""

    type: str
    code: str | None = None
    message: str
    param: str | None = None
    event_id: str | None = None


# =============================================================================
# Session Configuration
# =============================================================================


class TurnDetection(BaseModel):
    """Turn detection configuration."""

    type: Literal["server_vad", "none"] = "server_vad"
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    create_response: bool = True


class InputAudioTranscription(BaseModel):
    """Input audio transcription configuration."""

    model: str = "whisper-1"


class SessionConfig(BaseModel):
    """Session configuration."""

    modalities: list[Literal["text", "audio"]] = ["text", "audio"]
    instructions: str | None = None
    voice: Literal[
        "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"
    ] = "alloy"
    input_audio_format: Literal["pcm16", "g711_ulaw", "g711_alaw"] = "pcm16"
    output_audio_format: Literal["pcm16", "g711_ulaw", "g711_alaw"] = "pcm16"
    input_audio_transcription: InputAudioTranscription | None = None
    turn_detection: TurnDetection | None = Field(default_factory=TurnDetection)
    tools: list[dict[str, Any]] = []
    tool_choice: Literal["auto", "none", "required"] | dict[str, Any] = "auto"
    temperature: float = 0.8
    max_response_output_tokens: int | Literal["inf"] = "inf"


class Session(SessionConfig):
    """Full session object with ID and model."""

    id: str = Field(default_factory=generate_session_id)
    object: Literal["realtime.session"] = "realtime.session"
    model: str = "gpt-4o-realtime-preview"
    expires_at: int = Field(
        default_factory=lambda: int(time.time()) + 3600
    )  # 1 hour from now


# =============================================================================
# Conversation Items
# =============================================================================


class ContentPart(BaseModel):
    """Content part of a conversation item."""

    type: Literal["input_text", "input_audio", "text", "audio"]
    text: str | None = None
    audio: str | None = None  # Base64-encoded audio
    transcript: str | None = None


class ConversationItem(BaseModel):
    """A conversation item (message)."""

    id: str = Field(default_factory=generate_item_id)
    object: Literal["realtime.item"] = "realtime.item"
    type: Literal["message", "function_call", "function_call_output"]
    status: Literal["completed", "in_progress", "incomplete"] = "completed"
    role: Literal["user", "assistant", "system"] | None = None
    content: list[ContentPart] = []
    call_id: str | None = None  # For function calls
    name: str | None = None  # Function name
    arguments: str | None = None  # Function arguments (JSON string)
    output: str | None = None  # Function call output


class Conversation(BaseModel):
    """A conversation object."""

    id: str = Field(default_factory=generate_conversation_id)
    object: Literal["realtime.conversation"] = "realtime.conversation"


# =============================================================================
# Response Objects
# =============================================================================


class ResponseUsage(BaseModel):
    """Token usage information for a response."""

    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    input_token_details: dict[str, int] = Field(
        default_factory=lambda: {"cached_tokens": 0, "text_tokens": 0, "audio_tokens": 0}
    )
    output_token_details: dict[str, int] = Field(
        default_factory=lambda: {"text_tokens": 0, "audio_tokens": 0}
    )


class Response(BaseModel):
    """A response object."""

    id: str = Field(default_factory=generate_response_id)
    object: Literal["realtime.response"] = "realtime.response"
    status: Literal["in_progress", "completed", "cancelled", "incomplete", "failed"] = (
        "in_progress"
    )
    status_details: dict[str, Any] | None = None
    output: list[ConversationItem] = []
    usage: ResponseUsage | None = None


# =============================================================================
# Client Events (sent from client to server)
# =============================================================================


class SessionUpdateEvent(RealtimeBaseEvent):
    """Client event to update session configuration."""

    type: Literal["session.update"] = "session.update"
    session: SessionConfig


class InputAudioBufferAppendEvent(RealtimeBaseEvent):
    """Client event to append audio to the input buffer."""

    type: Literal["input_audio_buffer.append"] = "input_audio_buffer.append"
    audio: str  # Base64-encoded audio data


class InputAudioBufferCommitEvent(RealtimeBaseEvent):
    """Client event to commit the input audio buffer."""

    type: Literal["input_audio_buffer.commit"] = "input_audio_buffer.commit"


class InputAudioBufferClearEvent(RealtimeBaseEvent):
    """Client event to clear the input audio buffer."""

    type: Literal["input_audio_buffer.clear"] = "input_audio_buffer.clear"


class ConversationItemCreateEvent(RealtimeBaseEvent):
    """Client event to create a conversation item."""

    type: Literal["conversation.item.create"] = "conversation.item.create"
    previous_item_id: str | None = None
    item: ConversationItem


class ConversationItemTruncateEvent(RealtimeBaseEvent):
    """Client event to truncate a conversation item."""

    type: Literal["conversation.item.truncate"] = "conversation.item.truncate"
    item_id: str
    content_index: int
    audio_end_ms: int


class ConversationItemDeleteEvent(RealtimeBaseEvent):
    """Client event to delete a conversation item."""

    type: Literal["conversation.item.delete"] = "conversation.item.delete"
    item_id: str


class ResponseCreateConfig(BaseModel):
    """Configuration for response creation."""

    modalities: list[Literal["text", "audio"]] | None = None
    instructions: str | None = None
    voice: str | None = None
    output_audio_format: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = None
    max_output_tokens: int | Literal["inf"] | None = None


class ResponseCreateEvent(RealtimeBaseEvent):
    """Client event to create a response."""

    type: Literal["response.create"] = "response.create"
    response: ResponseCreateConfig | None = None


class ResponseCancelEvent(RealtimeBaseEvent):
    """Client event to cancel a response."""

    type: Literal["response.cancel"] = "response.cancel"


# =============================================================================
# Server Events (sent from server to client)
# =============================================================================


class ErrorEvent(RealtimeBaseEvent):
    """Server event for errors."""

    type: Literal["error"] = "error"
    error: RealtimeError


class SessionCreatedEvent(RealtimeBaseEvent):
    """Server event when session is created."""

    type: Literal["session.created"] = "session.created"
    session: Session


class SessionUpdatedEvent(RealtimeBaseEvent):
    """Server event when session is updated."""

    type: Literal["session.updated"] = "session.updated"
    session: Session


class ConversationCreatedEvent(RealtimeBaseEvent):
    """Server event when conversation is created."""

    type: Literal["conversation.created"] = "conversation.created"
    conversation: Conversation


class ConversationItemCreatedEvent(RealtimeBaseEvent):
    """Server event when conversation item is created."""

    type: Literal["conversation.item.created"] = "conversation.item.created"
    previous_item_id: str | None = None
    item: ConversationItem


class ConversationItemTruncatedEvent(RealtimeBaseEvent):
    """Server event when conversation item is truncated."""

    type: Literal["conversation.item.truncated"] = "conversation.item.truncated"
    item_id: str
    content_index: int
    audio_end_ms: int


class ConversationItemDeletedEvent(RealtimeBaseEvent):
    """Server event when conversation item is deleted."""

    type: Literal["conversation.item.deleted"] = "conversation.item.deleted"
    item_id: str


class ConversationItemInputAudioTranscriptionCompletedEvent(RealtimeBaseEvent):
    """Server event when input audio transcription is completed."""

    type: Literal["conversation.item.input_audio_transcription.completed"] = (
        "conversation.item.input_audio_transcription.completed"
    )
    item_id: str
    content_index: int
    transcript: str


class ConversationItemInputAudioTranscriptionFailedEvent(RealtimeBaseEvent):
    """Server event when input audio transcription fails."""

    type: Literal["conversation.item.input_audio_transcription.failed"] = (
        "conversation.item.input_audio_transcription.failed"
    )
    item_id: str
    content_index: int
    error: RealtimeError


class InputAudioBufferCommittedEvent(RealtimeBaseEvent):
    """Server event when input audio buffer is committed."""

    type: Literal["input_audio_buffer.committed"] = "input_audio_buffer.committed"
    previous_item_id: str | None = None
    item_id: str


class InputAudioBufferClearedEvent(RealtimeBaseEvent):
    """Server event when input audio buffer is cleared."""

    type: Literal["input_audio_buffer.cleared"] = "input_audio_buffer.cleared"


class InputAudioBufferSpeechStartedEvent(RealtimeBaseEvent):
    """Server event when speech is detected in input audio."""

    type: Literal["input_audio_buffer.speech_started"] = (
        "input_audio_buffer.speech_started"
    )
    audio_start_ms: int
    item_id: str


class InputAudioBufferSpeechStoppedEvent(RealtimeBaseEvent):
    """Server event when speech stops in input audio."""

    type: Literal["input_audio_buffer.speech_stopped"] = (
        "input_audio_buffer.speech_stopped"
    )
    audio_end_ms: int
    item_id: str


class ResponseCreatedEvent(RealtimeBaseEvent):
    """Server event when response is created."""

    type: Literal["response.created"] = "response.created"
    response: Response


class ResponseDoneEvent(RealtimeBaseEvent):
    """Server event when response is done."""

    type: Literal["response.done"] = "response.done"
    response: Response


class ResponseOutputItemAddedEvent(RealtimeBaseEvent):
    """Server event when output item is added to response."""

    type: Literal["response.output_item.added"] = "response.output_item.added"
    response_id: str
    output_index: int
    item: ConversationItem


class ResponseOutputItemDoneEvent(RealtimeBaseEvent):
    """Server event when output item is done."""

    type: Literal["response.output_item.done"] = "response.output_item.done"
    response_id: str
    output_index: int
    item: ConversationItem


class ResponseContentPartAddedEvent(RealtimeBaseEvent):
    """Server event when content part is added."""

    type: Literal["response.content_part.added"] = "response.content_part.added"
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    part: ContentPart


class ResponseContentPartDoneEvent(RealtimeBaseEvent):
    """Server event when content part is done."""

    type: Literal["response.content_part.done"] = "response.content_part.done"
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    part: ContentPart


class ResponseTextDeltaEvent(RealtimeBaseEvent):
    """Server event for text delta."""

    type: Literal["response.text.delta"] = "response.text.delta"
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseTextDoneEvent(RealtimeBaseEvent):
    """Server event when text is done."""

    type: Literal["response.text.done"] = "response.text.done"
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    text: str


class ResponseAudioTranscriptDeltaEvent(RealtimeBaseEvent):
    """Server event for audio transcript delta."""

    type: Literal["response.audio_transcript.delta"] = "response.audio_transcript.delta"
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseAudioTranscriptDoneEvent(RealtimeBaseEvent):
    """Server event when audio transcript is done."""

    type: Literal["response.audio_transcript.done"] = "response.audio_transcript.done"
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    transcript: str


class ResponseAudioDeltaEvent(RealtimeBaseEvent):
    """Server event for audio delta."""

    type: Literal["response.audio.delta"] = "response.audio.delta"
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str  # Base64-encoded audio data


class ResponseAudioDoneEvent(RealtimeBaseEvent):
    """Server event when audio is done."""

    type: Literal["response.audio.done"] = "response.audio.done"
    response_id: str
    item_id: str
    output_index: int
    content_index: int


class ResponseFunctionCallArgumentsDeltaEvent(RealtimeBaseEvent):
    """Server event for function call arguments delta."""

    type: Literal["response.function_call_arguments.delta"] = (
        "response.function_call_arguments.delta"
    )
    response_id: str
    item_id: str
    output_index: int
    call_id: str
    delta: str


class ResponseFunctionCallArgumentsDoneEvent(RealtimeBaseEvent):
    """Server event when function call arguments are done."""

    type: Literal["response.function_call_arguments.done"] = (
        "response.function_call_arguments.done"
    )
    response_id: str
    item_id: str
    output_index: int
    call_id: str
    arguments: str


class RateLimitsUpdatedEvent(RealtimeBaseEvent):
    """Server event when rate limits are updated."""

    type: Literal["rate_limits.updated"] = "rate_limits.updated"
    rate_limits: list[dict[str, Any]]


# =============================================================================
# Event Type Mapping
# =============================================================================

CLIENT_EVENT_TYPES = {
    "session.update": SessionUpdateEvent,
    "input_audio_buffer.append": InputAudioBufferAppendEvent,
    "input_audio_buffer.commit": InputAudioBufferCommitEvent,
    "input_audio_buffer.clear": InputAudioBufferClearEvent,
    "conversation.item.create": ConversationItemCreateEvent,
    "conversation.item.truncate": ConversationItemTruncateEvent,
    "conversation.item.delete": ConversationItemDeleteEvent,
    "response.create": ResponseCreateEvent,
    "response.cancel": ResponseCancelEvent,
}

SERVER_EVENT_TYPES = {
    "error": ErrorEvent,
    "session.created": SessionCreatedEvent,
    "session.updated": SessionUpdatedEvent,
    "conversation.created": ConversationCreatedEvent,
    "conversation.item.created": ConversationItemCreatedEvent,
    "conversation.item.truncated": ConversationItemTruncatedEvent,
    "conversation.item.deleted": ConversationItemDeletedEvent,
    "conversation.item.input_audio_transcription.completed": ConversationItemInputAudioTranscriptionCompletedEvent,
    "conversation.item.input_audio_transcription.failed": ConversationItemInputAudioTranscriptionFailedEvent,
    "input_audio_buffer.committed": InputAudioBufferCommittedEvent,
    "input_audio_buffer.cleared": InputAudioBufferClearedEvent,
    "input_audio_buffer.speech_started": InputAudioBufferSpeechStartedEvent,
    "input_audio_buffer.speech_stopped": InputAudioBufferSpeechStoppedEvent,
    "response.created": ResponseCreatedEvent,
    "response.done": ResponseDoneEvent,
    "response.output_item.added": ResponseOutputItemAddedEvent,
    "response.output_item.done": ResponseOutputItemDoneEvent,
    "response.content_part.added": ResponseContentPartAddedEvent,
    "response.content_part.done": ResponseContentPartDoneEvent,
    "response.text.delta": ResponseTextDeltaEvent,
    "response.text.done": ResponseTextDoneEvent,
    "response.audio_transcript.delta": ResponseAudioTranscriptDeltaEvent,
    "response.audio_transcript.done": ResponseAudioTranscriptDoneEvent,
    "response.audio.delta": ResponseAudioDeltaEvent,
    "response.audio.done": ResponseAudioDoneEvent,
    "response.function_call_arguments.delta": ResponseFunctionCallArgumentsDeltaEvent,
    "response.function_call_arguments.done": ResponseFunctionCallArgumentsDoneEvent,
    "rate_limits.updated": RateLimitsUpdatedEvent,
}


def parse_client_event(data: dict[str, Any]) -> RealtimeBaseEvent:
    """Parse a client event from a dictionary."""
    event_type = data.get("type")
    if event_type not in CLIENT_EVENT_TYPES:
        raise ValueError(f"Unknown client event type: {event_type}")
    return CLIENT_EVENT_TYPES[event_type].model_validate(data)

