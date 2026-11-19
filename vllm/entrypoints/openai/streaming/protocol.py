# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request/response models for the REST streaming API."""

from pydantic import BaseModel, ConfigDict, Field

from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.streaming.retention import StreamingRetentionParams


class SamplingConfig(BaseModel):
    """Per-reply sampling settings for a session (applied to every frame).

    `max_tokens` bounds each frame's reply, not the session. Structured output
    constraints are applied independently to every reply.
    """

    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(default=24, ge=1)
    temperature: float = Field(default=0.3, ge=0, allow_inf_nan=False)
    top_p: float = Field(default=0.9, gt=0, le=1)
    repetition_penalty: float = Field(default=1.1, gt=0, le=2)
    frequency_penalty: float = Field(default=0.0, ge=-2, le=2)
    structured_outputs: StructuredOutputsParams | None = None


class SessionRequest(BaseModel):
    """Client-supplied inputs for a streaming session
    (`POST /v1/streaming/sessions`)."""

    model_config = ConfigDict(extra="forbid")

    system_prompt: str
    """Pinned task prompt; survives eviction for the whole session."""

    question: str = ""
    """Optional user turn sent alongside the first frame only."""

    model: str | None = None
    """Requested model name; the server hosts exactly one model."""

    retention: StreamingRetentionParams = Field(
        default_factory=StreamingRetentionParams
    )
    """KV/encoder retention policy for this session (see
    `StreamingRetentionParams`)."""

    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    """Per-reply sampling settings for this session."""


class SessionResponse(BaseModel):
    """Reply to `POST /v1/streaming/sessions`: the new session's id plus its
    echoed effective config."""

    session_id: str
    model: str
    retention: StreamingRetentionParams


class ConfigResponse(BaseModel):
    """Reply to `GET /v1/streaming/config`: the server's model and the
    DEFAULT retention/sampling config (sessions may override per-create)."""

    model: str
    retention: StreamingRetentionParams
    sampling: SamplingConfig


class FrameResponse(BaseModel):
    """Reply to `POST /v1/streaming/sessions/{id}/frame`: the model's text
    for that frame plus per-frame timing/token accounting.

    `ttft_s` (submit -> first output text) and `latency_s` (submit ->
    finish) are None when no timing was captured for the frame.
    """

    frame_index: int
    text: str
    finish_reason: str | None = None
    token_count: int = 0
    ttft_s: float | None = None
    latency_s: float | None = None


class CloseResponse(BaseModel):
    """Reply to `DELETE /v1/streaming/sessions/{id}`: how many frames the
    session answered before closing."""

    session_id: str
    frames: int
    closed: bool = True
