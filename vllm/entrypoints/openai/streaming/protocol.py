# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request/response models for the REST streaming API."""

from typing import Any

from pydantic import BaseModel, Field, model_validator

from vllm.v1.streaming.retention import StreamingRetentionParams


class SamplingConfig(BaseModel):
    """Per-reply sampling settings for a session (applied to every frame).

    ``max_tokens`` bounds each frame's reply, not the session. At most one of
    the ``guided_*`` fields may be set; it constrains every reply's output
    (JSON schema, choice list, or regex).
    """

    max_tokens: int = 24
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    frequency_penalty: float = 0.0
    guided_json: Any | None = None
    guided_choice: list[str] | None = None
    guided_regex: str | None = None

    @model_validator(mode="after")
    def _at_most_one_guided(self) -> "SamplingConfig":
        n = sum(
            x is not None
            for x in (self.guided_json, self.guided_choice, self.guided_regex)
        )
        if n > 1:
            raise ValueError(
                "at most one of guided_json / guided_choice / guided_regex may be set"
            )
        return self


class SessionRequest(BaseModel):
    """Client-supplied inputs for a streaming session
    (``POST /v1/streaming/sessions``)."""

    system_prompt: str
    """Pinned task prompt; survives eviction for the whole session."""

    question: str = ""
    """Optional user turn sent alongside the first frame only."""

    fps: float = 1.0
    """Nominal frame rate of the client's stream (informational)."""

    model: str | None = None
    """Requested model name; the server hosts exactly one model."""

    retention: StreamingRetentionParams = Field(
        default_factory=StreamingRetentionParams
    )
    """KV/encoder retention policy for this session (see
    ``StreamingRetentionParams``)."""

    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    """Per-reply sampling settings for this session."""


class SessionResponse(BaseModel):
    """Reply to ``POST /v1/streaming/sessions``: the new session's id plus its
    echoed effective config."""

    session_id: str
    model: str
    fps: float
    retention: StreamingRetentionParams


class ConfigResponse(BaseModel):
    """Reply to ``GET /v1/streaming/config``: the server's model and the
    DEFAULT retention/sampling config (sessions may override per-create)."""

    model: str
    retention: StreamingRetentionParams
    sampling: SamplingConfig


class FrameResponse(BaseModel):
    """Reply to ``POST /v1/streaming/sessions/{id}/frame``: the model's text
    for that frame plus per-frame timing/token accounting.

    ``ttft_s`` (submit -> first output text) and ``latency_s`` (submit ->
    finish) are None when no timing was captured for the frame.
    """

    frame_index: int
    text: str
    finish_reason: str | None = None
    token_count: int = 0
    ttft_s: float | None = None
    latency_s: float | None = None


class CloseResponse(BaseModel):
    """Reply to ``DELETE /v1/streaming/sessions/{id}``: how many frames the
    session answered before closing."""

    session_id: str
    frames: int
    closed: bool = True
