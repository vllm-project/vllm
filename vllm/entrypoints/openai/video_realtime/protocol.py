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

class InputVideoBufferAppend(OpenAIBaseModel):
    """Append one video frame to buffer (video is sent frame-by-frame)"""

    type: Literal["input_video_buffer.append"] = "input_video_buffer.append"
    video: str  # base64-encoded frame (e.g., JPEG/PNG)
    format: str | None = None  # Optional format hint (e.g., "jpeg", "png")

class InputVideoBufferCommit(OpenAIBaseModel):
    """Process accumulated video buffer"""

    type: Literal["input_video_buffer.commit"] = "input_video_buffer.commit"
    final: bool = False

class GenerationTrigger(OpenAIBaseModel):
    """Trigger generation based with custom prompt"""

    type: Literal["generation.trigger"] = "generation.trigger"
    prompt: str | None = None  # Custom prompt to trigger generation

class InputVideoBufferWaterlevel(OpenAIBaseModel):
    """Server buffer water level so client can wait for capacity before sending."""

    queue_depth: int = 0
    """Number of batches in the server queue. 
    Client should send when queue_depth < max_queue_size."""
    max_queue_size: int = 1

    buffer_frames: int = 0

class SessionUpdate(OpenAIBaseModel):
    """Configure session parameters"""

    type: Literal["session.update"] = "session.update"
    model: str | None = None


class SessionCreated(OpenAIBaseModel):
    """Connection established notification"""

    type: Literal["session.created"] = "session.created"
    id: str = Field(default_factory=lambda: f"sess-{random_uuid()}")
    created: int = Field(default_factory=lambda: int(time.time()))
    input_video_buffer: InputVideoBufferWaterlevel | None = None


class CompletionDelta(OpenAIBaseModel):
    """Incremental completion text"""

    type: Literal["completion.delta"] = "completion.delta"
    delta: str


class CompletionDone(OpenAIBaseModel):
    """Final completion with usage stats"""

    type: Literal["completion.done"] = "completion.done"
    text: str  # Complete completion
    usage: UsageInfo | None = None


class ErrorEvent(OpenAIBaseModel):
    """Error notification"""

    type: Literal["error"] = "error"
    error: str
    code: str | None = None
