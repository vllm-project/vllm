# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pydantic models for Anthropic API protocol"""

import time
from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator


class AnthropicError(BaseModel):
    """Error structure for Anthropic API"""

    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    """Error response structure for Anthropic API"""

    type: Literal["error"] = "error"
    error: AnthropicError


class AnthropicUsage(BaseModel):
    """Token usage information"""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class AnthropicContentBlock(BaseModel):
    """Content block in message"""

    type: Literal["text", "image", "tool_use", "tool_result", "thinking"]
    text: str | None = None
    # For image content
    source: dict[str, Any] | None = None
    # For tool use/result
    id: str | None = None
    tool_use_id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None
    # For thinking (extended thinking / reasoning) content
    thinking: str | None = None
    signature: str | None = None


class AnthropicMessage(BaseModel):
    """Message structure"""

    # Anthropic spec only allows user/assistant here, but some clients
    # (e.g. recent Claude Code) inject role="system" entries inside the
    # messages array. We accept it at parse time and hoist them into the
    # top-level `system` field via AnthropicMessagesRequest's pre-validator.
    role: Literal["user", "assistant", "system"]
    content: str | list[AnthropicContentBlock]


class AnthropicTool(BaseModel):
    """Tool definition"""

    name: str
    description: str | None = None
    input_schema: dict[str, Any]

    @field_validator("input_schema")
    @classmethod
    def validate_input_schema(cls, v):
        if not isinstance(v, dict):
            raise ValueError("input_schema must be a dictionary")
        if "type" not in v:
            v["type"] = "object"  # Default to object type
        return v


class AnthropicToolChoice(BaseModel):
    """Tool Choice definition"""

    type: Literal["auto", "any", "tool"]
    name: str | None = None

    @model_validator(mode="after")
    def validate_name_required_for_tool(self) -> "AnthropicToolChoice":
        if self.type == "tool" and not self.name:
            raise ValueError("tool_choice.name is required when type is 'tool'")
        return self


class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request"""

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    metadata: dict[str, Any] | None = None
    stop_sequences: list[str] | None = None
    stream: bool | None = False
    system: str | list[AnthropicContentBlock] | None = None
    temperature: float | None = None
    tool_choice: AnthropicToolChoice | None = None
    tools: list[AnthropicTool] | None = None
    top_k: int | None = None
    top_p: float | None = None
    # vLLM-specific opt-in echo fields. When true, the response (and the
    # streaming ``message_delta`` event) gain top-level
    # ``rendered_prompts`` / ``raw_output_texts`` arrays.
    return_rendered_prompts: bool | None = None
    return_raw_output: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def hoist_system_messages(cls, data: Any) -> Any:
        """Move role=system entries from `messages` into the top-level
        `system` field. Some clients (e.g. recent Claude Code) put system
        prompts inside the messages array even though the Anthropic spec
        forbids it. We extract them here so the rest of the pipeline only
        sees user/assistant turns."""
        if not isinstance(data, dict):
            return data
        messages = data.get("messages")
        if not isinstance(messages, list):
            return data

        extracted: list[Any] = []
        remaining: list[Any] = []
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                extracted.append(msg.get("content"))
            else:
                remaining.append(msg)

        if not extracted:
            return data

        existing = data.get("system")
        merged: list[Any] = []
        if isinstance(existing, str):
            if existing:
                merged.append({"type": "text", "text": existing})
        elif isinstance(existing, list):
            merged.extend(existing)

        for content in extracted:
            if isinstance(content, str):
                if content:
                    merged.append({"type": "text", "text": content})
            elif isinstance(content, list):
                merged.extend(content)

        data["messages"] = remaining
        data["system"] = merged
        return data

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if not v:
            raise ValueError("Model is required")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v):
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


class AnthropicDelta(BaseModel):
    """Delta for streaming responses"""

    type: (
        Literal["text_delta", "input_json_delta", "thinking_delta", "signature_delta"]
        | None
    ) = None
    text: str | None = None
    partial_json: str | None = None
    thinking: str | None = None
    signature: str | None = None

    # Message delta
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    ) = None
    stop_sequence: str | None = None


class AnthropicStreamEvent(BaseModel):
    """Streaming event"""

    type: Literal[
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "ping",
        "error",
    ]
    message: "AnthropicMessagesResponse | None" = None
    delta: AnthropicDelta | None = None
    content_block: AnthropicContentBlock | None = None
    index: int | None = None
    error: AnthropicError | None = None
    usage: AnthropicUsage | None = None
    # vLLM-specific opt-in echo, attached on the final ``message_delta``
    # event when the request set ``return_rendered_prompts`` /
    # ``return_raw_output``.
    rendered_prompts: list[str | None] | None = None
    raw_output_texts: list[str] | None = None


class AnthropicMessagesResponse(BaseModel):
    """Anthropic Messages API response"""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicContentBlock]
    model: str
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    ) = None
    stop_sequence: str | None = None
    usage: AnthropicUsage | None = None
    # vLLM-specific opt-in echo: only set when the request asked for it.
    rendered_prompts: list[str | None] | None = None
    raw_output_texts: list[str] | None = None

    def model_post_init(self, __context):
        if not self.id:
            self.id = f"msg_{int(time.time() * 1000)}"
