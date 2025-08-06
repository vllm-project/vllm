# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/sgl-project/sglang/blob/220962e46b087b5829137a67eab0205b4d51720b/python/sglang/srt/entrypoints/anthropic/protocol.py
"""Pydantic models for Anthropic API protocol"""

import time
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


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
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


class AnthropicContentBlock(BaseModel):
    """Content block in message"""
    type: Literal["text", "image", "tool_use", "tool_result"]
    text: Optional[str] = None
    # For image content
    source: Optional[Dict[str, Any]] = None
    # For tool use/result
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    is_error: Optional[bool] = None


class AnthropicMessage(BaseModel):
    """Message structure"""
    role: Literal["user", "assistant"]
    content: Union[str, List[AnthropicContentBlock]]


class AnthropicTool(BaseModel):
    """Tool definition"""
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

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
    name: Optional[str] = None


class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request"""
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int
    metadata: Optional[Dict[str, Any]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    system: Optional[str] = None
    temperature: Optional[float] = None
    tool_choice: Optional[AnthropicToolChoice] = None
    tools: Optional[List[AnthropicTool]] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None

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
    type: Literal["text_delta", "input_json_delta"] = None
    text: Optional[str] = None
    partial_json: Optional[str] = None

    # Message delta
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"]] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage = None


class AnthropicStreamEvent(BaseModel):
    """Streaming event"""
    type: Literal[
        "message_start", "message_delta", "message_stop",
        "content_block_start", "content_block_delta", "content_block_stop",
        "ping", "error"
    ]
    message: Optional["AnthropicMessagesResponse"] = None
    delta: Optional[AnthropicDelta] = None
    content_block: Optional[AnthropicContentBlock] = None
    index: Optional[int] = None
    error: Optional[AnthropicError] = None


class AnthropicMessagesResponse(BaseModel):
    """Anthropic Messages API response"""
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[AnthropicContentBlock]
    model: str
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage = None

    def model_post_init(self, __context):
        if not self.id:
            self.id = f"msg_{int(time.time() * 1000)}"


# Forward reference resolution
AnthropicStreamEvent.model_rebuild()
