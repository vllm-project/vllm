# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pydantic models for Anthropic API protocol"""

import time
from typing import Any, Literal

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
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class AnthropicContentBlock(BaseModel):
    """Content block in message"""

    type: Literal[
        "text",
        "image",
        "tool_use",
        "tool_result",
        "tool_reference",
        "thinking",
        "redacted_thinking",
    ]
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
    # For tool_reference content
    tool_name: str | None = None
    # For thinking content
    thinking: str | None = None
    signature: str | None = None
    # For redacted thinking content (safety-filtered by the API)
    data: str | None = None


class AnthropicMessage(BaseModel):
    """Message structure"""

    role: Literal["user", "assistant"]
    content: str | list[AnthropicContentBlock]


class AnthropicTool(BaseModel):
    """Tool definition"""

    name: str
    description: str | None = None
    input_schema: dict[str, Any]
    defer_loading: bool | None = None

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

    type: Literal["auto", "any", "tool", "none"]
    name: str | None = None

    @model_validator(mode="after")
    def validate_name_required_for_tool(self) -> "AnthropicToolChoice":
        if self.type == "tool" and not self.name:
            raise ValueError("tool_choice.name is required when type is 'tool'")
        return self


class AnthropicJsonOutputFormat(BaseModel):
    """JSON output format configuration"""

    json_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    type: Literal["json_schema"] = "json_schema"


class AnthropicOutputConfig(BaseModel):
    """Configuration options for the model's output, such as the output format."""

    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    format: AnthropicJsonOutputFormat | None = None


class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request"""

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    metadata: dict[str, Any] | None = None
    output_config: AnthropicOutputConfig | None = None
    stop_sequences: list[str] | None = None
    stream: bool | None = False
    system: str | list[AnthropicContentBlock] | None = None
    temperature: float | None = None
    tool_choice: AnthropicToolChoice | None = None
    tools: list[AnthropicTool] | None = None
    top_k: int | None = None
    top_p: float | None = None

    # vLLM-specific fields that are not in Anthropic spec
    kv_transfer_params: dict[str, Any] | None = Field(
        default=None,
        description="KVTransfer parameters used for disaggregated serving.",
    )
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the chat template renderer. "
            "Will be accessible by the template."
        ),
    )

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

    @model_validator(mode="before")
    @classmethod
    def extract_system_messages(cls, request_body):
        """Extract system messages from the messages array.

        Claude Code sends the system prompt as a message with role
        ``"system"`` inside the messages array instead of using the
        top-level ``system`` field.  Silently move such messages to
        ``system`` so the request succeeds.
        """
        messages = request_body.get("messages")
        if not isinstance(messages, list):
            return request_body

        # Collect system-role messages (preserve original order)
        system_msgs: list[dict] = []
        other_msgs: list[dict] = []
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                other_msgs.append(msg)

        if not system_msgs:
            return request_body  # nothing to do

        # Build system content blocks from extracted messages
        extracted_blocks: list[dict[str, str]] = []
        for msg in system_msgs:
            content = msg.get("content")
            if isinstance(content, str):
                extracted_blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                # Already a list of content blocks – pass through as-is
                extracted_blocks.extend(content)

        if not extracted_blocks:
            return request_body

        # Merge with any existing top-level system field
        existing_system = request_body.get("system")
        if existing_system is None:
            merged_blocks = extracted_blocks
        elif isinstance(existing_system, str):
            merged_blocks = [
                {"type": "text", "text": existing_system},
                *extracted_blocks,
            ]
        elif isinstance(existing_system, list):
            merged_blocks = [*existing_system, *extracted_blocks]
        else:
            # Unexpected type – wrap it
            merged_blocks = [
                {"type": "text", "text": str(existing_system)},
                *extracted_blocks,
            ]

        return {
            **request_body,
            "messages": other_msgs,
            "system": merged_blocks,
        }


class AnthropicDelta(BaseModel):
    """Delta for streaming responses"""

    type: (
        Literal["text_delta", "input_json_delta", "thinking_delta", "signature_delta"]
        | None
    ) = None
    text: str | None = None
    thinking: str | None = None
    partial_json: str | None = None
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

    # vLLM-specific fields that are not in Anthropic spec
    kv_transfer_params: dict[str, Any] | None = Field(
        default=None, description="KVTransfer parameters."
    )

    def model_post_init(self, __context):
        if not self.id:
            self.id = f"msg_{int(time.time() * 1000)}"


class AnthropicContextManagement(BaseModel):
    """Context management information for token counting."""

    original_input_tokens: int


class AnthropicCountTokensRequest(BaseModel):
    """Anthropic messages.count_tokens request"""

    model: str
    messages: list[AnthropicMessage]
    system: str | list[AnthropicContentBlock] | None = None
    tool_choice: AnthropicToolChoice | None = None
    tools: list[AnthropicTool] | None = None

    # vLLM-specific fields that are not in Anthropic spec
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the chat template renderer. "
            "Will be accessible by the template."
        ),
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if not v:
            raise ValueError("Model is required")
        return v


class AnthropicCountTokensResponse(BaseModel):
    """Anthropic messages.count_tokens response"""

    input_tokens: int
    context_management: AnthropicContextManagement | None = None
