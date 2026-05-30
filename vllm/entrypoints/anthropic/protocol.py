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


def _normalize_system_blocks(content: Any) -> list[dict[str, Any]]:
    """Normalize Anthropic system content to a list of text content blocks.

    Accepts the shapes a system prompt may take — ``None``, a plain string, a
    list of content blocks (dicts or already-parsed objects), or a stray
    object — and returns a list of ``{"type": "text", "text": ...}`` blocks.
    Non-text blocks are passed through unchanged so downstream consumers can
    still inspect them. Used by ``AnthropicMessagesRequest.hoist_system_messages``
    to merge hoisted ``role="system"`` entries with any existing top-level
    ``system`` field.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, str):
                if block:
                    blocks.append({"type": "text", "text": block})
            elif isinstance(block, dict):
                blocks.append(block)
            else:
                # Already-parsed block model (e.g. AnthropicContentBlock).
                dump = getattr(block, "model_dump", None)
                blocks.append(dump(exclude_none=True) if dump else block)
        return blocks
    return [{"type": "text", "text": str(content)}]


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

    @model_validator(mode="before")
    @classmethod
    def hoist_system_messages(cls, data: Any) -> Any:
        """Hoist ``role="system"`` entries from ``messages[]`` into ``system``.

        Anthropic's spec only allows ``user``/``assistant`` roles inside
        ``messages[]`` and carries the system prompt in the top-level ``system``
        field. However, Claude Code (and some other clients) emit the system
        prompt as a ``role="system"`` entry *inside* ``messages[]``, which the
        strict per-message ``Literal["user", "assistant"]`` validator rejects
        with HTTP 400. Be lenient: pull any such entries out and prepend them to
        the top-level ``system`` field before per-message validation runs.

        Order is preserved and any pre-existing top-level ``system`` content is
        kept ahead of the hoisted entries. Content is normalized to a list of
        content blocks so downstream handling (e.g. billing-header stripping in
        the serving layer) still applies. Returns ``data`` unchanged when there
        is nothing to hoist or the shape is unexpected, so it is safe to run
        unconditionally.
        """
        if not isinstance(data, dict):
            return data
        messages = data.get("messages")
        if not isinstance(messages, list):
            return data

        hoisted: list[dict[str, Any]] = []
        kept: list[Any] = []
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "system":
                hoisted.extend(_normalize_system_blocks(message.get("content")))
            else:
                kept.append(message)

        if not hoisted:
            return data

        blocks = _normalize_system_blocks(data.get("system")) + hoisted

        patched = dict(data)
        patched["messages"] = kept
        patched["system"] = blocks
        return patched

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
