# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pydantic models for the Cohere Chat v2 API protocol.

See https://docs.cohere.com/reference/chat for the upstream specification.

This module mirrors the Cohere V2 wire format using Pydantic discriminated
unions for messages, content blocks, and stream events.
"""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field, RootModel, field_validator

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CohereError(BaseModel):
    """Top-level error body returned by /v2/chat error responses.

    Cohere's documented error schemas are uniform: ``{message, id}``.
    """

    message: str
    id: str | None = None


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


class CohereTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class CohereThinkingContent(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str


class CohereImageUrl(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] | None = None


class CohereImageContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: CohereImageUrl


CohereUserContentBlock = CohereTextContent | CohereImageContent
CohereAssistantContentBlock = CohereTextContent | CohereThinkingContent


# ---------------------------------------------------------------------------
# Documents (request side) and Document content (tool message side)
# ---------------------------------------------------------------------------


class CohereDocument(BaseModel):
    """A document supplied for grounded generation."""

    data: dict[str, Any] | str
    id: str | None = None


class CohereDocumentContent(BaseModel):
    type: Literal["document"] = "document"
    document: CohereDocument


CohereToolContent = CohereTextContent | CohereDocumentContent


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------


class CohereToolCallFunction(BaseModel):
    name: str | None = None
    # Cohere serializes arguments as a JSON-encoded string (same as OpenAI).
    arguments: str | None = None


class CohereToolCallV2(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: CohereToolCallFunction | None = None


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class CohereUserMessageV2(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[CohereUserContentBlock]


class CohereSystemMessageV2(BaseModel):
    role: Literal["system"] = "system"
    content: str | list[CohereTextContent]


class CohereAssistantMessageV2(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | list[CohereAssistantContentBlock] | None = None
    tool_calls: list[CohereToolCallV2] | None = None
    tool_plan: str | None = Field(
        default=None,
        description=(
            "Chain-of-thought plan emitted by the model alongside tool calls."
        ),
    )
    citations: list["CohereCitation"] | None = None


class CohereToolMessageV2(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str | list[CohereToolContent]


CohereChatMessageV2 = (
    CohereUserMessageV2
    | CohereAssistantMessageV2
    | CohereSystemMessageV2
    | CohereToolMessageV2
)


# ---------------------------------------------------------------------------
# Tools (request side) and tool choice
# ---------------------------------------------------------------------------


class CohereToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class CohereToolV2(BaseModel):
    type: Literal["function"] = "function"
    function: CohereToolFunction


CohereToolChoice = Literal["REQUIRED", "NONE"]


# ---------------------------------------------------------------------------
# Citations / sources
# ---------------------------------------------------------------------------


class CohereChatToolSource(BaseModel):
    type: Literal["tool"] = "tool"
    id: str | None = None
    tool_output: dict[str, Any] | None = None


class CohereChatDocumentSource(BaseModel):
    type: Literal["document"] = "document"
    id: str | None = None
    document: dict[str, Any] | None = None


CohereSource = CohereChatToolSource | CohereChatDocumentSource


class CohereCitation(BaseModel):
    start: int | None = None
    end: int | None = None
    text: str | None = None
    sources: list[CohereSource] | None = None
    content_index: int | None = None
    type: Literal["TEXT_CONTENT", "PLAN"] | None = None


# ---------------------------------------------------------------------------
# Response format / output config
# ---------------------------------------------------------------------------


class CohereTextResponseFormat(BaseModel):
    type: Literal["text"] = "text"


class CohereJsonResponseFormat(BaseModel):
    type: Literal["json_object"] = "json_object"
    json_schema: dict[str, Any] | None = None


CohereResponseFormatV2 = CohereTextResponseFormat | CohereJsonResponseFormat


# ---------------------------------------------------------------------------
# Misc options
# ---------------------------------------------------------------------------


CohereSafetyMode = Literal["CONTEXTUAL", "STRICT", "OFF"]
CohereCitationMode = Literal["ENABLED", "DISABLED", "FAST", "ACCURATE", "OFF"]


class CohereCitationOptions(BaseModel):
    mode: CohereCitationMode | None = None


class CohereThinking(BaseModel):
    type: Literal["enabled", "disabled"]
    token_budget: int | None = None


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class CohereChatV2Request(BaseModel):
    """Cohere Chat v2 request body.

    Mirrors the schema documented at
    https://docs.cohere.com/reference/chat.
    """

    model: str
    messages: list[CohereChatMessageV2]
    stream: bool | None = False

    # Tooling
    tools: list[CohereToolV2] | None = None
    strict_tools: bool | None = None
    tool_choice: CohereToolChoice | None = None

    # Grounding
    documents: list[str | CohereDocument] | None = None
    citation_options: CohereCitationOptions | None = None

    # Output
    response_format: CohereResponseFormatV2 | None = None
    safety_mode: CohereSafetyMode | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None

    # Sampling
    temperature: float | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    k: int | None = None
    p: float | None = None
    logprobs: bool | None = None

    # Reasoning
    thinking: CohereThinking | None = None

    # Scheduling
    priority: int | None = None

    # vLLM-specific extensions (not in Cohere spec). These mirror what the
    # Anthropic and OpenAI surfaces already expose so that V2 callers can
    # reach the same engine knobs when needed.
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
    def _validate_model(cls, v: str) -> str:
        if not v:
            raise ValueError("model is required")
        return v

    @field_validator("max_tokens")
    @classmethod
    def _validate_max_tokens(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


class CohereUsageBilledUnits(BaseModel):
    input_tokens: float | None = None
    output_tokens: float | None = None
    search_units: float | None = None
    classifications: float | None = None


class CohereUsageTokens(BaseModel):
    input_tokens: float | None = None
    output_tokens: float | None = None


class CohereUsage(BaseModel):
    billed_units: CohereUsageBilledUnits | None = None
    tokens: CohereUsageTokens | None = None
    cached_tokens: float | None = None


# ---------------------------------------------------------------------------
# Logprobs
# ---------------------------------------------------------------------------


class CohereLogprobItem(BaseModel):
    text: str | None = None
    token_ids: list[int]
    logprobs: list[float] | None = None


# ---------------------------------------------------------------------------
# Non-streaming response
# ---------------------------------------------------------------------------


CohereFinishReason = Literal[
    "COMPLETE",
    "STOP_SEQUENCE",
    "MAX_TOKENS",
    "TOOL_CALL",
    "ERROR",
    "TIMEOUT",
]


class CohereAssistantMessageResponse(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[CohereAssistantContentBlock] | None = None
    tool_calls: list[CohereToolCallV2] | None = None
    tool_plan: str | None = None
    citations: list[CohereCitation] | None = None


class CohereChatV2Response(BaseModel):
    id: str
    finish_reason: CohereFinishReason
    message: CohereAssistantMessageResponse
    usage: CohereUsage | None = None
    logprobs: list[CohereLogprobItem] | None = None

    # vLLM-specific extension.
    kv_transfer_params: dict[str, Any] | None = Field(
        default=None, description="KVTransfer parameters."
    )

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            self.id = f"chat_{int(time.time() * 1000)}"


# ---------------------------------------------------------------------------
# Streaming events
# ---------------------------------------------------------------------------
#
# Cohere V2 streams a sequence of typed JSON events delivered as
# Server-Sent Events. Each event is a discriminated union over the ``type``
# field. See:
#   https://docs.cohere.com/v2/docs/streaming
# and the OpenAPI ``StreamedChatResponseV2`` schema.


class _CohereStreamEventBase(BaseModel):
    """Common base – every event has a ``type`` discriminator and the
    nested ``delta`` shapes are loose dicts so we don't have to redefine the
    full Cohere schema for each event variant."""


class CohereChatMessageStartEvent(_CohereStreamEventBase):
    type: Literal["message-start"] = "message-start"
    id: str | None = None
    delta: dict[str, Any] | None = None


class CohereChatContentStartEvent(_CohereStreamEventBase):
    type: Literal["content-start"] = "content-start"
    index: int | None = None
    delta: dict[str, Any] | None = None


class CohereChatContentDeltaEvent(_CohereStreamEventBase):
    type: Literal["content-delta"] = "content-delta"
    index: int | None = None
    delta: dict[str, Any] | None = None
    logprobs: CohereLogprobItem | None = None


class CohereChatContentEndEvent(_CohereStreamEventBase):
    type: Literal["content-end"] = "content-end"
    index: int | None = None


class CohereChatToolPlanDeltaEvent(_CohereStreamEventBase):
    type: Literal["tool-plan-delta"] = "tool-plan-delta"
    delta: dict[str, Any] | None = None


class CohereChatToolCallStartEvent(_CohereStreamEventBase):
    type: Literal["tool-call-start"] = "tool-call-start"
    index: int | None = None
    delta: dict[str, Any] | None = None


class CohereChatToolCallDeltaEvent(_CohereStreamEventBase):
    type: Literal["tool-call-delta"] = "tool-call-delta"
    index: int | None = None
    delta: dict[str, Any] | None = None


class CohereChatToolCallEndEvent(_CohereStreamEventBase):
    type: Literal["tool-call-end"] = "tool-call-end"
    index: int | None = None


class CohereCitationStartEvent(_CohereStreamEventBase):
    type: Literal["citation-start"] = "citation-start"
    index: int | None = None
    delta: dict[str, Any] | None = None


class CohereCitationEndEvent(_CohereStreamEventBase):
    type: Literal["citation-end"] = "citation-end"
    index: int | None = None


class CohereChatMessageEndEvent(_CohereStreamEventBase):
    type: Literal["message-end"] = "message-end"
    id: str | None = None
    delta: dict[str, Any] | None = None


CohereStreamedChatResponseV2 = (
    CohereChatMessageStartEvent
    | CohereChatContentStartEvent
    | CohereChatContentDeltaEvent
    | CohereChatContentEndEvent
    | CohereChatToolPlanDeltaEvent
    | CohereChatToolCallStartEvent
    | CohereChatToolCallDeltaEvent
    | CohereChatToolCallEndEvent
    | CohereCitationStartEvent
    | CohereCitationEndEvent
    | CohereChatMessageEndEvent
)


class CohereStreamedChatResponseV2Envelope(RootModel[CohereStreamedChatResponseV2]):
    """Discriminated-union wrapper for stream events."""


# Resolve forward references created by ``CohereAssistantMessageV2.citations``.
CohereAssistantMessageV2.model_rebuild()
