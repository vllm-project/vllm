# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cohere Chat v2 API protocol.

The bulk of the wire types come straight from the official ``cohere``
Python SDK so we stay in lockstep with the upstream specification and
avoid hand-mirroring the schema. We only own three things locally:

1. The top-level request body model (the SDK doesn't ship one — its
   ``ClientV2.chat`` takes the body as kwargs), with vLLM-specific
   extensions (``kv_transfer_params`` / ``chat_template_kwargs``).
2. The non-streaming response envelope (the SDK exposes the message
   shape via :class:`AssistantMessageResponse` but no full response
   wrapper).
3. The streaming discriminated union (the SDK exports each event type
   individually but not as a combined ``Annotated[Union[...],
   discriminator]``).

Importing this module pulls in the ``cohere`` package. The router that
mounts ``POST /cohere/v2/chat`` guards on that import succeeding so vLLM still
boots without the SDK installed.

See https://docs.cohere.com/reference/chat for the upstream spec.
"""

from __future__ import annotations

from typing import Any, Literal

from cohere import types as _sdk
from cohere.types import (
    AssistantChatMessageV2,
    AssistantMessageResponse,
    ChatMessageV2,
    ChatRequestSafetyMode,
    Citation,
    CitationOptions,
    Document,
    ResponseFormatV2,
    SystemChatMessageV2,
    Thinking,
    ToolCallV2,
    ToolChatMessageV2,
    ToolV2,
    UserChatMessageV2,
)
from pydantic import BaseModel, Field, field_validator

# Re-export the SDK wire-format types alongside our local extensions so
# ``vllm.entrypoints.cohere.serving`` and friends can import everything
# they need from this module.
__all__ = [
    "AssistantChatMessageV2",
    "AssistantMessageResponse",
    "ChatMessageV2",
    "Citation",
    "CitationEndEvent",
    "CitationOptions",
    "CitationStartEvent",
    "CohereChatV2Request",
    "CohereChatV2Response",
    "CohereError",
    "CohereFinishReason",
    "CohereLogprobItem",
    "CohereUsage",
    "CohereUsageBilledUnits",
    "CohereUsageTokens",
    "ContentDeltaEvent",
    "ContentEndEvent",
    "ContentStartEvent",
    "Document",
    "MessageEndEvent",
    "MessageStartEvent",
    "ResponseFormatV2",
    "SystemChatMessageV2",
    "Thinking",
    "ToolCallDeltaEvent",
    "ToolCallEndEvent",
    "ToolCallStartEvent",
    "ToolCallV2",
    "ToolChatMessageV2",
    "ToolPlanDeltaEvent",
    "ToolV2",
    "UserChatMessageV2",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CohereError(BaseModel):
    """Top-level error body returned by ``/cohere/v2/chat`` error responses.

    Cohere's documented error schemas are uniform: ``{message, id}``.
    """

    message: str
    id: str | None = None


# ---------------------------------------------------------------------------
# Tool choice / finish reasons
# ---------------------------------------------------------------------------
#
# These literals aren't first-class enums in the SDK but are documented at
# https://docs.cohere.com/reference/chat. We declare them here so the
# request/response models can validate them.

CohereToolChoice = Literal["REQUIRED", "NONE"]

CohereFinishReason = Literal[
    "COMPLETE",
    "STOP_SEQUENCE",
    "MAX_TOKENS",
    "TOOL_CALL",
    "ERROR",
    "TIMEOUT",
]


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class CohereChatV2Request(BaseModel):
    """Cohere Chat v2 request body.

    Mirrors the schema documented at https://docs.cohere.com/reference/chat.
    All structured fields delegate to the official SDK types so the body
    schema stays in sync with the upstream spec.
    """

    model: str
    messages: list[ChatMessageV2]
    stream: bool | None = False

    # Tooling
    tools: list[ToolV2] | None = None
    strict_tools: bool | None = None
    tool_choice: CohereToolChoice | None = None

    # Grounding
    documents: list[str | Document] | None = None
    citation_options: CitationOptions | None = None

    # Output
    response_format: ResponseFormatV2 | None = None
    safety_mode: ChatRequestSafetyMode | None = None
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
    thinking: Thinking | None = None

    # Scheduling
    priority: int | None = None

    # vLLM-specific extensions (not in Cohere spec). These mirror what the
    # Anthropic and OpenAI surfaces already expose so V2 callers can reach
    # the same engine knobs when needed.
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
        if v is not None and v < 0:
            raise ValueError("max_tokens must be non-negative")
        return v

    @field_validator("messages", mode="before")
    @classmethod
    def _normalize_message_roles(cls, v: Any) -> Any:
        """Rewrite OpenAI-style ``developer`` roles to ``system``.

        Cohere's v2 ``ChatMessageV2`` discriminated union only admits
        the four literal roles ``user`` / ``assistant`` / ``system`` /
        ``tool``. OpenAI's ``developer`` role is documented as
        high-priority system instructions, so we alias it onto
        ``system`` *before* the SDK discriminator runs (otherwise it
        rejects the message with a ``literal_error`` against each union
        member). Mirrors ``_role_to_melody`` in the renderer so the
        same alias is honoured no matter which surface the message
        arrives through.

        ``mode="before"`` is required so the rewrite happens before the
        ``list[ChatMessageV2]`` coercion runs the SDK's discriminated
        union; a default-mode validator would never see ``developer``
        because validation would have already failed. On any structural
        surprise (non-iterable input, items without a dict-shaped
        ``role`` field, etc.) we hand ``v`` back unchanged and let
        Pydantic's normal coercion surface a precise error.
        """
        try:
            return [
                {**msg, "role": "system"}
                if msg.get("role", "").lower() == "developer"
                else msg
                for msg in v
            ]
        except (AttributeError, TypeError):
            return v

    @field_validator("messages")
    @classmethod
    def _validate_messages(cls, v: list[ChatMessageV2]) -> list[ChatMessageV2]:
        if not v:
            raise ValueError("messages must contain at least one message")
        return v


# ---------------------------------------------------------------------------
# Usage / Logprobs
# ---------------------------------------------------------------------------
#
# The Cohere SDK only exposes a v1 ``ApiMetaBilledUnits``. The v2 usage
# envelope is documented separately in the OpenAPI spec and we declare it
# here.


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


class CohereLogprobItem(BaseModel):
    text: str | None = None
    token_ids: list[int]
    logprobs: list[float] | None = None


# ---------------------------------------------------------------------------
# Non-streaming response
# ---------------------------------------------------------------------------


class CohereChatV2Response(BaseModel):
    """Cohere Chat v2 non-streaming response body.

    Wraps the SDK :class:`AssistantMessageResponse` (the message shape) in
    the documented v2 response envelope (``id``, ``finish_reason``,
    ``usage``, ``logprobs``). The single constructor in
    :class:`CohereServingChatV2._chat_completion_to_v2` is responsible for
    supplying a non-empty ``id`` (falling back to a synthesized one if
    the upstream response is missing it) to this model.
    """

    id: str
    finish_reason: CohereFinishReason
    message: AssistantMessageResponse
    usage: CohereUsage | None = None
    logprobs: list[CohereLogprobItem] | None = None

    # vLLM-specific extension.
    kv_transfer_params: dict[str, Any] | None = Field(
        default=None, description="KVTransfer parameters."
    )


# ---------------------------------------------------------------------------
# Streaming events
# ---------------------------------------------------------------------------
#
# Cohere V2 streams a sequence of typed JSON events delivered as Server-
# Sent Events; each event's ``type`` field carries the discriminator. The
# SDK exposes a Pydantic model per event but none of them declare ``type``
# as a field (the SDK relies on its own deserializer for discrimination),
# so a naive ``model_dump_json()`` would silently drop the discriminator
# and break clients that demux on ``type``.
#
# We therefore subclass each SDK event and bake the wire-format ``type``
# string in as a ``Literal`` field with a default. ``model_dump()`` now
# emits ``type`` for free, and surfaces can simply construct the event
# class and serialize it -- no manual ``type`` parameter needed.
#
# See https://docs.cohere.com/v2/docs/streaming and the OpenAPI
# ``StreamedChatResponseV2`` schema for the wire-format reference.


class MessageStartEvent(_sdk.ChatMessageStartEvent):
    type: Literal["message-start"] = "message-start"


class ContentStartEvent(_sdk.ChatContentStartEvent):
    type: Literal["content-start"] = "content-start"


class ContentDeltaEvent(_sdk.ChatContentDeltaEvent):
    type: Literal["content-delta"] = "content-delta"


class ContentEndEvent(_sdk.ChatContentEndEvent):
    type: Literal["content-end"] = "content-end"


class ToolPlanDeltaEvent(_sdk.ChatToolPlanDeltaEvent):
    type: Literal["tool-plan-delta"] = "tool-plan-delta"


class ToolCallStartEvent(_sdk.ChatToolCallStartEvent):
    type: Literal["tool-call-start"] = "tool-call-start"


class ToolCallDeltaEvent(_sdk.ChatToolCallDeltaEvent):
    type: Literal["tool-call-delta"] = "tool-call-delta"


class ToolCallEndEvent(_sdk.ChatToolCallEndEvent):
    type: Literal["tool-call-end"] = "tool-call-end"


class CitationStartEvent(_sdk.CitationStartEvent):
    type: Literal["citation-start"] = "citation-start"


class CitationEndEvent(_sdk.CitationEndEvent):
    type: Literal["citation-end"] = "citation-end"


class MessageEndEvent(_sdk.ChatMessageEndEvent):
    type: Literal["message-end"] = "message-end"
