# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM chat-protocol extensions for grounded (citation-carrying) models.

This module keeps the OpenAI chat completion protocol classes
(:class:`ChatMessage` / :class:`DeltaMessage`) free of Cohere-specific fields.
It provides two things:

* :class:`Citation` / :class:`CitationSource`: the vLLM-internal citation
  representation produced by the Cohere reasoning parser
  (:mod:`vllm.reasoning.cohere_command_reasoning_parser`) and consumed by
  :mod:`vllm.entrypoints.cohere.serving`. This is *not* the on-wire Cohere
  SDK shape -- that conversion happens in the serving layer.
* :class:`CohereChatMessage` / :class:`CohereDeltaMessage`: :class:`ChatMessage`
  / :class:`DeltaMessage` subclasses that add a ``citations`` field. They
  are only instantiated by the citation-aware serving handler
  (:class:`vllm.entrypoints.cohere.serving.CohereServingChatV2`) and by the
  Cohere reasoning parser's streaming path. Non-Cohere code paths continue
  to construct plain :class:`ChatMessage` / :class:`DeltaMessage`.

The module intentionally does *not* import the ``cohere`` Python SDK so the
reasoning parser (which is Cohere-model-specific but ships without the SDK
dependency) can import from here freely.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_serializer

from vllm.entrypoints.openai.chat_completion.protocol import ChatMessage
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    OpenAIBaseModel,
)


class CitationSource(OpenAIBaseModel):
    """Source attribution for a :class:`Citation`.

    Mirrors the shape used by Cohere's Chat v2 API. ``type`` is the source
    discriminator (``document`` or ``tool``); ``id`` is the citing
    document/tool-output identifier; ``document`` and ``tool_output`` carry
    the original payload that produced the citation.
    """

    type: Literal["document", "tool"]
    id: str | None = None
    document: dict[str, Any] | None = None
    tool_output: dict[str, Any] | None = None


class Citation(OpenAIBaseModel):
    """A citation grounding a span of generated text in source material.

    vLLM-internal representation used by the Cohere reasoning parser and
    the Cohere v2 serving layer. This is not the on-wire Cohere SDK shape:
    conversion to the SDK's ``cohere.types.Citation`` happens in
    :mod:`vllm.entrypoints.cohere.serving`.
    """

    start: int | None = None
    """Start character offset in the surrounding text content."""
    end: int | None = None
    """End character offset (exclusive) in the surrounding text content."""
    text: str | None = None
    """The cited text snippet."""
    sources: list[CitationSource] = Field(default_factory=list)
    """Source documents / tool outputs that ground this citation."""
    content_index: int | None = None
    """Index of the content block this citation refers to (when the message
    has multiple content blocks)."""
    type: Literal["TEXT_CONTENT", "THINKING_CONTENT", "PLAN"] | None = None
    """Which kind of content block this citation grounds: the user-visible
    text (``TEXT_CONTENT``), a thinking block (``THINKING_CONTENT``), or a
    tool-plan block (``PLAN``). ``None`` means unspecified."""


class CohereChatMessage(ChatMessage):
    """:class:`ChatMessage` extension carrying grounding citations.

    Only instantiated by :class:`CohereServingChatV2` (and any other
    :class:`OpenAIServingChat` subclass that opts in by overriding
    ``_create_chat_message``). Regular OpenAI-compatible handlers keep
    emitting plain :class:`ChatMessage` so their response schema is
    unchanged.

    The response envelope declares ``message: SerializeAsAny[ChatMessage]``,
    so pydantic serializes this subclass with its own schema (including
    ``citations``) when it flows through
    :class:`ChatCompletionResponseChoice`.
    """

    citations: list[Citation] | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        # ``mode="wrap"`` fully overrides (rather than chains) the
        # parent's ``@model_serializer``, so we explicitly delegate via
        # ``super()._serialize(handler)`` to preserve the parent's
        # cleanup (e.g. stripping empty ``tool_calls``). Then we drop an
        # unset ``citations`` field so the wire matches the OpenAI-style
        # contract that optional vLLM extensions are omitted rather than
        # serialized as ``null``.
        data = super()._serialize(handler)
        if not data.get("citations"):
            data.pop("citations", None)
        return data


class CohereDeltaMessage(DeltaMessage):
    """:class:`DeltaMessage` extension carrying grounding citations for streaming.

    Emitted by
    :class:`vllm.reasoning.cohere_command_reasoning_parser.BaseCohereCommandReasoningParser`
    on delta events whose payload includes citations. Non-Cohere parsers
    return plain :class:`DeltaMessage`, so their streamed shape is
    unchanged.

    The response envelope declares
    ``delta: SerializeAsAny[DeltaMessage]``, so pydantic serializes this
    subclass with its own schema when it flows through
    :class:`ChatCompletionResponseStreamChoice`.
    """

    citations: list[Citation] | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        # See ``CohereChatMessage._serialize`` for why we delegate via
        # ``super()`` instead of calling ``handler(self)`` directly.
        data = super()._serialize(handler)
        if not data.get("citations"):
            data.pop("citations", None)
        return data
