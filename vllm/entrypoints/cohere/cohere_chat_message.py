# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM chat-protocol extensions for grounded (citation-carrying) models.

This module keeps the OpenAI chat completion protocol classes
(`ChatMessage` / `DeltaMessage`) free of Cohere-specific fields.
It provides two things:

* `Citation` / `CitationSource`: the vLLM-internal citation
  representation produced by the Cohere parser
  (`vllm.parser.cohere_command`) and consumed by
  `vllm.entrypoints.cohere.serving`. This is *not* the on-wire Cohere
  SDK shape -- that conversion happens in the serving layer.
* `CohereChatMessage` / `CohereDeltaMessage`: `ChatMessage`
  / `DeltaMessage` subclasses that add a `citations` field. They
  are only instantiated by the citation-aware serving handler
  (`vllm.entrypoints.cohere.serving.CohereServingChatV2`) and by the
  Cohere reasoning parser's streaming path. Non-Cohere code paths continue
  to construct plain `ChatMessage` / `DeltaMessage`.

The module intentionally does *not* import the `cohere` Python SDK so the
reasoning parser (which is Cohere-model-specific but ships without the SDK
dependency) can import from here freely.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_serializer

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import ChatMessage
from vllm.entrypoints.serve.engine.protocol import OpenAIBaseModel


class CitationSource(OpenAIBaseModel):
    """Source attribution for a `Citation`.

    Mirrors the shape used by Cohere's Chat v2 API. `type` is the source
    discriminator (`document` or `tool`); `id` is the citing
    document/tool-output identifier; `document` and `tool_output` carry
    the original payload that produced the citation.

    Sources are fully resolved by the parser at emit time
    (see `_melody_sources_to_vllm` in
    `vllm.parser.cohere_command`) using the
    `POSITION_TO_SOURCE_KEY` map forwarded through
    `chat_template_kwargs` by
    `vllm.entrypoints.cohere.serving.CohereServingChatV2._apply_cohere_template_kwargs`.
    That means every instance that reaches the serving / wire layer
    already has `type` / `id` populated; the serving layer only has
    to coerce to `cohere.types.Citation` and rewrite
    `THINKING_CONTENT` -> `PLAN` for non-reasoning models.
    """

    type: Literal["document", "tool"] | None = None
    id: str | None = None
    document: dict[str, Any] | None = None
    tool_output: dict[str, Any] | None = None


class Citation(OpenAIBaseModel):
    """A citation grounding a span of generated text in source material.

    vLLM-internal representation used by the Cohere reasoning parser and
    the Cohere v2 serving layer. This is not the on-wire Cohere SDK shape:
    conversion to the SDK's `cohere.types.Citation` happens in
    `vllm.entrypoints.cohere.serving`.
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
    text (`TEXT_CONTENT`), a thinking block (`THINKING_CONTENT`), or a
    tool-plan block (`PLAN`). `None` means unspecified."""


class CohereChatMessage(ChatMessage):
    """`ChatMessage` extension carrying grounding citations.

    Only instantiated by `CohereServingChatV2` (and any other
    `OpenAIServingChat` subclass that opts in by overriding
    `_create_chat_message`). Regular OpenAI-compatible handlers keep
    emitting plain `ChatMessage` so their response schema is
    unchanged.

    The response envelope declares `message: SerializeAsAny[ChatMessage]`,
    so pydantic serializes this subclass with its own schema (including
    `citations`) when it flows through
    `ChatCompletionResponseChoice`.
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
    """`DeltaMessage` extension carrying grounding citations for streaming.

    Emitted by `vllm.parser.cohere_command.CohereCommandParser`
    on delta events whose payload includes citations. Non-Cohere parsers
    return plain `DeltaMessage`, so their streamed shape is
    unchanged.

    The response envelope declares
    `delta: SerializeAsAny[DeltaMessage]`, so pydantic serializes this
    subclass with its own schema when it flows through
    `ChatCompletionResponseStreamChoice`.
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
