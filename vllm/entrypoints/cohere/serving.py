# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/anthropic/serving.py
"""Cohere Chat v2 API serving handler.

Implements ``POST /cohere/v2/chat`` by translating the incoming Cohere v2 request
into a standard :class:`ChatCompletionRequest` and delegating to
:class:`OpenAIServingChat`. The actual prompt rendering is handled by
vLLM's renderer pipeline (``vllm.renderers``):

- For Cohere Command-family models, set ``--tokenizer-mode cohere`` and the
  :class:`vllm.renderers.cohere.CohereRenderer` will template the request
  via the ``cohere_melody`` library (``render_cmd3`` / ``render_cmd4``)
  and surface citations through the standard ``ChatMessage.citations`` /
  ``DeltaMessage.citations`` fields.
- For any other model the default Jinja-based :class:`HfRenderer` is used,
  and the endpoint behaves as a plain v2-shaped wrapper around chat
  completions.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncGenerator
from enum import Enum
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from cohere.types import (
    AssistantChatMessageV2,
    AssistantMessageResponse,
    Citation,
    SystemChatMessageV2,
    ToolCallV2,
    ToolCallV2Function,
    ToolChatMessageV2,
    UserChatMessageV2,
)
from fastapi import Request
from pydantic import BaseModel

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.cohere.protocol import (
    CitationEndEvent,
    CitationStartEvent,
    CohereChatV2Request,
    CohereChatV2Response,
    CohereFinishReason,
    CohereUsage,
    CohereUsageBilledUnits,
    CohereUsageTokens,
    ContentDeltaEvent,
    ContentEndEvent,
    ContentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolPlanDeltaEvent,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    JsonSchemaResponseFormat,
    ResponseFormat,
    StreamOptions,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.utils.request_logger import RequestLogger

if TYPE_CHECKING:
    from vllm.renderers.online_renderer import OnlineRenderer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse(data: str) -> str:
    """Wrap a JSON payload in a Server-Sent Event frame.

    Cohere's stream uses bare ``data:`` lines (no ``event:`` prefix); each
    JSON object's ``type`` field carries the event discriminator.
    """
    return f"data: {data}\n\n"


# Cohere v2 SSE stream terminator. Declared by the upstream OpenAPI spec
# (``x-fern-streaming.terminator: "[DONE]"`` on the ``/v2/chat`` stream
# operation) and observed by the cohere-python / Fern-generated clients
# (the Python SDK breaks its read loop on ``_sse.data == "[DONE]"``).
# Must be emitted after the closing ``message-end`` event.
_DONE_FRAME = _sse("[DONE]")


def _emit(event: BaseModel) -> str:
    """Serialize a typed stream event into an SSE frame.

    The typed event classes in ``vllm.entrypoints.cohere.protocol``
    (``MessageStartEvent``, ``ContentStartEvent``, ``CitationStartEvent``,
    ...) bake the wire-format ``type`` field into the model definition,
    so a plain ``model_dump_json()`` already carries the discriminator.
    """
    return _sse(event.model_dump_json(exclude_none=True))


class ContentBlockType(str, Enum):
    """Wire-format / internal discriminator for chat content blocks.

    ``THINKING`` and ``TEXT`` are the documented Cohere v2 content-block
    type discriminators on the wire. ``TOOL_CALL`` is reserved for the
    internal stream state machine (see :class:`_StreamState`) when a
    tool call is the currently open block; it is never serialized.
    """

    THINKING = "thinking"
    TEXT = "text"
    TOOL_CALL = "tool_call"


# Mapping of vLLM/OpenAI finish reasons to Cohere's enum.
_FINISH_REASON_MAP: dict[str | None, CohereFinishReason] = {
    "stop": "COMPLETE",
    "length": "MAX_TOKENS",
    "tool_calls": "TOOL_CALL",
    "stop_sequence": "STOP_SEQUENCE",
    "error": "ERROR",
    None: "COMPLETE",
}


def _map_finish_reason(reason: str | None) -> CohereFinishReason:
    return _FINISH_REASON_MAP.get(reason, "COMPLETE")


# ---------------------------------------------------------------------------
# Serving class
# ---------------------------------------------------------------------------


class CohereServingChatV2(OpenAIServingChat):
    """Handler for the Cohere Chat v2 API (``POST /cohere/v2/chat``).

    The handler is intentionally thin: it converts the v2 request into a
    :class:`ChatCompletionRequest` (preserving Cohere-specific fields such
    as ``documents``, ``safety_mode`` and ``citation_options`` via
    ``chat_template_kwargs``) and delegates to the underlying chat
    completion machinery. All Cohere-specific templating happens in the
    renderer (:class:`vllm.renderers.cohere.CohereRenderer`) when the
    engine is started with ``--tokenizer-mode cohere``; citations are read
    natively from the resulting :class:`ChatMessage` / :class:`DeltaMessage`.
    """

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        response_role: str,
        *,
        online_renderer: OnlineRenderer,
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        tool_parser: str | None = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        default_chat_template_kwargs: dict[str, Any] | None = None,
        is_reasoning_model: bool = True,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            response_role=response_role,
            online_renderer=online_renderer,
            request_logger=request_logger,
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
            reasoning_parser=reasoning_parser,
            enable_auto_tools=enable_auto_tools,
            tool_parser=tool_parser,
            enable_prompt_tokens_details=enable_prompt_tokens_details,
            enable_force_include_usage=enable_force_include_usage,
            default_chat_template_kwargs=default_chat_template_kwargs,
        )
        # Controls how the assistant's chain-of-thought is surfaced on
        # turns that also contain tool calls.
        #
        # - ``True`` (default): the model is a reasoning Command-family
        #   model; reasoning is always surfaced as a ``thinking`` content
        #   block (non-streaming) or as ``content-start`` / ``content-
        #   delta`` events for a thinking block (streaming), regardless of
        #   whether tool calls also appear.
        # - ``False``: the model is an older non-reasoning Command model
        #   that uses Cohere's ``tool_plan`` field for its chain-of-
        #   thought before tool calls; reasoning is surfaced as
        #   ``tool_plan`` (non-streaming) or as ``tool-plan-delta`` events
        #   (streaming) on tool-call turns, and the thinking content block
        #   is dropped.
        #
        # TODO: replace this manual flag with automatic detection from the
        # model's capabilities config once that exists.
        self._is_reasoning_model = is_reasoning_model

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def create_chat_v2(
        self,
        request: CohereChatV2Request,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | CohereChatV2Response | ErrorResponse:
        """Implements ``POST /cohere/v2/chat``."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Received Cohere v2 chat request %s", request.model_dump_json()
            )

        chat_req = self._convert_v2_to_chat_completion(request)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Converted Cohere v2 -> ChatCompletion: %s",
                chat_req.model_dump_json(),
            )

        generator = await self.create_chat_completion(chat_req, raw_request)

        if isinstance(generator, ErrorResponse):
            return generator
        if isinstance(generator, ChatCompletionResponse):
            return self._chat_completion_to_v2(generator, request)
        return self._chat_completion_stream_to_v2(generator, request)

    # ==================================================================
    # Request conversion: Cohere V2 -> ChatCompletionRequest
    # ==================================================================

    @classmethod
    def _convert_v2_to_chat_completion(
        cls, request: CohereChatV2Request
    ) -> ChatCompletionRequest:
        openai_messages: list[dict[str, Any]] = []
        cls._convert_messages(request.messages, openai_messages)

        chat_req = cls._build_base_chat_completion(request, openai_messages)
        cls._apply_streaming_options(chat_req, request)
        cls._apply_response_format(chat_req, request)
        cls._apply_tools(chat_req, request)
        cls._apply_tool_choice(chat_req, request)
        cls._apply_cohere_template_kwargs(chat_req, request)
        return chat_req

    @classmethod
    def _convert_messages(
        cls,
        messages: list,
        openai_messages: list[dict[str, Any]],
    ) -> None:
        for msg in messages:
            if isinstance(msg, SystemChatMessageV2):
                openai_messages.append(
                    {
                        "role": "system",
                        "content": cls._coerce_text_content(msg.content),
                    }
                )
            elif isinstance(msg, UserChatMessageV2):
                openai_messages.append(cls._convert_user_message(msg))
            elif isinstance(msg, AssistantChatMessageV2):
                openai_messages.append(cls._convert_assistant_message(msg))
            elif isinstance(msg, ToolChatMessageV2):
                openai_messages.append(cls._convert_tool_message(msg))
            else:  # pragma: no cover - guarded by Pydantic discriminator
                raise ValueError(f"Unsupported Cohere v2 message: {msg!r}")

    @staticmethod
    def _coerce_text_content(content: str | list[Any]) -> str:
        if isinstance(content, str):
            return content
        parts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)

    @classmethod
    def _convert_user_message(cls, msg: UserChatMessageV2) -> dict[str, Any]:
        if isinstance(msg.content, str):
            return {"role": "user", "content": msg.content}

        # Discriminate by ``type`` rather than isinstance so we don't have
        # to import every individual ``*Content`` variant from the SDK -
        # the union (``UserMessageV2Content``) covers both ``TextContent``
        # and ``ImageUrlContent`` and both expose ``type`` as a Literal.
        content_parts: list[dict[str, Any]] = []
        for block in msg.content:
            if block.type == "text":
                content_parts.append({"type": "text", "text": block.text})
            elif block.type == "image_url":
                image_url: dict[str, Any] = {"url": block.image_url.url}
                if getattr(block.image_url, "detail", None) is not None:
                    image_url["detail"] = block.image_url.detail
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": image_url,
                    }
                )
        if len(content_parts) == 1 and content_parts[0]["type"] == "text":
            return {"role": "user", "content": content_parts[0]["text"]}
        return {"role": "user", "content": content_parts}

    @classmethod
    def _convert_assistant_message(cls, msg: AssistantChatMessageV2) -> dict[str, Any]:
        out: dict[str, Any] = {"role": "assistant"}

        # Cohere splits reasoning out into ``thinking`` content blocks. We
        # collapse them back into the OpenAI ``reasoning`` field for the
        # downstream chat template, while text blocks become ``content``.
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        if isinstance(msg.content, str):
            text_parts.append(msg.content)
        elif msg.content is not None:
            for block in msg.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "thinking":
                    thinking_parts.append(block.thinking)

        # ``tool_plan`` is Cohere's chain-of-thought emitted alongside tool
        # calls; preserve it as reasoning so templates that expect a
        # planning block still see it.
        if msg.tool_plan:
            thinking_parts.append(msg.tool_plan)

        if text_parts:
            out["content"] = "".join(text_parts)
        if thinking_parts:
            out["reasoning"] = "".join(thinking_parts)

        if msg.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": (tc.function.name if tc.function else "") or "",
                        "arguments": (tc.function.arguments if tc.function else None)
                        or "{}",
                    },
                }
                for tc in msg.tool_calls
            ]
        return out

    @classmethod
    def _convert_tool_message(cls, msg: ToolChatMessageV2) -> dict[str, Any]:
        if isinstance(msg.content, str):
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
            }

        # When the tool result is text-only, flatten to a string for maximum
        # compatibility with vanilla chat templates. When it includes
        # documents, preserve them as structured content parts so the cohere
        # renderer can surface them as grounding sources (the
        # :class:`CohereRenderer` understands ``{type: document, document:
        # {...}}`` blocks). Non-cohere renderers may not honor document
        # blocks, but that matches the broader "documents are no-op for OSS
        # models" contract documented on this endpoint.
        #
        # Tool message content uses ``ToolMessageV2Content``, which is a
        # union of ``TextToolContent`` and ``DocumentToolContent`` -
        # distinct from the user-message text/image union. We discriminate
        # on the ``type`` literal so we don't have to import each variant.
        has_documents = any(block.type == "document" for block in msg.content)
        if not has_documents:
            text = "\n".join(
                block.text for block in msg.content if block.type == "text"
            )
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": text,
            }

        parts: list[dict[str, Any]] = []
        for block in msg.content:
            if block.type == "text":
                parts.append({"type": "text", "text": block.text})
            elif block.type == "document":
                parts.append(
                    {
                        "type": "document",
                        "document": block.document.model_dump(exclude_none=True),
                    }
                )
        return {
            "role": "tool",
            "tool_call_id": msg.tool_call_id,
            "content": parts,
        }

    @classmethod
    def _build_base_chat_completion(
        cls,
        request: CohereChatV2Request,
        openai_messages: list[dict[str, Any]],
    ) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model=request.model,
            messages=openai_messages,
            max_tokens=request.max_tokens,
            max_completion_tokens=request.max_tokens,
            stop=request.stop_sequences,
            temperature=request.temperature,
            top_p=request.p,
            top_k=request.k,
            seed=request.seed,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            logprobs=request.logprobs,
            priority=request.priority or 0,
            kv_transfer_params=request.kv_transfer_params,
            chat_template_kwargs=request.chat_template_kwargs,
        )

    @classmethod
    def _apply_streaming_options(
        cls,
        chat_req: ChatCompletionRequest,
        request: CohereChatV2Request,
    ) -> None:
        if request.stream:
            chat_req.stream = True
            chat_req.stream_options = StreamOptions.model_validate(
                {"include_usage": True}
            )

    @classmethod
    def _apply_response_format(
        cls,
        chat_req: ChatCompletionRequest,
        request: CohereChatV2Request,
    ) -> None:
        rf = request.response_format
        if rf is None or rf.type == "text":
            return
        chat_req.response_format = ResponseFormat(
            type="json_schema" if rf.json_schema else "json_object",
            json_schema=(
                JsonSchemaResponseFormat(
                    name="cohere_v2_json_schema",
                    json_schema=rf.json_schema,
                )
                if rf.json_schema
                else None
            ),
        )

    @classmethod
    def _apply_tools(
        cls,
        chat_req: ChatCompletionRequest,
        request: CohereChatV2Request,
    ) -> None:
        if not request.tools:
            return
        # Cohere's ``strict_tools`` is the spec-equivalent of OpenAI's
        # per-function ``strict`` flag: when true the API guarantees tool
        # call arguments match the declared JSON schema. We surface it in
        # both places so OpenAI-shaped consumers see it on the function
        # definition and the cohere renderer can read it back from
        # ``chat_template_kwargs`` for cmd3/cmd4 preamble selection.
        strict = bool(request.strict_tools) if request.strict_tools else False
        chat_req.tools = [
            ChatCompletionToolsParam.model_validate(
                {
                    "type": "function",
                    "function": {
                        "name": tool.function.name,
                        "description": tool.function.description,
                        "parameters": tool.function.parameters,
                        **({"strict": True} if strict else {}),
                    },
                }
            )
            for tool in request.tools
        ]

    @classmethod
    def _apply_tool_choice(
        cls,
        chat_req: ChatCompletionRequest,
        request: CohereChatV2Request,
    ) -> None:
        # TODO need to add support for this
        if request.tool_choice == "REQUIRED":
            chat_req.tool_choice = "required"
        elif request.tool_choice == "NONE":
            chat_req.tool_choice = "none"
        elif chat_req.tools:
            # Mirrors Cohere's "free choice" default when tools are present.
            chat_req.tool_choice = "auto"

    @classmethod
    def _apply_cohere_template_kwargs(
        cls,
        chat_req: ChatCompletionRequest,
        request: CohereChatV2Request,
    ) -> None:
        """Forward Cohere-specific request fields into ``chat_template_kwargs``.

        The :class:`vllm.renderers.cohere.CohereRenderer` consumes these
        kwargs to drive ``cohere_melody.render_cmd3`` / ``render_cmd4``.
        Other renderers ignore unknown kwargs, so this is also a no-op for
        non-cohere ``--tokenizer-mode`` settings.
        """
        kwargs = dict(chat_req.chat_template_kwargs or {})

        if request.documents:
            documents: list[dict[str, Any]] = []
            for idx, doc in enumerate(request.documents):
                if isinstance(doc, str):
                    documents.append({"id": f"doc_{idx}", "data": {"text": doc}})
                else:
                    documents.append(
                        {
                            "id": doc.id or f"doc_{idx}",
                            "data": (
                                doc.data
                                if isinstance(doc.data, dict)
                                else {"text": doc.data}
                            ),
                        }
                    )
            kwargs.setdefault("documents", documents)

        if request.safety_mode is not None:
            kwargs.setdefault("safety_mode", str(request.safety_mode).lower())
        if request.citation_options is not None:
            kwargs.setdefault(
                "citation_options",
                request.citation_options.model_dump(exclude_none=True),
            )
        if request.thinking is not None:
            kwargs.setdefault(
                "thinking",
                request.thinking.model_dump(exclude_none=True),
            )

        # ``strict_tools`` is intentionally NOT forwarded here. It's a
        # decoder-guidance flag that ``_apply_tools`` already maps onto
        # per-function OpenAI ``strict: true`` on ``chat_req.tools``;
        # ``cohere_melody.render_cmd3``/``render_cmd4`` have no
        # ``strict_tools`` knob, and the renderer's residual-forward path
        # would otherwise surface it as a Jinja variable ``{{ strict_tools }}``
        # that templates have no defined use for.

        if kwargs:
            chat_req.chat_template_kwargs = kwargs

    # ==================================================================
    # Response conversion: ChatCompletion -> Cohere V2
    # ==================================================================

    def _chat_completion_to_v2(
        self,
        response: ChatCompletionResponse,
        request: CohereChatV2Request,
    ) -> CohereChatV2Response:
        choice = response.choices[0]
        msg = choice.message

        # Build content blocks as dicts; ``AssistantMessageResponse``
        # validates them into the proper discriminated union variants
        # (``Text/ThinkingAssistantMessageResponseContentItem``).
        content_blocks: list[dict[str, Any]] = []
        if msg.reasoning:
            content_blocks.append(
                {"type": ContentBlockType.THINKING, "thinking": msg.reasoning}
            )
        if msg.content:
            content_blocks.append({"type": ContentBlockType.TEXT, "text": msg.content})

        tool_calls: list[ToolCallV2] | None = None
        if msg.tool_calls:
            tool_calls = [
                ToolCallV2(
                    id=tc.id,
                    function=ToolCallV2Function(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                )
                for tc in msg.tool_calls
            ]

        # Cohere's ``tool_plan`` is the planning text emitted before tool
        # calls on older, non-reasoning Command models. For those models
        # we surface ``reasoning`` as ``tool_plan`` and drop the thinking
        # block. Reasoning Command models emit a regular thinking block
        # alongside tool calls, so for them we leave the thinking block
        # in place and never set ``tool_plan``.
        tool_plan: str | None = None
        if not self._is_reasoning_model and tool_calls and msg.reasoning:
            tool_plan = msg.reasoning
            content_blocks = [
                blk
                for blk in content_blocks
                if blk.get("type") != ContentBlockType.THINKING
            ]

        assistant_msg = AssistantMessageResponse(
            content=content_blocks or None,
            tool_calls=tool_calls,
            tool_plan=tool_plan,
            citations=self._extract_citations_if_any(msg),
        )

        usage = self._build_usage(response)

        return CohereChatV2Response(
            id=response.id or f"chat_{int(time.time() * 1000)}",
            finish_reason=_map_finish_reason(choice.finish_reason),
            message=assistant_msg,
            usage=usage,
            kv_transfer_params=response.kv_transfer_params,
        )

    @staticmethod
    def _extract_citations_if_any(msg: Any) -> list[Citation] | None:
        """Coerce ``ChatMessage.citations`` into the Cohere v2 wire shape.

        ``ChatMessage`` natively carries a ``citations: list[Citation] |
        None`` field (see :mod:`vllm.entrypoints.openai.engine.protocol`).
        Renderers / parsers (the ``cohere_command3`` and
        ``cohere_command4`` reasoning parsers) populate it. We map each
        :class:`vllm...Citation` into the SDK :class:`cohere.types.Citation`
        wire model, preserving sources and span.
        """
        raw = getattr(msg, "citations", None)
        if not raw:
            return None
        out: list[Citation] = []
        for c in raw:
            try:
                if isinstance(c, Citation):
                    out.append(c)
                    continue
                if hasattr(c, "model_dump"):
                    payload = c.model_dump(exclude_none=True)
                else:
                    payload = c
                if isinstance(payload, dict):
                    out.append(Citation.model_validate(payload))
            except Exception:  # pragma: no cover - defensive
                logger.warning("Skipping malformed citation: %r", c, exc_info=True)
        return out or None

    @staticmethod
    def _build_usage(response: ChatCompletionResponse) -> CohereUsage | None:
        if response.usage is None:
            return None
        prompt = response.usage.prompt_tokens
        completion = response.usage.completion_tokens or 0
        cached: int | None = None
        if response.usage.prompt_tokens_details is not None:
            cached = response.usage.prompt_tokens_details.cached_tokens
        return CohereUsage(
            billed_units=CohereUsageBilledUnits(
                input_tokens=prompt,
                output_tokens=completion,
            ),
            tokens=CohereUsageTokens(
                input_tokens=prompt,
                output_tokens=completion,
            ),
            cached_tokens=cached,
        )

    # ==================================================================
    # Stream conversion: chat completion stream -> Cohere V2 SSE events
    # ==================================================================

    async def _chat_completion_stream_to_v2(
        self,
        generator: AsyncGenerator[str, None],
        request: CohereChatV2Request,
    ) -> AsyncGenerator[str, None]:
        """Translate an OpenAI-style chat completion SSE stream into Cohere's
        v2 stream-event format.

        Cohere's v2 stream lifecycle is:

            message-start
              [content-start, content-delta..., content-end]*
              [tool-plan-delta]*
              [tool-call-start, tool-call-delta..., tool-call-end]*
              message-end
        """
        state = _StreamState()

        try:
            async for item in generator:
                if not item.startswith("data:"):
                    continue
                data_str = item[len("data:") :].strip().rstrip("\n")
                if not data_str:
                    continue
                if data_str == "[DONE]":
                    # OpenAI's stream terminator. Fall through to the
                    # post-loop cleanup so we always emit ``message-end``
                    # even if the usage-only chunk was skipped.
                    break

                chunk = ChatCompletionStreamResponse.model_validate_json(data_str)
                state.last_chunk_id = chunk.id

                if not state.started:
                    yield _emit(
                        MessageStartEvent(
                            id=chunk.id,
                            delta={"message": {"role": "assistant"}},
                        )
                    )
                    state.started = True

                # The final OpenAI chunk has no choices and only carries usage.
                if not chunk.choices:
                    for ev in self._close_open_blocks(state):
                        yield ev
                    yield self._build_message_end_event(
                        chunk_id=chunk.id,
                        finish_reason=state.finish_reason,
                        usage_chunk=chunk,
                    )
                    state.ended = True
                    continue

                choice = chunk.choices[0]
                if choice.finish_reason is not None:
                    state.finish_reason = choice.finish_reason

                delta = choice.delta

                # Reasoning -> thinking content block
                reasoning = getattr(delta, "reasoning", None) or getattr(
                    delta, "reasoning_content", None
                )
                if reasoning:
                    for ev in self._handle_thinking_delta(state, reasoning):
                        yield ev

                if delta.content:
                    for ev in self._handle_text_delta(state, delta.content):
                        yield ev

                if delta.tool_calls:
                    for ev in self._handle_tool_call_deltas(state, delta.tool_calls):
                        yield ev

                # Citations: a Cohere-specific extension on DeltaMessage that
                # the cohere renderer/parsers may populate.
                delta_citations = getattr(delta, "citations", None)
                if delta_citations:
                    for ev in self._handle_citation_deltas(state, delta_citations):
                        yield ev

        except Exception as exc:
            logger.exception("Error converting chat completion stream to v2")
            if state.started and not state.ended:
                yield _sse(
                    json.dumps(
                        {
                            "type": "message-end",
                            "delta": {
                                "error": str(exc),
                                "finish_reason": "ERROR",
                            },
                        }
                    )
                )
                state.ended = True
            yield _DONE_FRAME
            return

        # Normal completion or ``[DONE]``: ensure ``message-end`` is always
        # emitted. Upstream may close the stream without sending the final
        # usage-only chunk (e.g. on shutdown, or when ``[DONE]`` is the only
        # terminator); without this fallback Cohere clients would hang
        # waiting for the closing event.
        if state.started and not state.ended:
            for ev in self._close_open_blocks(state):
                yield ev
            yield self._build_message_end_event(
                chunk_id=state.last_chunk_id,
                finish_reason=state.finish_reason,
                usage_chunk=None,
            )
            state.ended = True

        # Stream terminator. Cohere's v2 SSE protocol ends every stream
        # with ``data: [DONE]\n\n`` after ``message-end``; Fern-generated
        # clients (Go/Java) and cohere-python all key their read loop off
        # this sentinel.
        yield _DONE_FRAME

    # -- per-delta helpers --------------------------------------------

    def _handle_thinking_delta(self, state: _StreamState, delta_text: str) -> list[str]:
        # Non-reasoning Command models: emit ``tool-plan-delta`` events
        # directly instead of opening a thinking content block. The
        # ``tool-plan-delta`` event has no start/end pair around it.
        if not self._is_reasoning_model:
            events: list[str] = list(self._close_open_blocks(state))
            events.append(
                _emit(
                    ToolPlanDeltaEvent(
                        delta={"message": {"tool_plan": delta_text}},
                    )
                )
            )
            return events

        # Reasoning model (default): open / continue a thinking block.
        events = []
        if state.active_block != ContentBlockType.THINKING:
            events.extend(self._close_open_blocks(state))
            idx = state.next_content_index()
            state.active_block = ContentBlockType.THINKING
            state.active_block_index = idx
            events.append(
                _emit(
                    ContentStartEvent(
                        index=idx,
                        delta={
                            "message": {
                                "content": {
                                    "type": ContentBlockType.THINKING,
                                    "thinking": "",
                                }
                            }
                        },
                    )
                )
            )
        events.append(
            _emit(
                ContentDeltaEvent(
                    index=state.active_block_index,
                    delta={"message": {"content": {"thinking": delta_text}}},
                )
            )
        )
        return events

    def _handle_text_delta(self, state: _StreamState, delta_text: str) -> list[str]:
        events: list[str] = []
        if state.active_block != ContentBlockType.TEXT:
            events.extend(self._close_open_blocks(state))
            idx = state.next_content_index()
            state.active_block = ContentBlockType.TEXT
            state.active_block_index = idx
            events.append(
                _emit(
                    ContentStartEvent(
                        index=idx,
                        delta={
                            "message": {
                                "content": {
                                    "type": ContentBlockType.TEXT,
                                    "text": "",
                                }
                            }
                        },
                    )
                )
            )
        events.append(
            _emit(
                ContentDeltaEvent(
                    index=state.active_block_index,
                    delta={"message": {"content": {"text": delta_text}}},
                )
            )
        )
        return events

    def _handle_tool_call_deltas(self, state: _StreamState, deltas: list) -> list[str]:
        events: list[str] = []
        for tc in deltas:
            tc_index = tc.index
            fn = tc.function

            if tc_index not in state.tool_calls_seen:
                # New tool call. Close any open content/tool block first.
                events.extend(self._close_open_blocks(state))
                state.tool_calls_seen.add(tc_index)
                state.active_tool_index = tc_index
                state.active_block = ContentBlockType.TOOL_CALL
                events.append(
                    _emit(
                        ToolCallStartEvent(
                            index=tc_index,
                            delta={
                                "message": {
                                    "tool_calls": {
                                        "id": tc.id or "",
                                        "type": "function",
                                        "function": {
                                            "name": (fn.name if fn else "") or "",
                                            "arguments": (fn.arguments if fn else None)
                                            or "",
                                        },
                                    }
                                }
                            },
                        )
                    )
                )
                continue

            if fn and fn.arguments:
                events.append(
                    _emit(
                        ToolCallDeltaEvent(
                            index=tc_index,
                            delta={
                                "message": {
                                    "tool_calls": {
                                        "function": {
                                            "arguments": fn.arguments,
                                        }
                                    }
                                }
                            },
                        )
                    )
                )
        return events

    def _handle_citation_deltas(
        self, state: _StreamState, citations: list
    ) -> list[str]:
        """Emit ``citation-start`` / ``citation-end`` events for a delta.

        The cohere renderer/parsers populate ``DeltaMessage.citations`` with
        :class:`vllm.entrypoints.openai.engine.protocol.Citation` instances
        once a citation has fully resolved (start/end indices known). We
        emit a complete start+end pair for each citation in the delta so
        Cohere clients can attach the annotation to the surrounding text.
        """
        events: list[str] = []
        for c in citations:
            payload = c.model_dump(exclude_none=True) if hasattr(c, "model_dump") else c
            if not isinstance(payload, dict):
                continue
            try:
                citation = Citation.model_validate(payload)
            except Exception:  # pragma: no cover - defensive
                logger.warning(
                    "Skipping malformed streamed citation: %r",
                    payload,
                    exc_info=True,
                )
                continue
            idx = state.next_citation_index()
            events.append(
                _emit(
                    CitationStartEvent(
                        index=idx,
                        delta={
                            "message": {
                                "citations": citation.model_dump(exclude_none=True)
                            }
                        },
                    )
                )
            )
            events.append(_emit(CitationEndEvent(index=idx)))
        return events

    # -- block lifecycle helpers --------------------------------------

    def _close_open_blocks(self, state: _StreamState) -> list[str]:
        """Emit ``content-end`` / ``tool-call-end`` for the currently open
        block (if any) and reset the corresponding ``_StreamState`` slots.
        """
        events: list[str] = []
        if state.active_block in (ContentBlockType.TEXT, ContentBlockType.THINKING):
            events.append(_emit(ContentEndEvent(index=state.active_block_index)))
        elif state.active_block == ContentBlockType.TOOL_CALL:
            events.append(_emit(ToolCallEndEvent(index=state.active_tool_index)))
        state.active_block = None
        state.active_block_index = None
        state.active_tool_index = None
        return events

    def _build_message_end_event(
        self,
        chunk_id: str,
        finish_reason: str | None,
        usage_chunk: ChatCompletionStreamResponse | None = None,
    ) -> str:
        delta: dict[str, Any] = {
            "finish_reason": _map_finish_reason(finish_reason),
        }
        if usage_chunk is not None and usage_chunk.usage is not None:
            prompt = usage_chunk.usage.prompt_tokens
            completion = usage_chunk.usage.completion_tokens or 0
            usage_block: dict[str, Any] = {
                "billed_units": {
                    "input_tokens": prompt,
                    "output_tokens": completion,
                },
                "tokens": {
                    "input_tokens": prompt,
                    "output_tokens": completion,
                },
            }
            if usage_chunk.usage.prompt_tokens_details is not None:
                cached = usage_chunk.usage.prompt_tokens_details.cached_tokens
                if cached is not None:
                    usage_block["cached_tokens"] = cached
            delta["usage"] = usage_block
        return _emit(MessageEndEvent(id=chunk_id, delta=delta))

    # ==================================================================
    # Helpers for the router
    # ==================================================================

    @staticmethod
    def create_error_response(
        message: str | Exception,
        err_type: str = "bad_request",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> ErrorResponse:
        # Override of :meth:`BaseServing.create_error_response` that uses
        # Cohere-flavored defaults (``type="bad_request"``, ``code=400``)
        # so the router can translate the envelope uniformly. ``param``
        # is accepted for signature parity with the base class but is
        # not surfaced in the Cohere wire format.
        from vllm.entrypoints.openai.engine.protocol import ErrorInfo

        del param  # unused; kept for signature compatibility
        return ErrorResponse(
            error=ErrorInfo(
                message=str(message),
                type=err_type,
                code=int(status_code),
            )
        )


# ---------------------------------------------------------------------------
# Stream state
# ---------------------------------------------------------------------------


class _StreamState:
    """Tracks which Cohere v2 stream block (if any) is currently open."""

    def __init__(self) -> None:
        self.started: bool = False
        self.ended: bool = False
        self.finish_reason: str | None = None
        self.last_chunk_id: str = ""
        self.active_block: ContentBlockType | None = None
        self.active_block_index: int | None = None
        self.active_tool_index: int | None = None
        self._next_index: int = 0
        self._next_citation_index: int = 0
        self.tool_calls_seen: set[int] = set()

    def next_content_index(self) -> int:
        idx = self._next_index
        self._next_index += 1
        return idx

    def next_citation_index(self) -> int:
        idx = self._next_citation_index
        self._next_citation_index += 1
        return idx
