# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/anthropic/serving.py
"""Cohere Chat v2 API serving handler.

Implements ``POST /v2/chat`` by translating the incoming Cohere v2 request
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
from typing import TYPE_CHECKING, Any

from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.cohere.protocol import (
    CohereAssistantMessageResponse,
    CohereAssistantMessageV2,
    CohereChatContentDeltaEvent,
    CohereChatContentEndEvent,
    CohereChatContentStartEvent,
    CohereChatMessageEndEvent,
    CohereChatMessageStartEvent,
    CohereChatToolCallDeltaEvent,
    CohereChatToolCallEndEvent,
    CohereChatToolCallStartEvent,
    CohereChatV2Request,
    CohereChatV2Response,
    CohereCitation,
    CohereCitationEndEvent,
    CohereCitationStartEvent,
    CohereFinishReason,
    CohereSystemMessageV2,
    CohereTextContent,
    CohereThinkingContent,
    CohereToolCallFunction,
    CohereToolCallV2,
    CohereToolMessageV2,
    CohereUsage,
    CohereUsageBilledUnits,
    CohereUsageTokens,
    CohereUserMessageV2,
)
from vllm.entrypoints.logger import RequestLogger
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

if TYPE_CHECKING:
    from vllm.entrypoints.serve.render.serving import OpenAIServingRender

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
    """Handler for the Cohere Chat v2 API (``POST /v2/chat``).

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
        openai_serving_render: OpenAIServingRender,
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
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            response_role=response_role,
            openai_serving_render=openai_serving_render,
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

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def create_chat_v2(
        self,
        request: CohereChatV2Request,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | CohereChatV2Response | ErrorResponse:
        """Implements ``POST /v2/chat``."""
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
            if isinstance(msg, CohereSystemMessageV2):
                openai_messages.append(
                    {
                        "role": "system",
                        "content": cls._coerce_text_content(msg.content),
                    }
                )
            elif isinstance(msg, CohereUserMessageV2):
                openai_messages.append(cls._convert_user_message(msg))
            elif isinstance(msg, CohereAssistantMessageV2):
                openai_messages.append(cls._convert_assistant_message(msg))
            elif isinstance(msg, CohereToolMessageV2):
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
    def _convert_user_message(
        cls, msg: CohereUserMessageV2
    ) -> dict[str, Any]:
        if isinstance(msg.content, str):
            return {"role": "user", "content": msg.content}

        content_parts: list[dict[str, Any]] = []
        for block in msg.content:
            if isinstance(block, CohereTextContent):
                content_parts.append({"type": "text", "text": block.text})
            else:  # CohereImageContent
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": block.image_url.url},
                    }
                )
        if (
            len(content_parts) == 1
            and content_parts[0]["type"] == "text"
        ):
            return {"role": "user", "content": content_parts[0]["text"]}
        return {"role": "user", "content": content_parts}

    @classmethod
    def _convert_assistant_message(
        cls, msg: CohereAssistantMessageV2
    ) -> dict[str, Any]:
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
                if isinstance(block, CohereTextContent):
                    text_parts.append(block.text)
                elif isinstance(block, CohereThinkingContent):
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
                        "arguments": (
                            tc.function.arguments if tc.function else None
                        )
                        or "{}",
                    },
                }
                for tc in msg.tool_calls
            ]
        return out

    @classmethod
    def _convert_tool_message(
        cls, msg: CohereToolMessageV2
    ) -> dict[str, Any]:
        if isinstance(msg.content, str):
            content: Any = msg.content
        else:
            text_parts: list[str] = []
            for block in msg.content:
                if isinstance(block, CohereTextContent):
                    text_parts.append(block.text)
                else:  # CohereDocumentContent
                    # The OpenAI chat-completion shape only knows about
                    # text tool results. Serialize the document so its data
                    # is preserved in the prompt.
                    text_parts.append(
                        json.dumps(block.document.model_dump(exclude_none=True))
                    )
            content = "\n".join(text_parts)
        return {
            "role": "tool",
            "tool_call_id": msg.tool_call_id,
            "content": content,
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
                {"include_usage": True, "continuous_usage_stats": True}
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
        chat_req.tools = [
            ChatCompletionToolsParam.model_validate(
                {
                    "type": "function",
                    "function": {
                        "name": tool.function.name,
                        "description": tool.function.description,
                        "parameters": tool.function.parameters,
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

        content_blocks: list[Any] = []
        if msg.reasoning:
            content_blocks.append(CohereThinkingContent(thinking=msg.reasoning))
        if msg.content:
            content_blocks.append(CohereTextContent(text=msg.content))

        tool_calls: list[CohereToolCallV2] | None = None
        if msg.tool_calls:
            tool_calls = [
                CohereToolCallV2(
                    id=tc.id,
                    function=CohereToolCallFunction(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                )
                for tc in msg.tool_calls
            ]

        # Cohere's ``tool_plan`` is the planning text emitted before tool
        # calls. The closest analogue in chat completions is the reasoning
        # field; if a tool call is present, surface reasoning as tool_plan
        # rather than as a thinking content block to match Cohere clients.
        tool_plan: str | None = None
        if tool_calls and msg.reasoning:
            tool_plan = msg.reasoning
            content_blocks = [
                blk
                for blk in content_blocks
                if not isinstance(blk, CohereThinkingContent)
            ]

        assistant_msg = CohereAssistantMessageResponse(
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
    def _extract_citations_if_any(msg: Any) -> list[CohereCitation] | None:
        """Coerce ``ChatMessage.citations`` into the Cohere v2 wire shape.

        ``ChatMessage`` natively carries a ``citations: list[Citation] |
        None`` field (see :mod:`vllm.entrypoints.openai.engine.protocol`).
        Renderers/parsers (e.g. the ``cohere2`` reasoning parser) populate
        it. We map each :class:`vllm...Citation` into the
        :class:`CohereCitation` wire model, preserving sources and span.
        """
        raw = getattr(msg, "citations", None)
        if not raw:
            return None
        out: list[CohereCitation] = []
        for c in raw:
            try:
                if isinstance(c, CohereCitation):
                    out.append(c)
                    continue
                if hasattr(c, "model_dump"):
                    c = c.model_dump(exclude_none=True)
                if isinstance(c, dict):
                    out.append(CohereCitation.model_validate(c))
            except Exception:  # pragma: no cover - defensive
                logger.debug("Skipping malformed citation: %r", c, exc_info=True)
        return out or None

    @staticmethod
    def _build_usage(response: ChatCompletionResponse) -> CohereUsage | None:
        if response.usage is None:
            return None
        prompt = response.usage.prompt_tokens
        completion = response.usage.completion_tokens or 0
        return CohereUsage(
            billed_units=CohereUsageBilledUnits(
                input_tokens=prompt,
                output_tokens=completion,
            ),
            tokens=CohereUsageTokens(
                input_tokens=prompt,
                output_tokens=completion,
            ),
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
                data_str = item[len("data:"):].strip().rstrip("\n")
                if not data_str:
                    continue
                if data_str == "[DONE]":
                    # Cohere closes the stream with message-end (already
                    # emitted on the final usage chunk); nothing to do.
                    return

                chunk = ChatCompletionStreamResponse.model_validate_json(data_str)

                if not state.started:
                    yield _sse(
                        CohereChatMessageStartEvent(
                            id=chunk.id,
                            delta={"message": {"role": "assistant"}},
                        ).model_dump_json(exclude_none=True)
                    )
                    state.started = True

                # The final OpenAI chunk has no choices and only carries usage.
                if not chunk.choices:
                    yield from self._close_open_blocks(state)
                    yield _sse(
                        self._build_message_end_event(
                            chunk_id=chunk.id,
                            finish_reason=state.finish_reason,
                            usage_chunk=chunk,
                        )
                    )
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
                    for ev in self._handle_tool_call_deltas(
                        state, delta.tool_calls
                    ):
                        yield ev

                # Citations: a Cohere-specific extension on DeltaMessage that
                # the cohere renderer/parsers may populate.
                if getattr(delta, "citations", None):
                    for ev in self._handle_citation_deltas(
                        state, delta.citations
                    ):
                        yield ev

        except Exception as exc:
            logger.exception("Error converting chat completion stream to v2")
            yield _sse(
                json.dumps(
                    {
                        "type": "message-end",
                        "delta": {"error": str(exc), "finish_reason": "ERROR"},
                    }
                )
            )

    # -- per-delta helpers --------------------------------------------

    def _handle_thinking_delta(
        self, state: _StreamState, delta_text: str
    ) -> list[str]:
        events: list[str] = []
        if state.active_block != "thinking":
            events.extend(self._close_open_blocks_inner(state))
            idx = state.next_content_index()
            state.active_block = "thinking"
            state.active_block_index = idx
            events.append(
                _sse(
                    CohereChatContentStartEvent(
                        index=idx,
                        delta={
                            "message": {
                                "content": {"type": "thinking", "thinking": ""}
                            }
                        },
                    ).model_dump_json(exclude_none=True)
                )
            )
        events.append(
            _sse(
                CohereChatContentDeltaEvent(
                    index=state.active_block_index,
                    delta={
                        "message": {"content": {"thinking": delta_text}}
                    },
                ).model_dump_json(exclude_none=True)
            )
        )
        return events

    def _handle_text_delta(
        self, state: _StreamState, delta_text: str
    ) -> list[str]:
        events: list[str] = []
        if state.active_block != "text":
            events.extend(self._close_open_blocks_inner(state))
            idx = state.next_content_index()
            state.active_block = "text"
            state.active_block_index = idx
            events.append(
                _sse(
                    CohereChatContentStartEvent(
                        index=idx,
                        delta={
                            "message": {
                                "content": {"type": "text", "text": ""}
                            }
                        },
                    ).model_dump_json(exclude_none=True)
                )
            )
        events.append(
            _sse(
                CohereChatContentDeltaEvent(
                    index=state.active_block_index,
                    delta={"message": {"content": {"text": delta_text}}},
                ).model_dump_json(exclude_none=True)
            )
        )
        return events

    def _handle_tool_call_deltas(
        self, state: _StreamState, deltas: list
    ) -> list[str]:
        events: list[str] = []
        for tc in deltas:
            tc_index = tc.index
            tool_id = tc.id
            fn = tc.function

            if tool_id is not None and tc_index not in state.tool_calls_seen:
                # New tool call. Close any open content/tool block first.
                events.extend(self._close_open_blocks_inner(state))
                state.tool_calls_seen.add(tc_index)
                state.active_tool_index = tc_index
                state.active_block = "tool_call"
                events.append(
                    _sse(
                        CohereChatToolCallStartEvent(
                            index=tc_index,
                            delta={
                                "message": {
                                    "tool_calls": {
                                        "id": tool_id,
                                        "type": "function",
                                        "function": {
                                            "name": (fn.name if fn else "")
                                            or "",
                                            "arguments": (
                                                fn.arguments if fn else None
                                            )
                                            or "",
                                        },
                                    }
                                }
                            },
                        ).model_dump_json(exclude_none=True)
                    )
                )
                continue

            if fn and fn.arguments:
                events.append(
                    _sse(
                        CohereChatToolCallDeltaEvent(
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
                        ).model_dump_json(exclude_none=True)
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
            payload = (
                c.model_dump(exclude_none=True) if hasattr(c, "model_dump") else c
            )
            if not isinstance(payload, dict):
                continue
            try:
                citation = CohereCitation.model_validate(payload)
            except Exception:  # pragma: no cover - defensive
                logger.debug("Skipping malformed streamed citation: %r", payload)
                continue
            idx = state.next_citation_index()
            events.append(
                _sse(
                    CohereCitationStartEvent(
                        index=idx,
                        delta={"message": {"citations": citation.model_dump(
                            exclude_none=True
                        )}},
                    ).model_dump_json(exclude_none=True)
                )
            )
            events.append(
                _sse(
                    CohereCitationEndEvent(index=idx).model_dump_json(
                        exclude_none=True
                    )
                )
            )
        return events

    # -- block lifecycle helpers --------------------------------------

    def _close_open_blocks(self, state: _StreamState) -> list[str]:
        return self._close_open_blocks_inner(state)

    def _close_open_blocks_inner(self, state: _StreamState) -> list[str]:
        events: list[str] = []
        if state.active_block in ("text", "thinking"):
            events.append(
                _sse(
                    CohereChatContentEndEvent(
                        index=state.active_block_index,
                    ).model_dump_json(exclude_none=True)
                )
            )
        elif state.active_block == "tool_call":
            events.append(
                _sse(
                    CohereChatToolCallEndEvent(
                        index=state.active_tool_index,
                    ).model_dump_json(exclude_none=True)
                )
            )
        state.active_block = None
        state.active_block_index = None
        state.active_tool_index = None
        return events

    def _build_message_end_event(
        self,
        chunk_id: str,
        finish_reason: str | None,
        usage_chunk: ChatCompletionStreamResponse,
    ) -> str:
        delta: dict[str, Any] = {
            "finish_reason": _map_finish_reason(finish_reason),
        }
        if usage_chunk.usage is not None:
            prompt = usage_chunk.usage.prompt_tokens
            completion = usage_chunk.usage.completion_tokens or 0
            delta["usage"] = {
                "billed_units": {
                    "input_tokens": prompt,
                    "output_tokens": completion,
                },
                "tokens": {
                    "input_tokens": prompt,
                    "output_tokens": completion,
                },
            }
        return CohereChatMessageEndEvent(
            id=chunk_id, delta=delta
        ).model_dump_json(exclude_none=True)

    # ==================================================================
    # Helpers for the router
    # ==================================================================

    def create_error_response(self, message: str) -> ErrorResponse:
        # Reuse the OpenAI engine's error response so that the router can
        # translate it into Cohere's error envelope uniformly.
        from vllm.entrypoints.openai.engine.protocol import ErrorInfo

        return ErrorResponse(
            error=ErrorInfo(
                message=message,
                type="bad_request",
                code=400,
            )
        )


# ---------------------------------------------------------------------------
# Stream state
# ---------------------------------------------------------------------------


class _StreamState:
    """Tracks which Cohere v2 stream block (if any) is currently open."""

    def __init__(self) -> None:
        self.started: bool = False
        self.finish_reason: str | None = None
        self.active_block: str | None = None  # "text" | "thinking" | "tool_call"
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
