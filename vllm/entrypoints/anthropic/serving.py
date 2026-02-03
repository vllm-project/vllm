# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/vllm/vllm/entrypoints/openai/serving_chat.py

"""Anthropic Messages API serving handler"""

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicDelta,
    AnthropicError,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicStreamEvent,
    AnthropicUsage,
)
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    StreamOptions,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels

logger = logging.getLogger(__name__)


def wrap_data_with_event(data: str, event: str):
    return f"event: {event}\ndata: {data}\n\n"


class AnthropicServingMessages(OpenAIServingChat):
    """Handler for Anthropic Messages API requests"""

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        response_role: str,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        tool_parser: str | None = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            response_role=response_role,
            request_logger=request_logger,
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
            reasoning_parser=reasoning_parser,
            enable_auto_tools=enable_auto_tools,
            tool_parser=tool_parser,
            enable_prompt_tokens_details=enable_prompt_tokens_details,
            enable_force_include_usage=enable_force_include_usage,
        )
        self.stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
        }

    def _convert_anthropic_to_openai_request(
        self, anthropic_request: AnthropicMessagesRequest
    ) -> ChatCompletionRequest:
        """Convert Anthropic message format to OpenAI format"""
        openai_messages = []

        # Add system message if provided
        if anthropic_request.system:
            if isinstance(anthropic_request.system, str):
                openai_messages.append(
                    {"role": "system", "content": anthropic_request.system}
                )
            else:
                system_prompt = ""
                for block in anthropic_request.system:
                    if block.type == "text" and block.text:
                        system_prompt += block.text
                openai_messages.append({"role": "system", "content": system_prompt})

        for msg in anthropic_request.messages:
            openai_msg: dict[str, Any] = {"role": msg.role}  # type: ignore
            if isinstance(msg.content, str):
                openai_msg["content"] = msg.content
            else:
                # Handle complex content blocks
                content_parts: list[dict[str, Any]] = []
                tool_calls: list[dict[str, Any]] = []
                reasoning_parts: list[str] = []

                for block in msg.content:
                    if block.type == "text" and block.text:
                        content_parts.append({"type": "text", "text": block.text})
                    elif block.type == "image" and block.source:
                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": block.source.get("data", "")},
                            }
                        )
                    elif block.type == "thinking" and block.thinking is not None:
                        reasoning_parts.append(block.thinking)
                    elif block.type == "tool_use":
                        # Convert tool use to function call format
                        tool_call = {
                            "id": block.id or f"call_{int(time.time())}",
                            "type": "function",
                            "function": {
                                "name": block.name or "",
                                "arguments": json.dumps(block.input or {}),
                            },
                        }
                        tool_calls.append(tool_call)
                    elif block.type == "tool_result":
                        if msg.role == "user":
                            openai_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block.id or "",
                                    "content": str(block.content)
                                    if block.content
                                    else "",
                                }
                            )
                        else:
                            # Assistant tool result becomes regular text
                            tool_result_text = (
                                str(block.content) if block.content else ""
                            )
                            content_parts.append(
                                {
                                    "type": "text",
                                    "text": f"Tool result: {tool_result_text}",
                                }
                            )

                if reasoning_parts:
                    openai_msg["reasoning"] = "".join(reasoning_parts)

                # Add tool calls to the message if any
                if tool_calls:
                    openai_msg["tool_calls"] = tool_calls  # type: ignore

                # Add content parts if any
                if content_parts:
                    if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                        openai_msg["content"] = content_parts[0]["text"]
                    else:
                        openai_msg["content"] = content_parts  # type: ignore
                elif not tool_calls and not reasoning_parts:
                    continue

            openai_messages.append(openai_msg)

        req = ChatCompletionRequest(
            model=anthropic_request.model,
            messages=openai_messages,
            max_tokens=anthropic_request.max_tokens,
            max_completion_tokens=anthropic_request.max_tokens,
            stop=anthropic_request.stop_sequences,
            temperature=anthropic_request.temperature,
            top_p=anthropic_request.top_p,
            top_k=anthropic_request.top_k,
        )

        if anthropic_request.stream:
            req.stream = anthropic_request.stream
            req.stream_options = StreamOptions.validate(
                {"include_usage": True, "continuous_usage_stats": True}
            )

        if anthropic_request.tool_choice is None:
            req.tool_choice = None
        elif anthropic_request.tool_choice.type == "auto":
            req.tool_choice = "auto"
        elif anthropic_request.tool_choice.type == "any":
            req.tool_choice = "required"
        elif anthropic_request.tool_choice.type == "tool":
            req.tool_choice = ChatCompletionNamedToolChoiceParam.model_validate(
                {
                    "type": "function",
                    "function": {"name": anthropic_request.tool_choice.name},
                }
            )

        tools = []
        if anthropic_request.tools is None:
            return req
        for tool in anthropic_request.tools:
            tools.append(
                ChatCompletionToolsParam.model_validate(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        },
                    }
                )
            )
        if req.tool_choice is None:
            req.tool_choice = "auto"
        req.tools = tools
        return req

    async def create_messages(
        self,
        request: AnthropicMessagesRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | AnthropicMessagesResponse | ErrorResponse:
        """
        Messages API similar to Anthropic's API.

        See https://docs.anthropic.com/en/api/messages
        for the API specification. This API mimics the Anthropic messages API.
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Received messages request %s", request.model_dump_json())
        chat_req = self._convert_anthropic_to_openai_request(request)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Convert to OpenAI request %s", chat_req.model_dump_json())
        generator = await self.create_chat_completion(chat_req, raw_request)

        if isinstance(generator, ErrorResponse):
            return generator

        elif isinstance(generator, ChatCompletionResponse):
            return self.messages_full_converter(generator)

        return self.message_stream_converter(generator)

    def messages_full_converter(
        self,
        generator: ChatCompletionResponse,
    ) -> AnthropicMessagesResponse:
        result = AnthropicMessagesResponse(
            id=generator.id,
            content=[],
            model=generator.model,
            usage=AnthropicUsage(
                input_tokens=generator.usage.prompt_tokens,
                output_tokens=generator.usage.completion_tokens,
            ),
        )
        choice = generator.choices[0]
        if choice.finish_reason == "stop":
            result.stop_reason = "end_turn"
        elif choice.finish_reason == "length":
            result.stop_reason = "max_tokens"
        elif choice.finish_reason == "tool_calls":
            result.stop_reason = "tool_use"

        content: list[AnthropicContentBlock] = []
        if choice.message.reasoning:
            content.append(
                AnthropicContentBlock(
                    type="thinking",
                    thinking=choice.message.reasoning,
                    signature=uuid.uuid4().hex,
                )
            )
        if choice.message.content:
            content.append(
                AnthropicContentBlock(
                    type="text",
                    text=choice.message.content,
                )
            )

        for tool_call in choice.message.tool_calls:
            anthropic_tool_call = AnthropicContentBlock(
                type="tool_use",
                id=tool_call.id,
                name=tool_call.function.name,
                input=json.loads(tool_call.function.arguments),
            )
            content += [anthropic_tool_call]

        result.content = content

        return result

    async def message_stream_converter(
        self,
        generator: AsyncGenerator[str, None],
    ) -> AsyncGenerator[str, None]:
        try:
            first_item = True
            finish_reason = None
            content_block_index = 0
            active_block_type: str | None = None
            active_block_index: int | None = None
            active_block_signature: str | None = None
            signature_emitted = False
            active_tool_use_id: str | None = None

            def stop_active_block():
                nonlocal active_block_type, active_block_index, content_block_index
                nonlocal active_block_signature, signature_emitted, active_tool_use_id
                events: list[str] = []
                if active_block_type is None:
                    return events
                if (
                    active_block_type == "thinking"
                    and active_block_signature is not None
                    and not signature_emitted
                ):
                    chunk = AnthropicStreamEvent(
                        index=active_block_index,
                        type="content_block_delta",
                        delta=AnthropicDelta(
                            type="signature_delta",
                            signature=active_block_signature,
                        ),
                    )
                    data = chunk.model_dump_json(exclude_unset=True)
                    events.append(wrap_data_with_event(data, "content_block_delta"))
                    signature_emitted = True
                stop_chunk = AnthropicStreamEvent(
                    index=active_block_index,
                    type="content_block_stop",
                )
                data = stop_chunk.model_dump_json(exclude_unset=True)
                events.append(wrap_data_with_event(data, "content_block_stop"))
                active_block_type = None
                active_block_index = None
                active_block_signature = None
                signature_emitted = False
                active_tool_use_id = None
                content_block_index += 1
                return events

            def start_block(block: AnthropicContentBlock):
                nonlocal active_block_type, active_block_index, content_block_index
                nonlocal active_block_signature, signature_emitted, active_tool_use_id
                chunk = AnthropicStreamEvent(
                    index=content_block_index,
                    type="content_block_start",
                    content_block=block,
                )
                data = chunk.model_dump_json(exclude_unset=True)
                event = wrap_data_with_event(data, "content_block_start")
                active_block_type = block.type
                active_block_index = content_block_index
                if block.type == "thinking":
                    active_block_signature = uuid.uuid4().hex
                    signature_emitted = False
                    active_tool_use_id = None
                elif block.type == "tool_use":
                    active_block_signature = None
                    signature_emitted = True
                    active_tool_use_id = block.id
                else:
                    active_block_signature = None
                    signature_emitted = True
                    active_tool_use_id = None
                return event

            async for item in generator:
                if item.startswith("data:"):
                    data_str = item[5:].strip().rstrip("\n")
                    if data_str == "[DONE]":
                        stop_message = AnthropicStreamEvent(
                            type="message_stop",
                        )
                        data = stop_message.model_dump_json(
                            exclude_unset=True, exclude_none=True
                        )
                        yield wrap_data_with_event(data, "message_stop")
                        yield "data: [DONE]\n\n"
                    else:
                        origin_chunk = ChatCompletionStreamResponse.model_validate_json(
                            data_str
                        )

                        if first_item:
                            chunk = AnthropicStreamEvent(
                                type="message_start",
                                message=AnthropicMessagesResponse(
                                    id=origin_chunk.id,
                                    content=[],
                                    model=origin_chunk.model,
                                    stop_reason=None,
                                    stop_sequence=None,
                                    usage=AnthropicUsage(
                                        input_tokens=origin_chunk.usage.prompt_tokens
                                        if origin_chunk.usage
                                        else 0,
                                        output_tokens=0,
                                    ),
                                ),
                            )
                            first_item = False
                            data = chunk.model_dump_json(exclude_unset=True)
                            yield wrap_data_with_event(data, "message_start")
                            continue

                        # last chunk including usage info
                        if len(origin_chunk.choices) == 0:
                            for event in stop_active_block():
                                yield event
                            stop_reason = self.stop_reason_map.get(
                                finish_reason or "stop"
                            )
                            chunk = AnthropicStreamEvent(
                                type="message_delta",
                                delta=AnthropicDelta(stop_reason=stop_reason),
                                usage=AnthropicUsage(
                                    input_tokens=origin_chunk.usage.prompt_tokens
                                    if origin_chunk.usage
                                    else 0,
                                    output_tokens=origin_chunk.usage.completion_tokens
                                    if origin_chunk.usage
                                    else 0,
                                ),
                            )
                            data = chunk.model_dump_json(exclude_unset=True)
                            yield wrap_data_with_event(data, "message_delta")
                            continue

                        if origin_chunk.choices[0].finish_reason is not None:
                            finish_reason = origin_chunk.choices[0].finish_reason
                            # continue

                        # thinking / text content
                        reasoning_delta = origin_chunk.choices[0].delta.reasoning
                        if reasoning_delta is not None:
                            if reasoning_delta == "":
                                pass
                            else:
                                if active_block_type != "thinking":
                                    for event in stop_active_block():
                                        yield event
                                    start_event = start_block(
                                        AnthropicContentBlock(
                                            type="thinking", thinking=""
                                        )
                                    )
                                    yield start_event
                                chunk = AnthropicStreamEvent(
                                    index=(
                                        active_block_index
                                        if active_block_index is not None
                                        else content_block_index
                                    ),
                                    type="content_block_delta",
                                    delta=AnthropicDelta(
                                        type="thinking_delta",
                                        thinking=reasoning_delta,
                                    ),
                                )
                                data = chunk.model_dump_json(exclude_unset=True)
                                yield wrap_data_with_event(data, "content_block_delta")

                        if origin_chunk.choices[0].delta.content is not None:
                            if origin_chunk.choices[0].delta.content == "":
                                pass
                            else:
                                if active_block_type != "text":
                                    for event in stop_active_block():
                                        yield event
                                    start_event = start_block(
                                        AnthropicContentBlock(type="text", text="")
                                    )
                                    yield start_event
                                chunk = AnthropicStreamEvent(
                                    index=(
                                        active_block_index
                                        if active_block_index is not None
                                        else content_block_index
                                    ),
                                    type="content_block_delta",
                                    delta=AnthropicDelta(
                                        type="text_delta",
                                        text=origin_chunk.choices[0].delta.content,
                                    ),
                                )
                                data = chunk.model_dump_json(exclude_unset=True)
                                yield wrap_data_with_event(data, "content_block_delta")

                        # tool calls - process all tool calls in the delta
                        if len(origin_chunk.choices[0].delta.tool_calls) > 0:
                            for tool_call in origin_chunk.choices[0].delta.tool_calls:
                                if tool_call.id is not None:
                                    # Only create new block if different tool call
                                    # AND has a name
                                    tool_name = (
                                        tool_call.function.name
                                        if tool_call.function
                                        else None
                                    )
                                    if (
                                        active_tool_use_id != tool_call.id
                                        and tool_name is not None
                                    ):
                                        for event in stop_active_block():
                                            yield event
                                        start_event = start_block(
                                            AnthropicContentBlock(
                                                type="tool_use",
                                                id=tool_call.id,
                                                name=tool_name,
                                                input={},
                                            )
                                        )
                                        yield start_event
                                    if (
                                        tool_call.function
                                        and tool_call.function.arguments
                                        and active_tool_use_id == tool_call.id
                                    ):
                                        chunk = AnthropicStreamEvent(
                                            index=(
                                                active_block_index
                                                if active_block_index is not None
                                                else content_block_index
                                            ),
                                            type="content_block_delta",
                                            delta=AnthropicDelta(
                                                type="input_json_delta",
                                                partial_json=tool_call.function.arguments,
                                            ),
                                        )
                                        data = chunk.model_dump_json(exclude_unset=True)
                                        yield wrap_data_with_event(
                                            data, "content_block_delta"
                                        )
                            continue
                else:
                    error_response = AnthropicStreamEvent(
                        type="error",
                        error=AnthropicError(
                            type="internal_error",
                            message="Invalid data format received",
                        ),
                    )
                    data = error_response.model_dump_json(exclude_unset=True)
                    yield wrap_data_with_event(data, "error")
                    yield "data: [DONE]\n\n"

        except Exception as e:
            logger.exception("Error in message stream converter.")
            error_response = AnthropicStreamEvent(
                type="error",
                error=AnthropicError(type="internal_error", message=str(e)),
            )
            data = error_response.model_dump_json(exclude_unset=True)
            yield wrap_data_with_event(data, "error")
            yield "data: [DONE]\n\n"
