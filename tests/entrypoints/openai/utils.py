# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections.abc import AsyncGenerator
from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import (
    UsageInfo,
)


async def accumulate_streaming_response(
    stream_generator: AsyncGenerator[str, None],
) -> ChatCompletionResponse:
    """
    Accumulate streaming SSE chunks into a complete ChatCompletionResponse.

    This helper parses the SSE format and builds up the complete response
    by combining all the delta chunks.
    """
    accumulated_content = ""
    accumulated_reasoning = None
    accumulated_tool_calls: list[dict[str, Any]] = []
    role = None
    finish_reason = None
    response_id = None
    created = None
    model = None
    index = 0

    async for chunk_str in stream_generator:
        # Skip empty lines and [DONE] marker
        if not chunk_str.strip() or chunk_str.strip() == "data: [DONE]":
            continue

        # Parse SSE format: "data: {json}\n\n"
        if chunk_str.startswith("data: "):
            json_str = chunk_str[6:].strip()
            try:
                chunk_data = json.loads(json_str)
                # print(f"DEBUG: Parsed chunk_data: {chunk_data}")
                chunk = ChatCompletionStreamResponse(**chunk_data)

                # Store metadata from first chunk
                if response_id is None:
                    response_id = chunk.id
                    created = chunk.created
                    model = chunk.model

                # Process each choice in the chunk
                for choice in chunk.choices:
                    if choice.delta.role:
                        role = choice.delta.role
                    if choice.delta.content:
                        accumulated_content += choice.delta.content
                    if choice.delta.reasoning:
                        if accumulated_reasoning is None:
                            accumulated_reasoning = ""
                        accumulated_reasoning += choice.delta.reasoning
                    if choice.delta.tool_calls:
                        # Accumulate tool calls
                        for tool_call_delta in choice.delta.tool_calls:
                            # Find or create the tool call at this index
                            while len(accumulated_tool_calls) <= tool_call_delta.index:
                                accumulated_tool_calls.append(
                                    {
                                        "id": None,
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                )

                            if tool_call_delta.id:
                                accumulated_tool_calls[tool_call_delta.index]["id"] = (
                                    tool_call_delta.id
                                )
                            if tool_call_delta.function:
                                if tool_call_delta.function.name:
                                    accumulated_tool_calls[tool_call_delta.index][
                                        "function"
                                    ]["name"] += tool_call_delta.function.name
                                if tool_call_delta.function.arguments:
                                    accumulated_tool_calls[tool_call_delta.index][
                                        "function"
                                    ]["arguments"] += tool_call_delta.function.arguments

                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                    if choice.index is not None:
                        index = choice.index

            except json.JSONDecodeError:
                continue

    # Build the final message
    message_kwargs = {
        "role": role or "assistant",
        "content": accumulated_content if accumulated_content else None,
        "reasoning": accumulated_reasoning,
    }

    # Only include tool_calls if there are any
    if accumulated_tool_calls:
        message_kwargs["tool_calls"] = [
            {"id": tc["id"], "type": tc["type"], "function": tc["function"]}
            for tc in accumulated_tool_calls
        ]

    message = ChatMessage(**message_kwargs)

    # Build the final response
    choice = ChatCompletionResponseChoice(
        index=index,
        message=message,
        finish_reason=finish_reason or "stop",
    )

    # Create usage info (with dummy values for tests)
    usage = UsageInfo(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
    )

    response = ChatCompletionResponse(
        id=response_id or "chatcmpl-test",
        object="chat.completion",
        created=created or 0,
        model=model or "test-model",
        choices=[choice],
        usage=usage,
    )

    return response


def verify_harmony_messages(
    messages: list[Any], expected_messages: list[dict[str, Any]]
):
    assert len(messages) == len(expected_messages)
    for msg, expected in zip(messages, expected_messages):
        if "role" in expected:
            assert msg.author.role == expected["role"]
        if "author_name" in expected:
            assert msg.author.name == expected["author_name"]
        if "channel" in expected:
            assert msg.channel == expected["channel"]
        if "recipient" in expected:
            assert msg.recipient == expected["recipient"]
        if "content" in expected:
            assert msg.content[0].text == expected["content"]
        if "content_type" in expected:
            assert msg.content_type == expected["content_type"]
        if "tool_definitions" in expected:
            # Check that the tool definitions match the expected list of tool names
            actual_tools = [t.name for t in msg.content[0].tools["functions"].tools]
            assert actual_tools == expected["tool_definitions"]


def verify_chat_response(
    response: ChatCompletionResponse,
    content: str | None = None,
    reasoning: str | None = None,
    tool_calls: list[tuple[str, str]] | None = None,
):
    assert len(response.choices) == 1
    message = response.choices[0].message

    if content is not None:
        assert message.content == content
    else:
        assert not message.content

    if reasoning is not None:
        assert message.reasoning == reasoning
    else:
        assert not message.reasoning

    if tool_calls:
        assert message.tool_calls is not None
        assert len(message.tool_calls) == len(tool_calls)
        for tc, (expected_name, expected_args) in zip(message.tool_calls, tool_calls):
            assert tc.function.name == expected_name
            assert tc.function.arguments == expected_args
    else:
        assert not message.tool_calls
