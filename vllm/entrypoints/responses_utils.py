# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    Function as FunctionCallTool,
)
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputItem
from openai.types.responses.response import ToolChoice
from openai.types.responses.response_function_tool_call_output_item import (
    ResponseFunctionToolCallOutputItem,
)
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from openai.types.responses.tool import Tool

from vllm import envs
from vllm.entrypoints.constants import MCP_PREFIX
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionMessageParam
from vllm.entrypoints.openai.responses.protocol import ResponseInputOutputItem


def should_continue_final_message(
    request_input: str | list[ResponseInputOutputItem],
) -> bool:
    """
    Determine if the last input message is a partial assistant message
    that should be continued rather than starting a new generation.

    This enables partial message completion similar to Anthropic's Messages API,
    where users can provide an incomplete assistant message and have the model
    continue from where it left off.

    A message is considered partial if:
    1. It's a ResponseOutputMessage or ResponseReasoningItem
    2. Its status is "in_progress" or "incomplete"

    Args:
        request_input: The input to the Responses API request

    Returns:
        True if the final message should be continued, False otherwise
    """
    if isinstance(request_input, str):
        # Simple string input is always a user message
        return False

    if not request_input:
        return False

    last_item = request_input[-1]

    # Check if the last item is a partial assistant message
    if isinstance(last_item, ResponseOutputMessage):
        return last_item.status in ("in_progress", "incomplete")

    # Check if the last item is a partial reasoning item
    if isinstance(last_item, ResponseReasoningItem):
        return last_item.status in ("in_progress", "incomplete")

    if isinstance(last_item, dict):
        # only support partial completion for messages for now
        if last_item.get("type", "message") not in ("message", "reasoning"):
            return False
        return last_item.get("status") in ("in_progress", "incomplete")

    return False


def construct_input_messages(
    *,
    request_instructions: str | None = None,
    request_input: str | list[ResponseInputOutputItem],
    prev_msg: list[ChatCompletionMessageParam] | None = None,
    prev_response_output: list[ResponseOutputItem] | None = None,
):
    messages: list[ChatCompletionMessageParam] = []
    if request_instructions:
        messages.append(
            {
                "role": "system",
                "content": request_instructions,
            }
        )

    # Prepend the conversation history.
    if prev_msg is not None:
        # Add the previous messages.
        messages.extend(prev_msg)
    if prev_response_output is not None:
        # Add the previous output.
        for output_item in prev_response_output:
            # NOTE: We skip the reasoning output.
            if isinstance(output_item, ResponseOutputMessage):
                for content in output_item.content:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content.text,
                        }
                    )

    # Append the new input.
    # Responses API supports simple text inputs without chat format.
    if isinstance(request_input, str):
        messages.append({"role": "user", "content": request_input})
    else:
        input_messages = construct_chat_messages_with_tool_call(request_input)
        messages.extend(input_messages)
    return messages


def _maybe_combine_reasoning_and_tool_call(
    item: ResponseInputOutputItem, messages: list[ChatCompletionMessageParam]
) -> ChatCompletionMessageParam | None:
    """Many models treat MCP calls and reasoning as a single message.
    This function checks if the last message is a reasoning message and
    the current message is a tool call"""
    if not (
        isinstance(item, ResponseFunctionToolCall)
        and item.id
        and item.id.startswith(MCP_PREFIX)
    ):
        return None
    if len(messages) == 0:
        return None
    last_message = messages[-1]
    if not (
        last_message.get("role") == "assistant"
        and last_message.get("reasoning") is not None
    ):
        return None

    last_message["tool_calls"] = [
        ChatCompletionMessageToolCallParam(
            id=item.call_id,
            function=FunctionCallTool(
                name=item.name,
                arguments=item.arguments,
            ),
            type="function",
        )
    ]
    return last_message


def construct_chat_messages_with_tool_call(
    input_messages: list[ResponseInputOutputItem],
) -> list[ChatCompletionMessageParam]:
    """This function wraps _construct_single_message_from_response_item
    Because some chatMessages come from multiple response items
    for example a reasoning item and a MCP tool call are two response items
    but are one chat message
    """
    messages: list[ChatCompletionMessageParam] = []
    for item in input_messages:
        maybe_combined_message = _maybe_combine_reasoning_and_tool_call(item, messages)
        if maybe_combined_message is not None:
            messages[-1] = maybe_combined_message
        else:
            messages.append(_construct_single_message_from_response_item(item))

    return messages


def _construct_single_message_from_response_item(
    item: ResponseInputOutputItem,
) -> ChatCompletionMessageParam:
    if isinstance(item, ResponseFunctionToolCall):
        # Append the function call as a tool call.
        return ChatCompletionAssistantMessageParam(
            role="assistant",
            tool_calls=[
                ChatCompletionMessageToolCallParam(
                    id=item.call_id,
                    function=FunctionCallTool(
                        name=item.name,
                        arguments=item.arguments,
                    ),
                    type="function",
                )
            ],
        )
    elif isinstance(item, ResponseReasoningItem):
        reasoning_content = ""
        if item.encrypted_content:
            raise ValueError("Encrypted content is not supported.")
        if len(item.summary) == 1:
            reasoning_content = item.summary[0].text
        elif item.content and len(item.content) == 1:
            reasoning_content = item.content[0].text
        return {
            "role": "assistant",
            "reasoning": reasoning_content,
        }
    elif isinstance(item, ResponseOutputMessage):
        return {
            "role": "assistant",
            "content": item.content[0].text,
        }
    elif isinstance(item, ResponseFunctionToolCallOutputItem):
        return ChatCompletionToolMessageParam(
            role="tool",
            content=item.output,
            tool_call_id=item.call_id,
        )
    elif isinstance(item, dict) and item.get("type") == "function_call_output":
        # Append the function call output as a tool message.
        return ChatCompletionToolMessageParam(
            role="tool",
            content=item.get("output"),
            tool_call_id=item.get("call_id"),
        )
    return item  # type: ignore


def extract_tool_types(tools: list[Tool]) -> set[str]:
    """
    Extracts the tool types from the given tools.
    """
    tool_types: set[str] = set()
    for tool in tools:
        if tool.type == "mcp":
            # Allow the MCP Tool type to enable built in tools if the
            # server_label is allowlisted in
            # envs.VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS
            if tool.server_label in envs.VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS:
                tool_types.add(tool.server_label)
        else:
            tool_types.add(tool.type)
    return tool_types


def convert_tool_responses_to_completions_format(tool: dict) -> dict:
    """
    Convert a flat tool schema:
        {"type": "function", "name": "...", "description": "...", "parameters": {...}}
    into:
        {"type": "function", "function": {...}}
    """
    return {
        "type": "function",
        "function": tool,
    }


def construct_tool_dicts(
    tools: list[Tool], tool_choice: ToolChoice
) -> list[dict[str, Any]] | None:
    if tools is None or (tool_choice == "none"):
        tool_dicts = None
    else:
        tool_dicts = [
            convert_tool_responses_to_completions_format(tool.model_dump())
            for tool in tools
        ]
    return tool_dicts
