# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Any

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartTextParam,
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
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionMessageParam
from vllm.entrypoints.openai.responses.protocol import ResponseInputOutputItem
from vllm.logger import init_logger

logger = init_logger(__name__)


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
        # Filter out system messages from previous conversation -- per the
        # OpenAI spec, instructions should NOT carry over across responses.
        # The current request's instructions (if any) were already added above.
        messages.extend(m for m in prev_msg if m.get("role") != "system")
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


def construct_chat_messages_with_tool_call(
    input_messages: list[ResponseInputOutputItem],
) -> list[ChatCompletionMessageParam]:
    """Build chat messages from response items.

    Some chat messages span multiple response items (e.g., reasoning + tool calls).
    """
    messages: list[ChatCompletionMessageParam] = []
    for item in input_messages:
        message = _construct_message_from_response_item(item, messages)
        if message is not None:
            messages.append(message)

    return messages


def _construct_message_from_response_item(
    item: ResponseInputOutputItem,
    msgs: list[ChatCompletionMessageParam],
) -> ChatCompletionMessageParam | None:
    """Construct a chat message or update the prior assistant message."""
    previous_assistant = (
        msgs[-1] if msgs and msgs[-1].get("role") == "assistant" else None
    )

    if isinstance(item, ResponseFunctionToolCall):
        tool_call = ChatCompletionMessageToolCallParam(
            id=item.call_id,
            function=FunctionCallTool(
                name=item.name,
                arguments=item.arguments,
            ),
            type="function",
        )
        if previous_assistant is not None:
            tool_calls = previous_assistant.get("tool_calls")
            if tool_calls is None:
                previous_assistant["tool_calls"] = [tool_call]
            else:
                assert isinstance(tool_calls, list)
                tool_calls.append(tool_call)
            return None
        return ChatCompletionAssistantMessageParam(
            role="assistant",
            tool_calls=[tool_call],
        )
    elif isinstance(item, ResponseReasoningItem):
        reasoning = ""
        if item.encrypted_content:
            raise ValueError("Encrypted content is not supported.")
        elif item.content and len(item.content) >= 1:
            reasoning = item.content[0].text
        elif len(item.summary) >= 1:
            reasoning = item.summary[0].text
            logger.warning(
                "Using summary text as reasoning content for item %s. "
                "Please use content instead of summary for "
                "reasoning items.",
                item.id,
            )

        if previous_assistant is not None:
            previous_reasoning = previous_assistant.get("reasoning")
            if previous_reasoning is None:
                previous_assistant["reasoning"] = reasoning
                return None
            if isinstance(previous_reasoning, str):
                previous_assistant["reasoning"] = previous_reasoning + reasoning
                return None
            logger.warning(
                "Previous assistant message has unknown reasoning format. "
                "Reasoning concatenation is skipped. Item %s",
                item.id,
            )
        return {
            "role": "assistant",
            "reasoning": reasoning,
        }
    elif isinstance(item, ResponseOutputMessage):
        output_text = item.content[0].text
        if previous_assistant is not None:
            previous_content = previous_assistant.get("content")
            if previous_content is None:
                previous_assistant["content"] = output_text
                return None
            if isinstance(previous_content, str):
                previous_assistant["content"] = [
                    ChatCompletionContentPartTextParam(
                        type="text", text=previous_content
                    ),
                    ChatCompletionContentPartTextParam(type="text", text=output_text),
                ]
                return None
            if isinstance(previous_content, Iterable) and not isinstance(
                previous_content, dict
            ):
                previous_content_parts = list(previous_content)
                previous_content_parts.append(
                    ChatCompletionContentPartTextParam(type="text", text=output_text)
                )
                previous_assistant["content"] = previous_content_parts
                return None
            logger.warning(
                "Previous assistant message has unknown content format. "
                "Content merging is skipped. Item %s",
                item.id,
            )
        return {
            "role": "assistant",
            "content": output_text,
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
    return item  # type: ignore[arg-type]


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
    if not tools or (tool_choice == "none"):
        tool_dicts = None
    else:
        tool_dicts = [
            convert_tool_responses_to_completions_format(tool.model_dump())
            for tool in tools
        ]
    return tool_dicts
