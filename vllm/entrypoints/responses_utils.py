# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    Function as FunctionCallTool,
)
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputItem
from openai.types.responses.response_function_tool_call_output_item import (
    ResponseFunctionToolCallOutputItem,
)
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from openai.types.responses.tool import Tool

from vllm import envs
from vllm.entrypoints.openai.protocol import (
    ChatCompletionMessageParam,
    ResponseInputOutputItem,
)


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
        for item in request_input:
            messages.append(construct_chat_message_with_tool_call(item))
    return messages


def construct_chat_message_with_tool_call(
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
