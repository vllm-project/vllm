# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from copy import deepcopy
from typing import Any, Final

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    Function as FunctionCallTool,
)
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response import ToolChoice
from openai.types.responses.response_function_tool_call_output_item import (
    ResponseFunctionToolCallOutputItem,
)
from openai.types.responses.response_output_text import Logprob
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from openai.types.responses.tool import Mcp, Tool
from pydantic import TypeAdapter, ValidationError

from vllm import envs
from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.mcp.tool_server import ToolServer
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionMessageParam,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import FunctionCall, FunctionDefinition
from vllm.entrypoints.openai.responses.protocol import ResponseInputOutputItem
from vllm.logger import init_logger
from vllm.tool_parsers.utils import (
    build_responses_tool_call_name_map,
    flat_namespace_tool_name,
    iter_response_function_tool_dicts,
    resolve_responses_tool_call_name,
)
from vllm.utils import random_uuid

logger = init_logger(__name__)


def build_response_output_items(
    reasoning: str | None,
    content: str | None,
    tool_calls: list[FunctionCall] | None,
    logprobs: list[Logprob] | None = None,
    tools: list[Tool] | None = None,
) -> list[ResponseOutputItem]:
    outputs: list[ResponseOutputItem] = []
    tool_call_name_map = build_responses_tool_call_name_map(tools)

    if reasoning:
        outputs.append(
            ResponseReasoningItem(
                id=f"rs_{random_uuid()}",
                summary=[],
                type="reasoning",
                content=[
                    ResponseReasoningTextContent(text=reasoning, type="reasoning_text")
                ],
                status=None,
            )
        )

    if content:
        outputs.append(
            ResponseOutputMessage(
                id=f"msg_{random_uuid()}",
                content=[
                    ResponseOutputText(
                        text=content,
                        annotations=[],
                        type="output_text",
                        logprobs=logprobs,
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        )

    if tool_calls:
        for idx, tool_call in enumerate(tool_calls):
            call_name = resolve_responses_tool_call_name(
                tool_call.name, tool_call_name_map=tool_call_name_map
            )
            outputs.append(
                ResponseFunctionToolCall(
                    id=f"fc_{random_uuid()}",
                    call_id=tool_call.id
                    or make_tool_call_id(func_name=tool_call.name, idx=idx),
                    type="function_call",
                    status="completed",
                    name=call_name.name,
                    namespace=call_name.namespace,
                    arguments=tool_call.arguments,
                )
            )

    return outputs


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
        message = _construct_message_from_response_item(
            item, prev_msg=messages[-1] if messages else None
        )
        if message is not None:
            messages.append(message)

    return messages


def _construct_message_from_response_item(
    item: ResponseInputOutputItem,
    prev_msg: ChatCompletionMessageParam | None = None,
) -> ChatCompletionMessageParam | None:
    """
    Returns a new message or None. If `None`, `prev_msg` might be updated.
    If `prev_msg` is `None`, a new message is always returned.
    """
    prev_assistant_msg = (
        prev_msg if prev_msg and prev_msg.get("role") == "assistant" else None
    )

    if isinstance(item, ResponseFunctionToolCall):
        tool_name = item.name
        if item.namespace:
            tool_name = flat_namespace_tool_name(item.namespace, item.name)
        tool_call = ChatCompletionMessageToolCallParam(
            id=item.call_id,
            function=FunctionCallTool(
                name=tool_name,
                arguments=item.arguments,
            ),
            type="function",
        )
        if prev_assistant_msg:
            tool_calls = prev_assistant_msg.get("tool_calls")
            if tool_calls is None:
                prev_assistant_msg["tool_calls"] = [tool_call]
                return None
            if isinstance(tool_calls, list):
                tool_calls.append(tool_call)
                return None
            if isinstance(tool_calls, Iterable) and not isinstance(
                tool_calls, (dict, str)
            ):
                tool_calls = list(tool_calls)
                tool_calls.append(tool_call)
                prev_assistant_msg["tool_calls"] = tool_calls
                return None
            logger.warning(
                "Previous assistant message has unknown tool_calls format. "
                "Tool call merging is skipped and a new assistant message is created. "
                "Item %s",
                item.id,
            )
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

        if prev_assistant_msg:
            previous_reasoning = prev_assistant_msg.get("reasoning")
            if previous_reasoning is None:
                prev_assistant_msg["reasoning"] = reasoning
                return None
        return {
            "role": "assistant",
            "reasoning": reasoning,
        }
    elif isinstance(item, ResponseOutputMessage):
        output_text = item.content[0].text
        if prev_assistant_msg:
            previous_content = prev_assistant_msg.get("content")
            if previous_content is None:
                prev_assistant_msg["content"] = output_text
                return None
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
    elif isinstance(item, dict) and item.get("role") == "assistant":
        content = item.get("content")
        text: str | None = None
        if isinstance(content, str):
            text = content
        elif isinstance(content, list) and content:
            text = content[0].get("text")
        if text is not None:
            if prev_assistant_msg:
                previous_content = prev_assistant_msg.get("content")
                if previous_content is None:
                    prev_assistant_msg["content"] = text
                    return None
            return {"role": "assistant", "content": text}
    return item  # type: ignore[arg-type]


def extract_function_tool_names(tools: list[Tool]) -> frozenset[str]:
    names = []
    for tool in tools:
        if tool.type == "function":
            names.append(tool.name)
        elif tool.type == "namespace":
            names.extend(
                flat_namespace_tool_name(tool.name, namespaced_tool.name)
                for namespaced_tool in tool.tools
                if namespaced_tool.type == "function"
            )
    return frozenset(names)


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


def convert_tool_responses_to_completions_format(
    tool: dict,
) -> ChatCompletionToolsParam:
    """
    Convert a flat Responses tool schema:
        {"type": "function", "name": "...", "description": "...", "parameters": {...}}
    into a Chat Completions tool param for chat-template rendering.
    """
    return ChatCompletionToolsParam(
        type="function",
        function=FunctionDefinition.model_validate(
            {k: v for k, v in tool.items() if k != "type"}
        ),
    )


def construct_tool_dicts(
    tools: list[Tool],
    tool_choice: ToolChoice,
    exclude_tools_when_tool_choice_none: bool = False,
) -> list[dict[str, Any]] | None:
    if not tools or (tool_choice == "none" and exclude_tools_when_tool_choice_none):
        return None
    return [
        convert_tool_responses_to_completions_format(tool).model_dump()
        for tool in iter_response_function_tool_dicts(tools)
    ]


# Canonical flat function names and schemas used to expose builtin tools to
# models that call them through the standard function-call syntax. Names
# match the dispatch expectations in ``ParsableContext.need_builtin_tool_call``
# and the tool executors (e.g. ``call_python_tool`` expects ``{"code": ...}``).
_BUILTIN_TOOL_PROMPT_SPECS: Final[dict[str, dict[str, Any]]] = {
    "code_interpreter": {
        "description": (
            "Executes Python code in a stateful Jupyter notebook and returns"
            " the result. Use this tool to run computations instead of"
            " simulating them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute.",
                }
            },
            "required": ["code"],
        },
    },
    "web_search_preview": {
        "description": "Searches the web and returns relevant results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                }
            },
            "required": ["query"],
        },
    },
    "container": {
        "description": ("Executes a command inside a stateful container environment."),
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The command to execute.",
                },
                "workdir": {"type": "string"},
                "env": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "session_name": {"type": "string"},
                "timeout": {"type": "integer"},
                "user": {"type": "string"},
            },
            "required": ["cmd"],
        },
    },
}

# Request tool type -> ToolServer registration name.
_REQUEST_TYPE_TO_SERVER_TOOL: Final[dict[str, str]] = {
    "code_interpreter": "python",
    "web_search_preview": "browser",
    "container": "container",
}

_TOOL_ADAPTER: Final[TypeAdapter[Tool]] = TypeAdapter(Tool)


def _extract_mcp_allowed_tool_names(tool: Mcp) -> list[str] | None:
    """Return explicit MCP tool names, or None when unrestricted."""
    allowed_tools_val = None
    if tool.allowed_tools is not None:
        if isinstance(tool.allowed_tools, list):
            allowed_tools_val = tool.allowed_tools
        elif hasattr(tool.allowed_tools, "tool_names"):
            allowed_tools_val = tool.allowed_tools.tool_names
    if allowed_tools_val is not None and "*" in allowed_tools_val:
        return None
    return allowed_tools_val


def _builtin_tool_dict(
    request_type: str,
) -> dict[str, Any]:
    spec = _BUILTIN_TOOL_PROMPT_SPECS[request_type]
    return {
        "type": "function",
        "function": {
            "name": request_type,
            "description": spec["description"],
            "parameters": deepcopy(spec["parameters"]),
        },
    }


def _mcp_tool_dict(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Tool '{name}' provided by an MCP server.",
        },
    }


def construct_builtin_and_mcp_tool_dicts(
    tools: list[Tool],
    tool_server: ToolServer | None,
) -> list[dict[str, Any]] | None:
    """Build chat-completions function dicts for builtin and MCP tools.

    Builtin tools (``code_interpreter``, ``web_search_preview``,
    ``container``) are only rendered when a local tool server actually
    provides the backing implementation. ``mcp`` tools are rendered from
    their explicitly listed ``allowed_tools``; unrestricted lists cannot be
    enumerated without contacting the remote server and are skipped.

    Entries may be SDK tool models or raw dicts; raw dicts that do not
    validate against the SDK tool union (e.g. tool types absent from the
    installed SDK version) are matched by their raw ``type`` field.

    Returns None when there is nothing to render so callers can merge the
    result with ``construct_tool_dicts`` output directly.
    """
    prompt_tools: list[dict[str, Any]] = []
    for tool in _normalize_tools(tools):
        tool_type = _tool_type(tool)
        if tool_type in _BUILTIN_TOOL_PROMPT_SPECS:
            server_tool = _REQUEST_TYPE_TO_SERVER_TOOL[tool_type]
            if tool_server is None or not tool_server.has_tool(server_tool):
                continue
            prompt_tools.append(_builtin_tool_dict(tool_type))
        elif isinstance(tool, Mcp):
            allowed = _extract_mcp_allowed_tool_names(tool)
            if allowed is None:
                logger.debug(
                    "MCP server %r has unrestricted allowed_tools; skipping "
                    "prompt rendering because the remote tool list cannot be "
                    "enumerated.",
                    tool.server_label,
                )
                continue
            prompt_tools.extend(_mcp_tool_dict(name) for name in allowed)
    return prompt_tools or None


def _tool_type(tool: Tool | dict[str, Any]) -> str:
    if isinstance(tool, dict):
        return tool.get("type", "")
    return tool.type


def _normalize_tools(tools: list[Tool]) -> list[Tool]:
    normalized: list[Tool] = []
    for tool in tools:
        if not isinstance(tool, dict):
            normalized.append(tool)
            continue
        try:
            normalized.append(_TOOL_ADAPTER.validate_python(tool))
        except ValidationError:
            # Keep raw dicts for tool types the installed SDK cannot model.
            normalized.append(tool)
    return normalized
