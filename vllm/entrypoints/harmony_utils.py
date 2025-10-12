# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import datetime
import json
from collections.abc import Iterable, Sequence
from typing import Literal, Optional, Union

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_function_web_search import (
    ActionFind,
    ActionOpenPage,
    ActionSearch,
    ResponseFunctionWebSearch,
)
from openai.types.responses.response_output_item import McpCall
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from openai.types.responses.tool import Tool
from openai_harmony import (
    Author,
    ChannelConfig,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    SystemContent,
    TextContent,
    ToolDescription,
    ToolNamespaceConfig,
    load_harmony_encoding,
)

from vllm import envs
from vllm.entrypoints.openai.protocol import (
    ChatCompletionToolsParam,
    ResponseInputOutputItem,
)
from vllm.entrypoints.tool_server import ToolServer
from vllm.logger import init_logger
from vllm.utils import random_uuid

logger = init_logger(__name__)

REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}

_harmony_encoding = None

# Builtin tools that should be included in the system message when
# they are available and requested by the user.
# Tool args are provided by MCP tool descriptions. Output
# of the tools are stringified.
BUILTIN_TOOLS = {
    "web_search_preview",
    "code_interpreter",
    "container",
}


def build_system_and_developer_messages(
    # Tool for ResponsesAPI, ChatCompletionToolsParam for CompletionsAPI
    request_tools: Union[list[Tool], list[ChatCompletionToolsParam]],
    tool_server: Optional[ToolServer],
    instructions: Optional[str] = None,
    reasoning_effort: Optional[Literal["high", "medium", "low"]] = None,
    start_date: Optional[str] = None,
    model_identity: Optional[str] = None,
) -> list[Message]:
    """Builds system and developer messages for a Harmony request.

    This function standardizes message construction between Responses API
    and Chat API. It handles tool elevation, message construction, and
    namespace collection.

    Args:
        request_tools: List of tools (already normalized to MCP format)
        tool_server: Tool server for fetching tool descriptions
        instructions: Custom instructions for the assistant
        reasoning_effort: Reasoning effort level
        start_date: Start date for the conversation
        model_identity: Model identity string

    Returns:
        List of system message and developer message if nedeed
    """
    messages = []

    # Get elevation list from environment
    elevated_namespaces = envs.GPT_OSS_SYSTEM_TOOL_MCP_LABELS or []

    # Classify tools by elevation status
    elevated_namespace_descriptions = []
    custom_namespace_descriptions = []
    function_tools = []

    for tool in request_tools:
        if tool.type == "mcp":
            if tool_server and tool_server.has_namespace(tool.server_label):
                tool_description = tool_server.get_tool_description(tool.server_label)
            else:
                available = (
                    list(tool_server.harmony_tool_descriptions.keys())
                    if tool_server
                    else []
                )
                raise ValueError(
                    f"MCP namespace '{tool.server_label}' in the request "
                    f"is not available in tool server. "
                    f"Available namespaces: {available}"
                )
            if tool.server_label in elevated_namespaces:
                elevated_namespace_descriptions.append(tool_description)
            else:
                custom_namespace_descriptions.append(tool_description)
        # type is function for responses and completions luckily
        elif tool.type == "function":
            function_tools.append(tool)
        else:
            raise ValueError(
                f"Tools should be of type 'mcp' or 'function', got {tool.type}"
                f" Tool type conversion should happen before this point. "
            )
    if function_tools:
        custom_namespace_descriptions.append(
            create_function_tools_namespace(function_tools)
        )

    sys_msg = get_system_message(
        model_identity=model_identity,
        reasoning_effort=reasoning_effort,
        start_date=start_date,
        elevated_namespace_descriptions=elevated_namespace_descriptions,
        custom_namespace_descriptions=custom_namespace_descriptions,
        instructions=instructions,
    )
    messages.append(sys_msg)

    dev_msg = get_developer_message(
        instructions=instructions,
        tool_namespaces=custom_namespace_descriptions,
    )
    if dev_msg is not None:
        messages.append(dev_msg)

    return messages


def get_encoding():
    global _harmony_encoding
    if _harmony_encoding is None:
        _harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _harmony_encoding


def create_function_tools_namespace(
    function_tools: list[Union[Tool, ChatCompletionToolsParam]],
) -> ToolNamespaceConfig:
    """
    Create a Harmony ToolNamespaceConfig from function tools.

    Function tools are converted to a namespace called "functions" that can be
    included in either system or developer messages.

    Args:
        function_tools: List of function-type tools

    Returns:
        ToolNamespaceConfig with namespace="functions" and all function tool definitions
    """
    tool_descriptions = [create_tool_definition(tool) for tool in function_tools]

    # Create namespace config with "functions" as the namespace name
    namespace_config = ToolNamespaceConfig(
        name="functions",
        # Empty to match harmony implementation of functions namespace
        description="",
        tools=tool_descriptions,
    )

    return namespace_config


def create_tool_definition(tool: Union[ChatCompletionToolsParam, Tool]):
    """Convert a tool to a Harmony ToolDescription."""
    if isinstance(tool, ChatCompletionToolsParam):
        return ToolDescription.new(
            name=tool.function.name,
            description=tool.function.description,
            parameters=tool.function.parameters,
        )
    return ToolDescription.new(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
    )


def get_system_message(
    model_identity: str | None = None,
    reasoning_effort: Literal["high", "medium", "low"] | None = None,
    start_date: str | None = None,
    elevated_namespace_descriptions: list | None = None,
    custom_namespace_descriptions: list | None = None,
    instructions: str | None = None,
) -> Message:
    """
    Construct system message for gpt-oss models.

    Args:
        model_identity: Model identity string
        reasoning_effort: Reasoning effort level (high/medium/low)
        start_date: Conversation start date
        elevated_namespace_descriptions: List of ToolNamespaceConfig
                                         for elevated namespaces
        custom_namespace_descriptions: List of ToolNamespaceConfig
                                         for custom namespaces
        instructions: User-provided instructions

    Returns:
        System message for Harmony protocol
    """
    sys_msg_content = SystemContent.new()
    if model_identity is not None:
        sys_msg_content = sys_msg_content.with_model_identity(model_identity)
    if instructions is not None and envs.VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS:
        current_identity = sys_msg_content.model_identity
        new_identity = (
            f"{current_identity}\n{instructions}" if current_identity else instructions
        )
        sys_msg_content = sys_msg_content.with_model_identity(new_identity)
    if reasoning_effort is not None:
        sys_msg_content = sys_msg_content.with_reasoning_effort(
            REASONING_EFFORT[reasoning_effort]
        )
    if start_date is None:
        # NOTE(woosuk): This brings non-determinism in vLLM. Be careful.
        start_date = datetime.datetime.now().strftime("%Y-%m-%d")
    sys_msg_content = sys_msg_content.with_conversation_start_date(start_date)

    # Elevated namespaces are registered in the system message
    if elevated_namespace_descriptions is not None:
        for tool_namespace in elevated_namespace_descriptions:
            sys_msg_content = sys_msg_content.with_tools(tool_namespace)

    # If no custom namespaces are provided, remove the "commentary" channel
    if not custom_namespace_descriptions:
        channel_config = sys_msg_content.channel_config
        invalid_channel = "commentary"
        new_config = ChannelConfig.require_channels(
            [c for c in channel_config.valid_channels if c != invalid_channel]
        )
        sys_msg_content = sys_msg_content.with_channel_config(new_config)
    sys_msg = Message.from_role_and_content(Role.SYSTEM, sys_msg_content)
    return sys_msg


def get_developer_message(
    instructions: str | None = None,
    tool_namespaces: list | None = None,
) -> Optional[Message]:
    """
    Construct developer message for custom (non-elevated) tools.

    Args:
        instructions: User-provided instructions
        tool_namespaces: List of ToolNamespaceConfig for all custom tools
            (MCP and function)

    Returns:
        Developer message for Harmony protocol, if needed
    """
    developer_instructions = (
        instructions if not envs.VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS else None
    )
    if not developer_instructions and not tool_namespaces:
        return None

    dev_msg_content = DeveloperContent.new()
    if developer_instructions:
        dev_msg_content = dev_msg_content.with_instructions(developer_instructions)

    # Add all tool namespaces
    if tool_namespaces:
        for tool_namespace in tool_namespaces:
            # Use with_tools instead of with_function_tools to simplify
            # adding non-functions namespaces to developer message
            dev_msg_content = dev_msg_content.with_tools(tool_namespace)

    dev_msg = Message.from_role_and_content(Role.DEVELOPER, dev_msg_content)
    return dev_msg


def get_user_message(content: str) -> Message:
    return Message.from_role_and_content(Role.USER, content)


def parse_response_input(
    response_msg: ResponseInputOutputItem,
    prev_responses: list[Union[ResponseOutputItem, ResponseReasoningItem]],
) -> Message:
    if not isinstance(response_msg, dict):
        response_msg = response_msg.model_dump()
    if "type" not in response_msg or response_msg["type"] == "message":
        role = response_msg["role"]
        content = response_msg["content"]
        if role == "system":
            # User is trying to set a system message. Change it to:
            # <|start|>developer<|message|># Instructions
            # {instructions}<|end|>
            role = "developer"
            text_prefix = "Instructions:\n"
        else:
            text_prefix = ""
        if isinstance(content, str):
            msg = Message.from_role_and_content(role, text_prefix + content)
        else:
            contents = [TextContent(text=text_prefix + c["text"]) for c in content]
            msg = Message.from_role_and_contents(role, contents)
        if role == "assistant":
            msg = msg.with_channel("final")
    elif response_msg["type"] == "function_call_output":
        call_id = response_msg["call_id"]
        call_response: ResponseFunctionToolCall | None = None
        for prev_response in reversed(prev_responses):
            if (
                isinstance(prev_response, ResponseFunctionToolCall)
                and prev_response.call_id == call_id
            ):
                call_response = prev_response
                break
        if call_response is None:
            raise ValueError(f"No call message found for {call_id}")
        msg = Message.from_author_and_content(
            Author.new(Role.TOOL, f"functions.{call_response.name}"),
            response_msg["output"],
        )
    elif response_msg["type"] == "reasoning":
        content = response_msg["content"]
        assert len(content) == 1
        msg = Message.from_role_and_content(Role.ASSISTANT, content[0]["text"])
    elif response_msg["type"] == "function_call":
        msg = Message.from_role_and_content(Role.ASSISTANT, response_msg["arguments"])
        msg = msg.with_channel("commentary")
        msg = msg.with_recipient(f"functions.{response_msg['name']}")
        msg = msg.with_content_type("json")
    else:
        raise ValueError(f"Unknown input type: {response_msg['type']}")
    return msg


def parse_chat_input(chat_msg) -> list[Message]:
    if not isinstance(chat_msg, dict):
        # Handle Pydantic models
        chat_msg = chat_msg.model_dump(exclude_none=True)

    role = chat_msg.get("role")

    # Assistant message with tool calls
    tool_calls = chat_msg.get("tool_calls")
    if role == "assistant" and tool_calls:
        msgs: list[Message] = []
        for call in tool_calls:
            func = call.get("function", {})
            name = func.get("name", "")
            arguments = func.get("arguments", "") or ""
            msg = Message.from_role_and_content(Role.ASSISTANT, arguments)
            msg = msg.with_channel("commentary")
            msg = msg.with_recipient(f"functions.{name}")
            msg = msg.with_content_type("json")
            msgs.append(msg)
        return msgs

    # Tool role message (tool output)
    if role == "tool":
        name = chat_msg.get("name", "")
        content = chat_msg.get("content", "") or ""
        if isinstance(content, list):
            # Handle array format for tool message content
            # by concatenating all text parts.
            content = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )

        msg = Message.from_author_and_content(
            Author.new(Role.TOOL, f"functions.{name}"), content
        ).with_channel("commentary")
        return [msg]

    # Default: user/assistant/system messages with content
    content = chat_msg.get("content", "")
    if isinstance(content, str):
        contents = [TextContent(text=content)]
    else:
        # TODO: Support refusal.
        contents = [TextContent(text=c.get("text", "")) for c in content]
    msg = Message.from_role_and_contents(role, contents)
    return [msg]


def render_for_completion(messages: list[Message]) -> list[int]:
    conversation = Conversation.from_messages(messages)
    token_ids = get_encoding().render_conversation_for_completion(
        conversation, Role.ASSISTANT
    )
    return token_ids


def parse_output_message(
    message: Message,
    output_items_so_far: list[ResponseOutputItem] | None = None,
) -> list[ResponseOutputItem]:
    """
    Parse a Harmony message into a list of output response items.

    Args:
        message: The message to parse
        output_items_so_far: List of output items parsed so far. When we see
            a tool response message, we search backward to find the most recent
            matching McpCall (by tool name) that has no output yet.
    """
    # Handle tool response messages (look-behind pattern)
    if message.author.role == "tool":
        # This is a tool response. Search backward to find matching tool call.
        if not output_items_so_far:
            logger.warning(
                "Tool response with no prior output items: %s", message.author.name
            )
            return []

        # Find the most recent McpCall that matches this tool and has no output
        tool_name = message.author.name  # e.g., "memory.store"
        matching_call = None

        for item in reversed(output_items_so_far):
            if isinstance(item, McpCall):
                call_full_name = f"{item.server_label}.{item.name}"
                if call_full_name == tool_name and item.output is None:
                    matching_call = item
                    break

        if matching_call:
            matching_call.output = message.content[0].text if message.content else None
            return []
        else:
            # We should error here, but it wouldn't make much sense
            # before we switch to using McpCall for all tool calls + output
            logger.error("Tool call output not output for tool: %s", tool_name)
            return []

    if message.author.role != "assistant":
        # This is some other role (not assistant, not tool) - skip it
        return []

    output_items: list[ResponseOutputItem] = []
    recipient = message.recipient
    if recipient is not None and recipient.startswith("browser."):
        if len(message.content) != 1:
            raise ValueError("Invalid number of contents in browser message")
        content = message.content[0]
        browser_call = json.loads(content.text)
        # TODO: translate to url properly!
        if recipient == "browser.search":
            action = ActionSearch(
                query=f"cursor:{browser_call.get('query', '')}", type="search"
            )
        elif recipient == "browser.open":
            action = ActionOpenPage(
                url=f"cursor:{browser_call.get('url', '')}", type="open_page"
            )
        elif recipient == "browser.find":
            action = ActionFind(
                pattern=browser_call["pattern"],
                url=f"cursor:{browser_call.get('url', '')}",
                type="find",
            )
        else:
            raise ValueError(f"Unknown browser action: {recipient}")
        web_search_item = ResponseFunctionWebSearch(
            id=f"ws_{random_uuid()}",
            action=action,
            status="completed",
            type="web_search_call",
        )
        output_items.append(web_search_item)
    elif message.channel == "analysis":
        for content in message.content:
            reasoning_item = ResponseReasoningItem(
                id=f"rs_{random_uuid()}",
                summary=[],
                type="reasoning",
                content=[
                    ResponseReasoningTextContent(
                        text=content.text, type="reasoning_text"
                    )
                ],
                status=None,
            )
            output_items.append(reasoning_item)
    elif message.channel == "commentary":
        if recipient is not None and recipient.startswith("functions."):
            function_name = recipient.split(".")[-1]
            for content in message.content:
                random_id = random_uuid()
                response_item = ResponseFunctionToolCall(
                    arguments=content.text,
                    call_id=f"call_{random_id}",
                    type="function_call",
                    name=function_name,
                    id=f"fc_{random_id}",
                )
                output_items.append(response_item)
        elif recipient is not None and (
            recipient.startswith("python")
            or recipient.startswith("browser")
            or recipient.startswith("container")
        ):
            # Built-in tools on commentary channel → reasoning items
            # For legacy compatibility for now
            # TODO: Use McpCall here too
            for content in message.content:
                reasoning_item = ResponseReasoningItem(
                    id=f"rs_{random_uuid()}",
                    summary=[],
                    type="reasoning",
                    content=[
                        ResponseReasoningTextContent(
                            text=content.text, type="reasoning_text"
                        )
                    ],
                    status=None,
                )
                output_items.append(reasoning_item)
        elif recipient is not None:
            # Any other non-function recipient on commentary channel → MCP call
            namespace = recipient.split(".")[0] if "." in recipient else recipient
            tool_name = recipient.split(".")[1] if "." in recipient else recipient

            for content in message.content:
                mcp_call = McpCall(
                    id=f"mcp_{random_uuid()}",
                    type="mcp_call",
                    name=tool_name,
                    server_label=namespace,
                    arguments=content.text,
                    output=None,
                    error=None,
                )
                output_items.append(mcp_call)
    elif message.channel == "final":
        contents = []
        for content in message.content:
            output_text = ResponseOutputText(
                text=content.text,
                annotations=[],  # TODO
                type="output_text",
                logprobs=None,  # TODO
            )
            contents.append(output_text)
        text_item = ResponseOutputMessage(
            id=f"msg_{random_uuid()}",
            content=contents,
            role=message.author.role,
            status="completed",
            type="message",
        )
        output_items.append(text_item)
    else:
        raise ValueError(f"Unknown channel: {message.channel}")
    return output_items


def parse_remaining_state(parser: StreamableParser) -> list[ResponseOutputItem]:
    if not parser.current_content:
        return []
    if parser.current_role != Role.ASSISTANT:
        return []
    current_recipient = parser.current_recipient
    if current_recipient is not None and current_recipient.startswith("browser."):
        return []

    if parser.current_channel == "analysis":
        reasoning_item = ResponseReasoningItem(
            id=f"rs_{random_uuid()}",
            summary=[],
            type="reasoning",
            content=[
                ResponseReasoningTextContent(
                    text=parser.current_content, type="reasoning_text"
                )
            ],
            status=None,
        )
        return [reasoning_item]
    elif parser.current_channel == "final":
        output_text = ResponseOutputText(
            text=parser.current_content,
            annotations=[],  # TODO
            type="output_text",
            logprobs=None,  # TODO
        )
        text_item = ResponseOutputMessage(
            id=f"msg_{random_uuid()}",
            content=[output_text],
            role="assistant",
            # if the parser still has messages (ie if the generator got cut
            # abruptly), this should be incomplete
            status="incomplete",
            type="message",
        )
        return [text_item]
    return []


def get_stop_tokens_for_assistant_actions() -> list[int]:
    return get_encoding().stop_tokens_for_assistant_actions()


def get_streamable_parser_for_assistant() -> StreamableParser:
    return StreamableParser(get_encoding(), role=Role.ASSISTANT)


def parse_output_into_messages(token_ids: Iterable[int]) -> StreamableParser:
    parser = get_streamable_parser_for_assistant()
    for token_id in token_ids:
        parser.process(token_id)
    return parser


def parse_chat_output(
    token_ids: Sequence[int],
) -> tuple[str | None, str | None, bool]:
    parser = parse_output_into_messages(token_ids)
    output_msgs = parser.messages
    is_tool_call = False  # TODO: update this when tool call is supported
    if len(output_msgs) == 0:
        # The generation has stopped during reasoning.
        reasoning_content = parser.current_content
        final_content = None
    elif len(output_msgs) == 1:
        # The generation has stopped during final message.
        reasoning_content = output_msgs[0].content[0].text
        final_content = parser.current_content
    else:
        reasoning_msg = output_msgs[:-1]
        final_msg = output_msgs[-1]
        reasoning_content = "\n".join([msg.content[0].text for msg in reasoning_msg])
        final_content = final_msg.content[0].text
    return reasoning_content, final_content, is_tool_call
