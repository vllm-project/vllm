# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import datetime
import json
from collections.abc import Iterable, Sequence
from typing import Literal

import regex as re
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
    load_harmony_encoding,
)
from openai_harmony import Message as OpenAIHarmonyMessage
from openai_harmony import Role as OpenAIHarmonyRole

from vllm import envs
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.entrypoints.openai.responses.protocol import (
    ResponseInputOutputItem,
    ResponsesRequest,
)
from vllm.utils import random_uuid

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
MCP_BUILTIN_TOOLS: set[str] = {
    "web_search_preview",
    "code_interpreter",
    "container",
}

_HARMONY_CONTROL_TOKEN_RE = re.compile(r"<\|[^>]*?\|>")
_INVALID_TOOL_NAME = "__invalid_tool__"


def sanitize_harmony_tool_name(name: str | None) -> str:
    if not name:
        return ""
    cleaned = name.strip()
    if "<|" not in cleaned:
        return cleaned
    prefix = cleaned.split("<|", 1)[0].strip()
    if prefix:
        return prefix
    cleaned = _HARMONY_CONTROL_TOKEN_RE.sub("", cleaned).strip()
    return cleaned or _INVALID_TOOL_NAME


def strip_harmony_control_tokens(text: str | None) -> str | None:
    if text is None:
        return None
    if "<|" not in text:
        return text
    return _HARMONY_CONTROL_TOKEN_RE.sub("", text)


def has_custom_tools(tool_types: set[str]) -> bool:
    """
    Checks if the given tool types are custom tools
    (i.e. any tool other than MCP buildin tools)
    """
    return not tool_types.issubset(MCP_BUILTIN_TOOLS)


def get_encoding():
    global _harmony_encoding
    if _harmony_encoding is None:
        _harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _harmony_encoding


def get_system_message(
    model_identity: str | None = None,
    reasoning_effort: Literal["high", "medium", "low"] | None = None,
    start_date: str | None = None,
    browser_description: str | None = None,
    python_description: str | None = None,
    container_description: str | None = None,
    instructions: str | None = None,
    with_custom_tools: bool = False,
) -> Message:
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
    if browser_description is not None:
        sys_msg_content = sys_msg_content.with_tools(browser_description)
    if python_description is not None:
        sys_msg_content = sys_msg_content.with_tools(python_description)
    if container_description is not None:
        sys_msg_content = sys_msg_content.with_tools(container_description)
    if not with_custom_tools:
        channel_config = sys_msg_content.channel_config
        invalid_channel = "commentary"
        new_config = ChannelConfig.require_channels(
            [c for c in channel_config.valid_channels if c != invalid_channel]
        )
        sys_msg_content = sys_msg_content.with_channel_config(new_config)
    sys_msg = Message.from_role_and_content(Role.SYSTEM, sys_msg_content)
    return sys_msg


def create_tool_definition(tool: ChatCompletionToolsParam | Tool):
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


def get_developer_message(
    instructions: str | None = None,
    tools: list[Tool | ChatCompletionToolsParam] | None = None,
) -> Message:
    dev_msg_content = DeveloperContent.new()
    if instructions is not None and not envs.VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS:
        dev_msg_content = dev_msg_content.with_instructions(instructions)
    if tools is not None:
        function_tools: list[Tool | ChatCompletionToolsParam] = []
        for tool in tools:
            if tool.type in (
                "web_search_preview",
                "code_interpreter",
                "container",
            ):
                pass

            elif tool.type == "function":
                function_tools.append(tool)
            else:
                raise ValueError(f"tool type {tool.type} not supported")
        if function_tools:
            function_tool_descriptions = [
                create_tool_definition(tool) for tool in function_tools
            ]
            dev_msg_content = dev_msg_content.with_function_tools(
                function_tool_descriptions
            )
    dev_msg = Message.from_role_and_content(Role.DEVELOPER, dev_msg_content)
    return dev_msg


def get_user_message(content: str) -> Message:
    return Message.from_role_and_content(Role.USER, content)


def parse_response_input(
    response_msg: ResponseInputOutputItem,
    prev_responses: list[ResponseOutputItem | ResponseReasoningItem],
) -> Message:
    if not isinstance(response_msg, dict):
        response_msg = response_msg.model_dump()
    if "type" not in response_msg or response_msg["type"] == "message":
        role = response_msg["role"]
        content = response_msg["content"]
        # Add prefix for developer messages.
        # <|start|>developer<|message|># Instructions {instructions}<|end|>
        text_prefix = "Instructions:\n" if role == "developer" else ""
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
        tool_name = sanitize_harmony_tool_name(response_msg.get("name"))
        msg = msg.with_recipient(f"functions.{tool_name}")
        msg = msg.with_content_type("json")
    else:
        raise ValueError(f"Unknown input type: {response_msg['type']}")
    return msg


def parse_chat_inputs_to_harmony_messages(chat_msgs: list) -> list[Message]:
    """
    Parse a list of messages from request.messages in the Chat Completion API to
    Harmony messages.
    """
    msgs: list[Message] = []
    tool_id_names: dict[str, str] = {}

    # Collect tool id to name mappings for tool response recipient values
    for chat_msg in chat_msgs:
        for tool_call in chat_msg.get("tool_calls", []):
            raw_name = tool_call.get("function", {}).get("name")
            tool_id_names[tool_call.get("id")] = sanitize_harmony_tool_name(raw_name)

    for chat_msg in chat_msgs:
        msgs.extend(parse_chat_input_to_harmony_message(chat_msg, tool_id_names))

    msgs = auto_drop_analysis_messages(msgs)
    return msgs


def auto_drop_analysis_messages(msgs: list[Message]) -> list[Message]:
    """
    Harmony models expect the analysis messages (representing raw chain of thought) to
    be dropped after an assistant message to the final channel is produced from the
    reasoning of those messages.

    The openai-harmony library does this if the very last assistant message is to the
    final channel, but it does not handle the case where we're in longer multi-turn
    conversations and the client gave us reasoning content from previous turns of
    the conversation with multiple assistant messages to the final channel in the
    conversation.

    So, we find the index of the last assistant message to the final channel and drop
    all analysis messages that precede it, leaving only the analysis messages that
    are relevant to the current part of the conversation.
    """
    last_assistant_final_index = -1
    for i in range(len(msgs) - 1, -1, -1):
        msg = msgs[i]
        if msg.author.role == "assistant" and msg.channel == "final":
            last_assistant_final_index = i
            break

    cleaned_msgs: list[Message] = []
    for i, msg in enumerate(msgs):
        if i < last_assistant_final_index and msg.channel == "analysis":
            continue
        cleaned_msgs.append(msg)

    return cleaned_msgs


def flatten_chat_text_content(content: str | list | None) -> str | None:
    """
    Extract the text parts from a chat message content field and flatten them
    into a single string.
    """
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return content


def parse_chat_input_to_harmony_message(
    chat_msg, tool_id_names: dict[str, str] | None = None
) -> list[Message]:
    """
    Parse a message from request.messages in the Chat Completion API to
    Harmony messages.
    """
    tool_id_names = tool_id_names or {}

    if not isinstance(chat_msg, dict):
        # Handle Pydantic models
        chat_msg = chat_msg.model_dump(exclude_none=True)

    role = chat_msg.get("role")
    msgs: list[Message] = []

    # Assistant message with tool calls
    tool_calls = chat_msg.get("tool_calls", [])

    if role == "assistant" and tool_calls:
        content = flatten_chat_text_content(chat_msg.get("content"))
        if content:
            commentary_msg = Message.from_role_and_content(Role.ASSISTANT, content)
            commentary_msg = commentary_msg.with_channel("commentary")
            msgs.append(commentary_msg)

        reasoning = chat_msg.get("reasoning")
        if reasoning:
            analysis_msg = Message.from_role_and_content(Role.ASSISTANT, reasoning)
            analysis_msg = analysis_msg.with_channel("analysis")
            msgs.append(analysis_msg)

        for call in tool_calls:
            func = call.get("function", {})
            name = sanitize_harmony_tool_name(func.get("name"))
            arguments = func.get("arguments", "") or ""
            msg = Message.from_role_and_content(Role.ASSISTANT, arguments)
            msg = msg.with_channel("commentary")
            msg = msg.with_recipient(f"functions.{name}")
            # Officially, this should be `<|constrain|>json` but there is not clear
            # evidence that improves accuracy over `json` and some anecdotes to the
            # contrary. Further testing of the different content_types is needed.
            msg = msg.with_content_type("json")
            msgs.append(msg)
        return msgs

    # Tool role message (tool output)
    if role == "tool":
        tool_call_id = chat_msg.get("tool_call_id", "")
        name = sanitize_harmony_tool_name(tool_id_names.get(tool_call_id, ""))
        content = chat_msg.get("content", "") or ""
        content = flatten_chat_text_content(content)

        msg = (
            Message.from_author_and_content(
                Author.new(Role.TOOL, f"functions.{name}"), content
            )
            .with_channel("commentary")
            .with_recipient("assistant")
        )
        return [msg]

    # Non-tool reasoning content
    reasoning = chat_msg.get("reasoning")
    if role == "assistant" and reasoning:
        analysis_msg = Message.from_role_and_content(Role.ASSISTANT, reasoning)
        analysis_msg = analysis_msg.with_channel("analysis")
        msgs.append(analysis_msg)

    # Default: user/assistant/system messages with content
    content = chat_msg.get("content") or ""
    if content is None:
        content = ""
    if isinstance(content, str):
        contents = [TextContent(text=content)]
    else:
        # TODO: Support refusal.
        contents = [TextContent(text=c.get("text", "")) for c in content]

    # Only add assistant messages if they have content, as reasoning or tool calling
    # assistant messages were already added above.
    if role == "assistant" and contents and contents[0].text:
        msg = Message.from_role_and_contents(role, contents)
        # Send non-tool assistant messages to the final channel
        msg = msg.with_channel("final")
        msgs.append(msg)
    # For user/system/developer messages, add them directly even if no content.
    elif role != "assistant":
        msg = Message.from_role_and_contents(role, contents)
        msgs.append(msg)

    return msgs


def parse_input_to_harmony_message(chat_msg) -> list[Message]:
    """
    Parse a message from request.previous_input_messages in the Responsees API to
    Harmony messages.
    """
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
            name = sanitize_harmony_tool_name(func.get("name"))
            arguments = func.get("arguments", "") or ""
            msg = Message.from_role_and_content(Role.ASSISTANT, arguments)
            msg = msg.with_channel("commentary")
            msg = msg.with_recipient(f"functions.{name}")
            msg = msg.with_content_type("json")
            msgs.append(msg)
        return msgs

    # Tool role message (tool output)
    if role == "tool":
        name = sanitize_harmony_tool_name(chat_msg.get("name"))
        content = chat_msg.get("content", "") or ""
        content = flatten_chat_text_content(content)

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


def construct_harmony_previous_input_messages(
    request: ResponsesRequest,
) -> list[OpenAIHarmonyMessage]:
    messages: list[OpenAIHarmonyMessage] = []
    if request.previous_input_messages:
        for message in request.previous_input_messages:
            # Handle both OpenAIHarmonyMessage objects and dictionary inputs
            if isinstance(message, OpenAIHarmonyMessage):
                message_role = message.author.role
                # To match OpenAI, instructions, reasoning and tools are
                # always taken from the most recent Responses API request
                # not carried over from previous requests
                if (
                    message_role == OpenAIHarmonyRole.SYSTEM
                    or message_role == OpenAIHarmonyRole.DEVELOPER
                ):
                    continue
                messages.append(message)
            else:
                harmony_messages = parse_input_to_harmony_message(message)
                for harmony_msg in harmony_messages:
                    message_role = harmony_msg.author.role
                    # To match OpenAI, instructions, reasoning and tools are
                    # always taken from the most recent Responses API request
                    # not carried over from previous requests
                    if (
                        message_role == OpenAIHarmonyRole.SYSTEM
                        or message_role == OpenAIHarmonyRole.DEVELOPER
                    ):
                        continue
                    messages.append(harmony_msg)
    return messages


def render_for_completion(messages: list[Message]) -> list[int]:
    conversation = Conversation.from_messages(messages)
    token_ids = get_encoding().render_conversation_for_completion(
        conversation, Role.ASSISTANT
    )
    return token_ids


def _parse_browser_tool_call(message: Message, recipient: str) -> ResponseOutputItem:
    """Parse browser tool calls (search, open, find) into web search items."""
    if len(message.content) != 1:
        raise ValueError("Invalid number of contents in browser message")
    content = message.content[0]

    # Parse JSON args (with retry detection)
    try:
        browser_call = json.loads(content.text)
    except json.JSONDecodeError:
        json_retry_output_message = (
            f"Invalid JSON args, caught and retried: {content.text}"
        )
        browser_call = {
            "query": json_retry_output_message,
            "url": json_retry_output_message,
            "pattern": json_retry_output_message,
        }

    # Create appropriate action based on recipient
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
            pattern=browser_call.get("pattern", ""),
            url=f"cursor:{browser_call.get('url', '')}",
            type="find",
        )
    else:
        raise ValueError(f"Unknown browser action: {recipient}")

    return ResponseFunctionWebSearch(
        id=f"ws_{random_uuid()}",
        action=action,
        status="completed",
        type="web_search_call",
    )


def _parse_function_call(message: Message, recipient: str) -> list[ResponseOutputItem]:
    """Parse function calls into function tool call items."""
    function_name = sanitize_harmony_tool_name(recipient.split(".")[-1])
    output_items = []
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
    return output_items


def _parse_reasoning(message: Message) -> list[ResponseOutputItem]:
    """Parse reasoning/analysis content into reasoning items."""
    output_items = []
    for content in message.content:
        reasoning_item = ResponseReasoningItem(
            id=f"rs_{random_uuid()}",
            summary=[],
            type="reasoning",
            content=[
                ResponseReasoningTextContent(
                    text=strip_harmony_control_tokens(content.text),
                    type="reasoning_text",
                )
            ],
            status=None,
        )
        output_items.append(reasoning_item)
    return output_items


def _parse_final_message(message: Message) -> ResponseOutputItem:
    """Parse final channel messages into output message items."""
    contents = []
    for content in message.content:
        output_text = ResponseOutputText(
            text=strip_harmony_control_tokens(content.text),
            annotations=[],  # TODO
            type="output_text",
            logprobs=None,  # TODO
        )
        contents.append(output_text)
    return ResponseOutputMessage(
        id=f"msg_{random_uuid()}",
        content=contents,
        role=message.author.role,
        status="completed",
        type="message",
    )


def _parse_mcp_recipient(recipient: str) -> tuple[str, str]:
    """
    Parse MCP recipient into (server_label, tool_name).

    For dotted recipients like "repo_browser.list":
        - server_label: "repo_browser" (namespace/server)
        - tool_name: "list" (specific tool)

    For simple recipients like "filesystem":
        - server_label: "filesystem"
        - tool_name: "filesystem"
    """
    if "." in recipient:
        server_label = recipient.split(".")[0]
        tool_name = recipient.split(".")[-1]
    else:
        server_label = recipient
        tool_name = recipient
    return server_label, tool_name


def _parse_mcp_call(message: Message, recipient: str) -> list[ResponseOutputItem]:
    """Parse MCP calls into MCP call items."""
    server_label, tool_name = _parse_mcp_recipient(recipient)
    output_items = []
    for content in message.content:
        response_item = McpCall(
            arguments=content.text,
            type="mcp_call",
            name=tool_name,
            server_label=server_label,
            id=f"mcp_{random_uuid()}",
            status="completed",
        )
        output_items.append(response_item)
    return output_items


def parse_output_message(message: Message) -> list[ResponseOutputItem]:
    """
    Parse a Harmony message into a list of output response items.
    """
    if message.author.role != "assistant":
        # This is a message from a tool to the assistant (e.g., search result).
        # Don't include it in the final output for now. This aligns with
        # OpenAI's behavior on models like o4-mini.
        return []

    output_items: list[ResponseOutputItem] = []
    recipient = message.recipient

    if recipient is not None:
        # Browser tool calls
        if recipient.startswith("browser."):
            output_items.append(_parse_browser_tool_call(message, recipient))

        # Function calls (should only happen on commentary channel)
        elif message.channel == "commentary" and recipient.startswith("functions."):
            output_items.extend(_parse_function_call(message, recipient))

        # Built-in tools are treated as reasoning
        elif recipient.startswith(("python", "browser", "container")):
            # Built-in tool recipients (python/browser/container)
            # generate reasoning output
            output_items.extend(_parse_reasoning(message))

        # All other recipients are MCP calls
        else:
            output_items.extend(_parse_mcp_call(message, recipient))

    # No recipient - handle based on channel for non-tool messages
    elif message.channel == "analysis":
        output_items.extend(_parse_reasoning(message))

    elif message.channel == "commentary":
        # Per Harmony format, commentary channel can contain preambles to calling
        # multiple functions - explanatory text with no recipient
        output_items.extend(_parse_reasoning(message))

    elif message.channel == "final":
        output_items.append(_parse_final_message(message))

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

    if current_recipient and parser.current_channel in ("commentary", "analysis"):
        if current_recipient.startswith("functions."):
            function_name = sanitize_harmony_tool_name(current_recipient.split(".")[-1])
            rid = random_uuid()
            return [
                ResponseFunctionToolCall(
                    arguments=parser.current_content,
                    call_id=f"call_{rid}",
                    type="function_call",
                    name=function_name,
                    id=f"fc_{rid}",
                    status="in_progress",
                )
            ]
        # Built-in tools (python, browser, container) should be treated as reasoning
        elif not (
            current_recipient.startswith("python")
            or current_recipient.startswith("browser")
            or current_recipient.startswith("container")
        ):
            # All other recipients are MCP calls
            rid = random_uuid()
            server_label, tool_name = _parse_mcp_recipient(current_recipient)
            return [
                McpCall(
                    arguments=parser.current_content,
                    type="mcp_call",
                    name=tool_name,
                    server_label=server_label,
                    id=f"mcp_{rid}",
                    status="in_progress",
                )
            ]

    if parser.current_channel == "commentary":
        return [
            ResponseReasoningItem(
                id=f"rs_{random_uuid()}",
                summary=[],
                type="reasoning",
                content=[
                    ResponseReasoningTextContent(
                        text=strip_harmony_control_tokens(parser.current_content),
                        type="reasoning_text",
                    )
                ],
                status=None,
            )
        ]

    if parser.current_channel == "analysis":
        return [
            ResponseReasoningItem(
                id=f"rs_{random_uuid()}",
                summary=[],
                type="reasoning",
                content=[
                    ResponseReasoningTextContent(
                        text=strip_harmony_control_tokens(parser.current_content),
                        type="reasoning_text",
                    )
                ],
                status=None,
            )
        ]

    if parser.current_channel == "final":
        output_text = ResponseOutputText(
            text=strip_harmony_control_tokens(parser.current_content),
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
    """
    Parse the output of a Harmony chat completion into reasoning and final content.
    Note that when the `openai` tool parser is used, serving_chat only uses this
    for the reasoning content and gets the final content from the tool call parser.

    When the `openai` tool parser is not enabled, or when `GptOssReasoningParser` is
    in use,this needs to return the final content without any tool calls parsed.

    Empty reasoning or final content is returned as None instead of an empty string.
    """
    parser = parse_output_into_messages(token_ids)
    output_msgs = parser.messages
    is_tool_call = False  # TODO: update this when tool call is supported

    # Get completed messages from the parser
    reasoning_texts = [
        msg.content[0].text for msg in output_msgs if msg.channel == "analysis"
    ]
    final_texts = [
        msg.content[0].text for msg in output_msgs if msg.channel != "analysis"
    ]

    # Extract partial messages from the parser
    if parser.current_channel == "analysis" and parser.current_content:
        reasoning_texts.append(parser.current_content)
    elif parser.current_channel != "analysis" and parser.current_content:
        final_texts.append(parser.current_content)

    # Flatten multiple messages into a single string
    reasoning: str | None = "\n".join(reasoning_texts)
    final_content: str | None = "\n".join(final_texts)

    # Return None instead of empty string since existing callers check for None
    reasoning = strip_harmony_control_tokens(reasoning)
    final_content = strip_harmony_control_tokens(final_content)
    reasoning = reasoning or None
    final_content = final_content or None

    return reasoning, final_content, is_tool_call
