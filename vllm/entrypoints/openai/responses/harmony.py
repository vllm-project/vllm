# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Harmony ↔ Responses API conversion utilities.

Handles two directions:
  1. Response Input → Harmony Messages  (input parsing)
  2. Harmony Messages → Response Output Items  (output parsing)
"""

import json
from dataclasses import dataclass
from enum import Enum, auto

from openai.types.responses import (
    ResponseCodeInterpreterToolCall,
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
from openai_harmony import Author, Message, Role, TextContent

from vllm.entrypoints.openai.parser.harmony_utils import (
    BUILTIN_TOOL_TO_MCP_SERVER_LABEL,
    extract_function_from_recipient,
    flatten_input_text_content,
    get_system_or_developer_message,
    is_function_recipient,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponseInputOutputItem,
    ResponsesRequest,
)
from vllm.logger import init_logger
from vllm.utils import random_uuid

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# 1. Private helpers for input parsing
# ---------------------------------------------------------------------------


def _parse_harmony_format_message(chat_msg: dict) -> Message:
    """Reconstruct a Message from Harmony-format dict,
    preserving channel, recipient, and content_type."""
    author_dict = chat_msg["author"]
    role = author_dict.get("role")
    name = author_dict.get("name")

    raw_content = chat_msg.get("content", "")
    if isinstance(raw_content, list):
        # TODO: Support refusal and non-text content types.
        contents = [TextContent(text=c.get("text", "")) for c in raw_content]
    elif isinstance(raw_content, str):
        contents = [TextContent(text=raw_content)]
    else:
        contents = [TextContent(text="")]

    if name:
        msg = Message.from_author_and_contents(Author.new(Role(role), name), contents)
    else:
        msg = Message.from_role_and_contents(Role(role), contents)

    channel = chat_msg.get("channel")
    if channel:
        msg = msg.with_channel(channel)
    recipient = chat_msg.get("recipient")
    if recipient:
        msg = msg.with_recipient(recipient)
    content_type = chat_msg.get("content_type")
    if content_type:
        msg = msg.with_content_type(content_type)

    return msg


def _parse_chat_format_message(chat_msg: dict) -> list[Message]:
    """Parse an OpenAI chat-format dict into Harmony messages."""
    role = chat_msg.get("role")
    if role is None:
        raise ValueError(f"Message has no 'role' key: {chat_msg}")

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
        if name and not name.startswith("functions."):
            name = f"functions.{name}"
        content = flatten_input_text_content(chat_msg.get("content")) or ""
        # NOTE: .with_recipient("assistant") is required on tool messages
        # to match parse_chat_input_to_harmony_message behavior and ensure
        # proper routing in the Harmony protocol.
        msg = (
            Message.from_author_and_content(Author.new(Role.TOOL, name), content)
            .with_channel("commentary")
            .with_recipient("assistant")
        )
        return [msg]

    # System/developer messages into proper DeveloperContent
    if role in ("system", "developer"):
        text = flatten_input_text_content(chat_msg.get("content"))
        if text:
            msg = get_system_or_developer_message(role, text)
            return [msg]
        return []

    # Default: user/assistant messages
    content = chat_msg.get("content", "")
    if isinstance(content, str):
        contents = [TextContent(text=content)]
    else:
        # TODO: Support refusal.
        contents = [TextContent(text=c.get("text", "")) for c in content]
    msg = Message.from_role_and_contents(role, contents)
    return [msg]


# ---------------------------------------------------------------------------
# 2. Public input parsing functions
# ---------------------------------------------------------------------------


def response_input_to_harmony(
    response_msg: ResponseInputOutputItem,
    prev_responses: list[ResponseOutputItem | ResponseReasoningItem],
) -> Message | None:
    """Convert a single ResponseInputOutputItem into a Harmony Message.

    Returns None for reasoning items with empty or absent content so
    the caller can skip them.
    """
    if not isinstance(response_msg, dict):
        response_msg = response_msg.model_dump()
    if "type" not in response_msg or response_msg["type"] == "message":
        role = response_msg["role"]
        content = response_msg["content"]
        if role in ("system", "developer"):
            text = flatten_input_text_content(content)
            if text:
                msg = get_system_or_developer_message(role, text)
            else:
                # Empty content — skip, no message emitted.
                return None
        elif isinstance(content, str):
            msg = Message.from_role_and_content(role, content)
        else:
            contents = [TextContent(text=c.get("text", "")) for c in content]
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
        msg = msg.with_channel("commentary")
        msg = msg.with_recipient("assistant")
    elif response_msg["type"] == "reasoning":
        content = response_msg.get("content")
        if content and len(content) >= 1:
            reasoning_text = "\n".join(item["text"] for item in content)
            msg = Message.from_role_and_content(Role.ASSISTANT, reasoning_text)
            msg = msg.with_channel("analysis")
        else:
            return None
    elif response_msg["type"] == "function_call":
        msg = Message.from_role_and_content(Role.ASSISTANT, response_msg["arguments"])
        msg = msg.with_channel("commentary")
        msg = msg.with_recipient(f"functions.{response_msg['name']}")
        msg = msg.with_content_type("json")
    else:
        raise ValueError(f"Unknown input type: {response_msg['type']}")
    return msg


def response_previous_input_to_harmony(chat_msg) -> list[Message]:
    """Parse a message from request.previous_input_messages
    into Harmony messages.

    Supports both OpenAI chat format ({"role": "..."}) and
    Harmony format ({"author": {"role": "..."}}).
    """
    if not isinstance(chat_msg, dict):
        chat_msg = chat_msg.model_dump(exclude_none=True)

    if "author" in chat_msg and isinstance(chat_msg.get("author"), dict):
        return [_parse_harmony_format_message(chat_msg)]

    return _parse_chat_format_message(chat_msg)


def construct_harmony_previous_input_messages(
    request: ResponsesRequest,
) -> list[Message]:
    """Build a Harmony message list from request.previous_input_messages.

    Filters out system/developer messages to match OpenAI behavior where
    instructions are always taken from the most recent Responses API request.
    """
    messages: list[Message] = []
    if request.previous_input_messages:
        for message in request.previous_input_messages:
            # Handle both Message objects and dictionary inputs
            if isinstance(message, Message):
                message_role = message.author.role
                if message_role == Role.SYSTEM or message_role == Role.DEVELOPER:
                    continue
                messages.append(message)
            else:
                harmony_messages = response_previous_input_to_harmony(message)
                for harmony_msg in harmony_messages:
                    message_role = harmony_msg.author.role
                    if message_role == Role.SYSTEM or message_role == Role.DEVELOPER:
                        continue
                    messages.append(harmony_msg)
    return messages


# ---------------------------------------------------------------------------
# 3. Private helpers for output parsing
# ---------------------------------------------------------------------------


class ResponseItemKind(Enum):
    REASONING = auto()
    TEXT = auto()
    FUNCTION = auto()
    CODE_INTERPRETER = auto()
    WEB_SEARCH = auto()
    CONTAINER = auto()
    MCP = auto()
    IGNORE = auto()


@dataclass(frozen=True)
class ResponseItemType:
    """Normalized response target derived from a Harmony channel/recipient."""

    kind: ResponseItemKind
    name: str | None = None
    action: str | None = None


_VALID_BROWSER_ACTIONS = frozenset({"search", "open", "find"})
_VALID_CONTAINER_ACTIONS = frozenset({"exec"})


def resolve_response_item_type(
    channel: str | None,
    recipient: str | None,
    function_tool_names: frozenset[str] | None = None,
) -> ResponseItemType:
    """Classify and normalize Harmony channel/recipient pairs."""
    if recipient == "assistant":
        # TODO: In reality, we should emit built-in tool calls
        # based on whether the tool result was successful or not.
        # For now, we assume ignore the actual result and assumes
        # tool calls are always successful.
        return ResponseItemType(ResponseItemKind.IGNORE)

    if recipient:
        if is_function_recipient(recipient, function_tool_names):
            return ResponseItemType(
                ResponseItemKind.FUNCTION,
                name="functions",
                action=extract_function_from_recipient(recipient),
            )

        recipient_parts = recipient.split(".", 1)
        name = recipient_parts[0].lower()
        name_or_server_label = BUILTIN_TOOL_TO_MCP_SERVER_LABEL.get(name, name)
        action = recipient_parts[1].lower() if len(recipient_parts) > 1 else None

        if name_or_server_label == "code_interpreter":
            return ResponseItemType(
                ResponseItemKind.CODE_INTERPRETER,
                name_or_server_label,
            )

        if name_or_server_label == "web_search_preview":
            return ResponseItemType(
                ResponseItemKind.WEB_SEARCH,
                name_or_server_label,
                action if action in _VALID_BROWSER_ACTIONS else "search",
            )

        if name_or_server_label == "container":
            return ResponseItemType(
                ResponseItemKind.CONTAINER,
                name_or_server_label,
                action if action in _VALID_CONTAINER_ACTIONS else "exec",
            )

        return ResponseItemType(
            ResponseItemKind.MCP,
            name_or_server_label,
            action or name_or_server_label,
        )

    if channel == "analysis":
        return ResponseItemType(ResponseItemKind.REASONING)

    if channel in ("commentary", "final"):
        return ResponseItemType(ResponseItemKind.TEXT)

    return ResponseItemType(ResponseItemKind.IGNORE)


def build_code_interpreter_call(
    item_type: ResponseItemType,
    code: str,
    item_id: str,
    status: str = "completed",
) -> ResponseCodeInterpreterToolCall:
    assert item_type.kind == ResponseItemKind.CODE_INTERPRETER
    return ResponseCodeInterpreterToolCall(
        id=item_id,
        code=code,
        container_id="auto",
        outputs=[],
        status=status,
        type="code_interpreter_call",
    )


def build_web_search_call(
    item_type: ResponseItemType,
    arguments: str,
    item_id: str,
) -> ResponseFunctionWebSearch:
    assert item_type.kind == ResponseItemKind.WEB_SEARCH
    browser_action = item_type.action

    try:
        browser_args = json.loads(arguments)
    except json.JSONDecodeError as e:
        # ResponseFunctionWebSearch does not allow an error message
        # so we return a failed MCP call instead
        return McpCall(
            arguments=arguments,
            name=browser_action,
            server_label="browser",
            id=item_id,
            status="failed",
            error=f"Invalid JSON arguments: {e}",
            type="mcp_call",
        )

    match browser_action:
        case "search":
            action = ActionSearch(
                query=f"cursor:{browser_args.get('query', '')}",
                type="search",
            )
        case "open":
            action = ActionOpenPage(
                type="open_page",
                url=f"cursor:{browser_args.get('url', '')}",
            )
        case "find":
            action = ActionFind(
                pattern=browser_args.get("pattern", ""),
                type="find_in_page",
                url=f"cursor:{browser_args.get('url', '')}",
            )
        case _:
            raise ValueError(f"Invalid browser action: {browser_action}")

    return ResponseFunctionWebSearch(
        id=item_id,
        action=action,
        status="completed",
        type="web_search_call",
    )


def build_mcp_or_container_call(
    item_type: ResponseItemType,
    arguments: str,
    item_id: str,
    status: str = "completed",
) -> McpCall:
    assert item_type.kind in (ResponseItemKind.MCP, ResponseItemKind.CONTAINER)
    server_label, name = item_type.name, item_type.action

    return McpCall(
        arguments=arguments,
        name=name,
        server_label=server_label,
        id=item_id,
        status=status,
        type="mcp_call",
    )


def build_reasoning(
    item_type: ResponseItemType,
    content: list[str],
    item_id: str,
) -> ResponseReasoningItem:
    """TODO: Unify ResponseReasoningItem creation between
    Harmony and non-Harmony and streaming and non-streaming"""
    assert item_type.kind == ResponseItemKind.REASONING
    return ResponseReasoningItem(
        id=item_id,
        summary=[],
        type="reasoning",
        content=[
            ResponseReasoningTextContent(text=c, type="reasoning_text") for c in content
        ],
    )


def build_output_message(
    item_type: ResponseItemType,
    content: list[str],
    item_id: str,
) -> ResponseOutputMessage:
    """TODO: Unify ResponseOutputMessage creation between
    Harmony and non-Harmony and streaming and non-streaming"""
    assert item_type.kind == ResponseItemKind.TEXT
    return ResponseOutputMessage(
        id=item_id,
        content=[
            ResponseOutputText(text=c, annotations=[], type="output_text")
            for c in content
        ],
        role="assistant",
        status="completed",
        type="message",
    )


def build_function_call(
    item_type: ResponseItemType,
    arguments: str,
    call_id: str,
    item_id: str,
) -> ResponseFunctionToolCall:
    """TODO: Unify ResponseFunctionToolCall creation between
    Harmony and non-Harmony and streaming and non-streaming"""
    assert item_type.kind == ResponseItemKind.FUNCTION

    try:
        arguments = json.dumps(json.loads(arguments))
    except json.JSONDecodeError:
        # Ignore JSON decode error
        arguments = arguments.strip()

    return ResponseFunctionToolCall(
        id=item_id,
        arguments=arguments,
        call_id=call_id,
        name=item_type.action,
        type="function_call",
    )


def message_text_content(message: Message) -> list[str]:
    """Extract text content from a Harmony message."""
    return [c.text for c in message.content]


# ---------------------------------------------------------------------------
# 4. Public output parsing functions
# ---------------------------------------------------------------------------


def harmony_to_response_output(
    message: Message,
    function_tool_names: frozenset[str] | None = None,
) -> list[ResponseOutputItem]:
    """Parse a Harmony message into a list of output response items.

    This is the main dispatcher that routes based on channel and recipient.
    """
    item_type = resolve_response_item_type(
        message.channel,
        message.recipient,
        function_tool_names,
    )
    content = message_text_content(message)
    match item_type.kind:
        case ResponseItemKind.REASONING:
            return [build_reasoning(item_type, content, f"rs_{random_uuid()}")]
        case ResponseItemKind.TEXT:
            return [build_output_message(item_type, content, f"msg_{random_uuid()}")]
        case ResponseItemKind.FUNCTION:
            rid = random_uuid()
            item_id = f"fc_{rid}"
            tool_call_id = f"call_{rid}"
            return [build_function_call(item_type, content[0], tool_call_id, item_id)]
        case ResponseItemKind.CODE_INTERPRETER:
            return [
                build_code_interpreter_call(
                    item_type, content[0], f"ci_{random_uuid()}"
                )
            ]
        case ResponseItemKind.WEB_SEARCH:
            return [build_web_search_call(item_type, content[0], f"ws_{random_uuid()}")]
        case ResponseItemKind.CONTAINER:
            return [
                build_mcp_or_container_call(
                    item_type, content[0], f"container_{random_uuid()}"
                )
            ]
        case ResponseItemKind.MCP:
            return [
                build_mcp_or_container_call(
                    item_type, content[0], f"mcp_{random_uuid()}"
                )
            ]
        case ResponseItemKind.IGNORE:
            return []
