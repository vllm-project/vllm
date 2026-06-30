# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import datetime
from collections.abc import Sequence
from typing import Any

from openai.types.responses.tool import Tool
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    RenderConversationConfig,
    Role,
    StreamableParser,
    SystemContent,
    TextContent,
    ToolDescription,
    load_harmony_encoding,
)

from vllm import envs
from vllm.entrypoints.chat_utils import get_supported_content_part_types
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

logger = init_logger(__name__)


def is_function_recipient(
    recipient: str,
    allowed_function_tool_names: frozenset[str] | None = None,
) -> bool:
    """Check whether *recipient* refers to a function tool call.

    The optional *allowed_function_tool_names* parameter is used by the
    Responses API to distinguish bare function-call recipients (missing the
    ``functions.`` prefix) from MCP tool calls.  When provided, a bare
    recipient is only treated as a function call if it appears in the set.
    The Chat Completions path omits this parameter so that all bare
    recipients are accepted as function calls (the heuristic fallback).
    """
    if not recipient:
        return False
    if recipient.startswith("<|"):
        return False
    if recipient.startswith("functions."):
        return len(recipient) > len("functions.")
    if recipient == "assistant":
        return False
    if recipient in BUILTIN_TOOL_TO_MCP_SERVER_LABEL:
        return False
    first_segment = recipient.split(".", 1)[0]
    if first_segment in BUILTIN_TOOL_TO_MCP_SERVER_LABEL:
        return False
    if allowed_function_tool_names is not None:
        return recipient in allowed_function_tool_names
    return True


def extract_function_from_recipient(recipient: str) -> str:
    return recipient.removeprefix("functions.")


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
BUILTIN_TOOL_TO_MCP_SERVER_LABEL: dict[str, str] = {
    "python": "code_interpreter",
    "browser": "web_search_preview",
    "container": "container",
}

# Derive MCP_BUILTIN_TOOLS from the canonical mapping
MCP_BUILTIN_TOOLS: set[str] = set(BUILTIN_TOOL_TO_MCP_SERVER_LABEL.values())


def has_custom_tools(tool_types: set[str]) -> bool:
    """
    Checks if the given tool types are custom tools
    (i.e. any tool other than MCP builtin tools)
    """
    return not tool_types.issubset(MCP_BUILTIN_TOOLS)


def get_encoding():
    global _harmony_encoding
    if _harmony_encoding is None:
        _harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _harmony_encoding


def get_system_message(
    model_identity: str | None = None,
    reasoning_effort: str | None = None,
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
        if reasoning_effort not in REASONING_EFFORT:
            supported_values = ", ".join(REASONING_EFFORT)
            raise ValueError(
                f"reasoning_effort={reasoning_effort!r} is not supported by "
                f"Harmony. Supported values are: {supported_values}."
            )
        sys_msg_content = sys_msg_content.with_reasoning_effort(
            REASONING_EFFORT[reasoning_effort]
        )
    if start_date is None:
        # NOTE(woosuk): This brings non-determinism in vLLM.
        # Set VLLM_SYSTEM_START_DATE to pin it.
        start_date = envs.VLLM_SYSTEM_START_DATE or datetime.datetime.now().strftime(
            "%Y-%m-%d"
        )
    sys_msg_content = sys_msg_content.with_conversation_start_date(start_date)
    if browser_description is not None:
        sys_msg_content = sys_msg_content.with_tools(browser_description)
    if python_description is not None:
        sys_msg_content = sys_msg_content.with_tools(python_description)
    if container_description is not None:
        sys_msg_content = sys_msg_content.with_tools(container_description)
    sys_msg = Message.from_role_and_content(Role.SYSTEM, sys_msg_content)
    return sys_msg


def create_tool_definition(tool: ChatCompletionToolsParam | Tool):
    if isinstance(tool, ChatCompletionToolsParam):
        return ToolDescription.new(
            name=tool.function.name,
            description=tool.function.description or "",
            parameters=tool.function.parameters,
        )
    return ToolDescription.new(
        name=tool.name,
        description=tool.description or "",
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


def get_system_or_developer_message(role: str, instructions: str) -> Message:
    if role == "system" and envs.VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS:
        return get_system_message(instructions=instructions)
    return get_developer_message(instructions=instructions)


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
            tool_id_names[tool_call.get("id")] = tool_call.get("function", {}).get(
                "name"
            )

    for chat_msg in chat_msgs:
        msgs.extend(parse_chat_input_to_harmony_message(chat_msg, tool_id_names))

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


def flatten_input_text_content(content: Any) -> str | None:
    """
    Extract text parts from a Chat Completion or Responses API content field and
    flatten them into a single string. Returns None if no text content is found.
    """
    if content is None or isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None

    texts: list[str] = []
    for item in content:
        if isinstance(item, str):
            texts.append(item)
            continue
        if isinstance(item, dict):
            text = item.get("text")
            if text is not None:
                texts.append(text)
    return "".join(texts) if texts else None


def extract_instructions_from_messages(
    messages: Sequence[Any],
) -> tuple[str | None, list[Any]]:
    """
    Peel a leading system/developer Chat Completion or Responses message and
    flatten its instruction text.
    """
    remaining_messages = list(messages)
    if not remaining_messages:
        return None, remaining_messages

    first_message = remaining_messages[0]
    if not isinstance(first_message, dict):
        if hasattr(first_message, "to_dict"):
            # Handle OpenAI Harmony Message
            first_message = first_message.to_dict()
        elif hasattr(first_message, "model_dump"):
            first_message = first_message.model_dump(exclude_none=True)
        else:
            raise ValueError(f"Unknown message type: {type(first_message)}")

    if first_message.get("role") not in (
        "system",
        "developer",
    ):
        return None, remaining_messages

    instructions = flatten_input_text_content(first_message.get("content"))
    return instructions, remaining_messages[1:]


def build_harmony_preamble(
    *,
    instructions: str | None = None,
    tools: list[Tool | ChatCompletionToolsParam] | None = None,
    reasoning_effort: str | None = None,
    browser_description: str | None = None,
    python_description: str | None = None,
    container_description: str | None = None,
    with_custom_tools: bool = False,
) -> list[Message]:
    """
    Build the standard Harmony system/developer prefix for a request.
    """
    developer_instructions = system_instructions = None
    if envs.VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS:
        system_instructions = instructions
    else:
        developer_instructions = instructions

    messages = [
        get_system_message(
            reasoning_effort=reasoning_effort,
            browser_description=browser_description,
            python_description=python_description,
            container_description=container_description,
            instructions=system_instructions,
            with_custom_tools=with_custom_tools,
        )
    ]
    if developer_instructions or tools:
        messages.append(
            get_developer_message(
                instructions=developer_instructions,
                tools=tools,
            )
        )
    return messages


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
        content = flatten_input_text_content(chat_msg.get("content"))
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
            name = func.get("name", "")
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
        name = tool_id_names.get(tool_call_id, "")
        content = flatten_input_text_content(chat_msg.get("content")) or ""

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
        # Validate content part types - match non-Harmony path validation
        # Use shared helper to ensure consistency across all parsing paths
        supported_types = get_supported_content_part_types()
        for c in content:
            if isinstance(c, dict):
                part_type = c.get("type")
                if part_type and part_type not in supported_types:
                    raise VLLMValidationError(
                        f"Unsupported chat content part type: {part_type!r}. "
                        f"Supported types: {', '.join(sorted(supported_types))}.",
                        parameter="type",
                        value=part_type,
                    )
        contents = [TextContent(text=c.get("text", "")) for c in content]

    # Only add assistant messages if they have content, as reasoning or tool calling
    # assistant messages were already added above.
    if role == "assistant" and contents and contents[0].text:
        msg = Message.from_role_and_contents(role, contents)
        # Send non-tool assistant messages to the final channel
        msg = msg.with_channel("final")
        msgs.append(msg)
    elif role in ("system", "developer"):
        instructions = flatten_input_text_content(chat_msg.get("content"))
        if instructions is not None:
            msg = get_system_or_developer_message(role, instructions)
            msgs.append(msg)
    # For user messages, add them directly even if no content.
    elif role != "assistant":
        msg = Message.from_role_and_contents(role, contents)
        msgs.append(msg)

    return msgs


def render_for_completion(messages: list[Message]) -> list[int]:
    messages = auto_drop_analysis_messages(messages)
    conversation = Conversation.from_messages(messages)
    token_ids = get_encoding().render_conversation_for_completion(
        conversation,
        Role.ASSISTANT,
        config=RenderConversationConfig(auto_drop_analysis=False),
    )
    return token_ids


def get_streamable_parser_for_assistant() -> StreamableParser:
    return StreamableParser(get_encoding(), role=Role.ASSISTANT)
