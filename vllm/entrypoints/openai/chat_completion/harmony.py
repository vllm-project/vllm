# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Harmony ↔ Chat Completion API conversion utilities.

Handles two directions:
  1. Chat Completion Input → Harmony Messages  (input parsing)
  2. Harmony Messages → Chat Completion Output  (output parsing)
"""

from collections.abc import Iterable, Sequence

from openai_harmony import (
    Author,
    Message,
    Role,
    StreamableParser,
    TextContent,
)

from vllm.entrypoints.openai.harmony import (
    flatten_chat_text_content,
    get_streamable_parser_for_assistant,
)

# ---------------------------------------------------------------------------
# Input parsing: Chat Completion messages → Harmony Messages
# ---------------------------------------------------------------------------


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


def chat_input_to_harmony(
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


def chat_inputs_to_harmony(chat_msgs: list) -> list[Message]:
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
        msgs.extend(chat_input_to_harmony(chat_msg, tool_id_names))

    msgs = auto_drop_analysis_messages(msgs)
    return msgs


# ---------------------------------------------------------------------------
# Output parsing: Harmony Messages → Chat Completion Output
# ---------------------------------------------------------------------------


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
    # - analysis channel: hidden reasoning
    # - commentary channel without recipient (preambles): visible to user
    # - final channel: visible to user
    # - commentary with recipient (tool calls): handled separately by tool parser
    reasoning_texts = [
        msg.content[0].text for msg in output_msgs if msg.channel == "analysis"
    ]
    final_texts = [
        msg.content[0].text
        for msg in output_msgs
        if msg.channel == "final" or (msg.channel == "commentary" and not msg.recipient)
    ]

    # Extract partial messages from the parser
    if parser.current_channel == "analysis" and parser.current_content:
        reasoning_texts.append(parser.current_content)
    elif parser.current_channel == "final" and parser.current_content:
        final_texts.append(parser.current_content)
    elif (
        parser.current_channel == "commentary"
        and not parser.current_recipient
        and parser.current_content
    ):
        # Preambles (commentary without recipient) are visible to user
        final_texts.append(parser.current_content)

    # Flatten multiple messages into a single string
    reasoning: str | None = "\n".join(reasoning_texts)
    final_content: str | None = "\n".join(final_texts)

    # Return None instead of empty string since existing callers check for None
    reasoning = reasoning or None
    final_content = final_content or None

    return reasoning, final_content, is_tool_call
