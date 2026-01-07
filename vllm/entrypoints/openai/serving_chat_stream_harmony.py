# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Harmony-specific streaming delta extraction for chat completions.

This module handles the extraction of DeltaMessage objects from
harmony parser state during streaming chat completions.
"""

from openai_harmony import StreamableParser

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)


def extract_harmony_streaming_delta(
    harmony_parser: StreamableParser,
    token_states: list[tuple[str | None, str | None, str]],
    prev_recipient: str | None,
    include_reasoning: bool,
) -> tuple[DeltaMessage | None, bool]:
    """
    Extract a DeltaMessage from harmony parser state during streaming.

    Args:
        harmony_parser: The StreamableParser instance tracking parse state
        token_states: List of (channel, recipient, text) tuples for each token
        prev_recipient: Previous recipient for detecting tool call transitions
        include_reasoning: Whether to include reasoning content

    Returns:
        A tuple of (DeltaMessage or None, tools_streamed_flag)
    """
    tools_streamed = False

    # Group consecutive tokens with same channel/recipient
    groups: list[dict[str, str | None]] = []
    for channel, recipient, text in token_states:
        if (
            groups
            and groups[-1]["channel"] == channel
            and groups[-1]["recipient"] == recipient
        ):
            current_text = groups[-1]["text"] or ""
            groups[-1]["text"] = current_text + text
        else:
            groups.append(
                {
                    "channel": channel,
                    "recipient": recipient,
                    "text": text,
                }
            )

    # Process each group and create delta messages
    delta_message = None
    combined_content = ""
    combined_reasoning = ""
    tool_messages = []

    # Calculate base_index once before the loop
    # This counts completed tool calls in messages
    base_index = 0
    for msg in harmony_parser.messages:
        if (
            (msg.channel == "commentary" or msg.channel == "analysis")
            and msg.recipient
            and msg.recipient.startswith("functions.")
        ):
            base_index += 1

    # If there's an ongoing tool call from previous chunk,
    # the next new tool call starts at base_index + 1
    if prev_recipient and prev_recipient.startswith("functions."):
        next_tool_index = base_index + 1
        # Ongoing call is at base_index
        ongoing_tool_index = base_index
    else:
        # No ongoing call, next new call is at base_index
        next_tool_index = base_index
        ongoing_tool_index = None

    for group in groups:
        group_channel = group["channel"]
        group_recipient = group["recipient"]
        group_text = group["text"]

        if group_channel == "final":
            combined_content += group_text or ""
        elif (
            (group_channel == "commentary" or group_channel == "analysis")
            and group_recipient
            and group_recipient.startswith("functions.")
        ):
            opened_new_call = False
            if prev_recipient != group_recipient:
                # New tool call - emit the opening message
                tool_name = group_recipient.split("functions.", 1)[1]
                tool_messages.append(
                    DeltaToolCall(
                        id=make_tool_call_id(),
                        type="function",
                        function=DeltaFunctionCall(
                            name=tool_name,
                            arguments="",
                        ),
                        index=next_tool_index,
                    )
                )
                opened_new_call = True
                prev_recipient = group_recipient
                # Increment for subsequent new tool calls
                next_tool_index += 1

            if group_text:
                # Stream arguments for the ongoing tool call
                if opened_new_call:
                    # Just opened in this group
                    tool_call_index = next_tool_index - 1
                else:
                    # Continuing from previous chunk
                    # If ongoing_tool_index is None here, it means
                    # we're continuing a call but prev_recipient
                    # wasn't a function. Use base_index.
                    tool_call_index = (
                        ongoing_tool_index
                        if ongoing_tool_index is not None
                        else base_index
                    )
                tool_messages.append(
                    DeltaToolCall(
                        index=tool_call_index,
                        function=DeltaFunctionCall(arguments=group_text),
                    )
                )
        elif group_channel == "commentary":
            # Tool call preambles meant to be shown to the user
            combined_content += group_text or ""
        elif group_channel == "analysis" and include_reasoning:
            combined_reasoning += group_text or ""

    # Combine all non-empty fields into a single message
    if combined_content or combined_reasoning or tool_messages:
        delta_kwargs: dict[str, str | list[DeltaToolCall]] = {}
        if combined_content:
            delta_kwargs["content"] = combined_content
        if combined_reasoning:
            delta_kwargs["reasoning"] = combined_reasoning
        if tool_messages:
            delta_kwargs["tool_calls"] = tool_messages
            tools_streamed = True
        delta_message = DeltaMessage(**delta_kwargs)
    else:
        delta_message = None

    return delta_message, tools_streamed
