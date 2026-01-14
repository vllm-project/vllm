# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Harmony-specific streaming delta extraction for chat completions.

This module handles the extraction of DeltaMessage objects from
harmony parser state during streaming chat completions.
"""

from openai_harmony import StreamableParser

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)


def extract_harmony_streaming_delta(
    harmony_parser: StreamableParser,
    cur_channel: str | None,
    cur_recipient: str | None,
    prev_recipient: str | None,
    delta_text: str,
    include_reasoning: bool,
) -> tuple[DeltaMessage | None, bool]:
    """
    Extract a DeltaMessage from harmony parser state during streaming.

    Args:
        harmony_parser: The StreamableParser instance tracking parse state
        cur_channel: Current channel ("final", "analysis", "commentary", etc.)
        cur_recipient: Current recipient (e.g., "functions.my_func")
        prev_recipient: Previous recipient for detecting tool call transitions
        delta_text: The text delta to include in the message
        include_reasoning: Whether to include reasoning content

    Returns:
        A tuple of (DeltaMessage or None, tools_streamed_flag)
    """
    tools_streamed = False

    if cur_channel == "final":
        delta_message = DeltaMessage(content=delta_text)
    elif (
        (cur_channel == "commentary" or cur_channel == "analysis")
        and cur_recipient
        and cur_recipient.startswith("functions.")
    ):
        # Count completed tool calls to determine index
        base_index = 0
        for msg in harmony_parser.messages:
            if (
                (msg.channel == "commentary" or msg.channel == "analysis")
                and msg.recipient
                and msg.recipient.startswith("functions.")
            ):
                base_index += 1

        if prev_recipient != cur_recipient:
            tool_name = cur_recipient.split("functions.", 1)[1]
            delta_message = DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        id=make_tool_call_id(),
                        type="function",
                        function=DeltaFunctionCall(
                            name=tool_name,
                            arguments="",
                        ),
                        index=base_index,
                    )
                ]
            )
        elif delta_text:
            delta_message = DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=base_index,
                        function=DeltaFunctionCall(arguments=delta_text),
                    )
                ]
            )
        else:
            delta_message = None

        if delta_message is not None:
            tools_streamed = True
    elif cur_channel == "commentary":
        # Tool call preambles meant to be shown to the user
        delta_message = DeltaMessage(content=delta_text)
    elif cur_channel == "analysis":
        if include_reasoning:
            delta_message = DeltaMessage(reasoning=delta_text)
        else:
            delta_message = None
    else:
        delta_message = None

    return delta_message, tools_streamed
