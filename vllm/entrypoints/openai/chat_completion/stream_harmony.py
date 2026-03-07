# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Harmony-specific streaming delta extraction for chat completions.

This module handles the extraction of DeltaMessage objects from
harmony parser state during streaming chat completions.
"""

from dataclasses import dataclass
from typing import NamedTuple

from openai_harmony import StreamableParser

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)


class TokenState(NamedTuple):
    channel: str | None
    recipient: str | None
    text: str


@dataclass
class HarmonyStreamingState:
    emitted_message_count: int = 0
    next_tool_call_index: int = 0
    prev_current_signature: tuple[str | None, str | None] | None = None
    prev_current_emitted_len: int = 0
    prev_current_tool_index: int | None = None
    prev_current_tool_header_emitted: bool = False


def _is_function_tool_message(channel: str | None, recipient: str | None) -> bool:
    return (
        channel in ("commentary", "analysis")
        and recipient is not None
        and recipient.startswith("functions.")
    )


def _extract_message_text(message) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            else:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    texts.append(text)
        return "".join(texts)

    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text

    return ""


def _append_tool_deltas(
    *,
    tool_messages: list[DeltaToolCall],
    recipient: str,
    tool_index: int,
    emit_header: bool,
    args_delta: str,
) -> None:
    if emit_header:
        tool_messages.append(
            DeltaToolCall(
                id=make_tool_call_id(),
                type="function",
                function=DeltaFunctionCall(
                    name=recipient.split("functions.", 1)[1],
                    arguments="",
                ),
                index=tool_index,
            )
        )

    if args_delta:
        tool_messages.append(
            DeltaToolCall(
                index=tool_index,
                function=DeltaFunctionCall(arguments=args_delta),
            )
        )


def extract_harmony_streaming_delta(
    harmony_parser: StreamableParser,
    stream_state: HarmonyStreamingState,
    include_reasoning: bool,
) -> tuple[DeltaMessage | None, bool]:
    """
    Extract a DeltaMessage from harmony parser state during streaming.

    Unlike the previous token-group heuristic, this function diffs parser-level
    completed messages and parser current_content against persistent state.
    This makes tool call indexing and argument streaming robust across arbitrary
    chunk boundaries, including repeated recipients.
    """

    tool_messages: list[DeltaToolCall] = []
    combined_content = ""
    combined_reasoning = ""
    content_encountered = False

    prev_signature = stream_state.prev_current_signature
    prev_emitted_len = stream_state.prev_current_emitted_len
    prev_tool_index = stream_state.prev_current_tool_index
    prev_tool_header_emitted = stream_state.prev_current_tool_header_emitted

    carryover_consumed = False

    for msg in harmony_parser.messages[stream_state.emitted_message_count :]:
        channel = msg.channel
        recipient = msg.recipient
        signature = (channel, recipient)
        msg_text = _extract_message_text(msg)

        already_emitted_len = 0
        tool_index: int | None = None
        tool_header_emitted = False

        if (
            not carryover_consumed
            and prev_signature is not None
            and signature == prev_signature
        ):
            already_emitted_len = min(prev_emitted_len, len(msg_text))
            tool_index = prev_tool_index
            tool_header_emitted = prev_tool_header_emitted
            carryover_consumed = True

        delta_text = msg_text[already_emitted_len:]

        if _is_function_tool_message(channel, recipient):
            assert recipient is not None
            if tool_index is None:
                tool_index = stream_state.next_tool_call_index
                stream_state.next_tool_call_index += 1
            _append_tool_deltas(
                tool_messages=tool_messages,
                recipient=recipient,
                tool_index=tool_index,
                emit_header=not tool_header_emitted,
                args_delta=delta_text,
            )
        elif channel in ("final", "commentary"):
            if delta_text:
                combined_content += delta_text
                content_encountered = True
        elif channel == "analysis" and include_reasoning and delta_text:
            combined_reasoning += delta_text

    stream_state.emitted_message_count = len(harmony_parser.messages)

    current_channel = harmony_parser.current_channel
    current_recipient = harmony_parser.current_recipient
    current_signature: tuple[str | None, str | None] | None = None
    if current_channel is not None:
        current_signature = (current_channel, current_recipient)
    current_content = harmony_parser.current_content or ""

    if current_signature is None:
        stream_state.prev_current_signature = None
        stream_state.prev_current_emitted_len = 0
        stream_state.prev_current_tool_index = None
        stream_state.prev_current_tool_header_emitted = False
    else:
        same_current_message = (
            not carryover_consumed
            and prev_signature is not None
            and current_signature == prev_signature
            and len(current_content) >= prev_emitted_len
        )

        if same_current_message:
            already_emitted_len = prev_emitted_len
            current_tool_index = prev_tool_index
            current_header_emitted = prev_tool_header_emitted
        else:
            already_emitted_len = 0
            current_tool_index = None
            current_header_emitted = False

        delta_text = current_content[already_emitted_len:]

        if _is_function_tool_message(current_channel, current_recipient):
            assert current_recipient is not None
            if current_tool_index is None:
                current_tool_index = stream_state.next_tool_call_index
                stream_state.next_tool_call_index += 1

            _append_tool_deltas(
                tool_messages=tool_messages,
                recipient=current_recipient,
                tool_index=current_tool_index,
                emit_header=not current_header_emitted,
                args_delta=delta_text,
            )

            stream_state.prev_current_tool_index = current_tool_index
            stream_state.prev_current_tool_header_emitted = True
        else:
            if current_channel in ("final", "commentary"):
                if delta_text:
                    combined_content += delta_text
                    content_encountered = True
            elif current_channel == "analysis" and include_reasoning and delta_text:
                combined_reasoning += delta_text

            stream_state.prev_current_tool_index = None
            stream_state.prev_current_tool_header_emitted = False

        stream_state.prev_current_signature = current_signature
        stream_state.prev_current_emitted_len = len(current_content)

    if not (content_encountered or combined_reasoning or tool_messages):
        return None, False

    delta_kwargs: dict[str, str | list[DeltaToolCall]] = {}
    if content_encountered:
        delta_kwargs["content"] = combined_content
    if combined_reasoning:
        delta_kwargs["reasoning"] = combined_reasoning
    if tool_messages:
        delta_kwargs["tool_calls"] = tool_messages

    return DeltaMessage(**delta_kwargs), bool(tool_messages)
