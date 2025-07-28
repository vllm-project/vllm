# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import datetime
from typing import Literal, Optional

from openai.types.responses import (ResponseInputParam, ResponseOutputMessage,
                                    ResponseOutputText)
from openai_harmony import (Conversation, DeveloperContent,
                            HarmonyEncodingName, Message, ReasoningEffort,
                            Role, StreamableParser, SystemContent, TextContent,
                            load_harmony_encoding)

from vllm.entrypoints.openai.protocol import ResponseReasoningItem
from vllm.utils import random_uuid

REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}

_harmony_encoding = None


def get_encoding():
    global _harmony_encoding
    if _harmony_encoding is None:
        _harmony_encoding = load_harmony_encoding(
            HarmonyEncodingName.HARMONY_GPT_OSS)
    return _harmony_encoding


def get_system_message(
    model_identity: Optional[str] = None,
    reasoning_effort: Optional[Literal["high", "medium", "low"]] = None,
    start_date: Optional[str] = None,
    enable_browsing: bool = True,
    enable_python: bool = True,
) -> Message:
    sys_msg_content = SystemContent.new()
    if model_identity is not None:
        sys_msg_content = sys_msg_content.with_model_identity(model_identity)
    if reasoning_effort is not None:
        sys_msg_content = sys_msg_content.with_reasoning_effort(
            REASONING_EFFORT[reasoning_effort])
    if start_date is None:
        start_date = datetime.datetime.now().strftime("%Y-%m-%d")
    sys_msg_content = sys_msg_content.with_conversation_start_date(start_date)
    if enable_browsing:
        sys_msg_content = sys_msg_content.with_browser_tool()
    if enable_python:
        sys_msg_content = sys_msg_content.with_python_tool()
    sys_msg = Message.from_role_and_content(Role.SYSTEM, sys_msg_content)
    return sys_msg


def get_developer_message(instructions: Optional[str] = None) -> Message:
    dev_msg_content = DeveloperContent.new()
    if instructions is not None:
        dev_msg_content = dev_msg_content.with_instructions(instructions)
    dev_msg = Message.from_role_and_content(Role.DEVELOPER, dev_msg_content)
    return dev_msg


def get_user_message(content: str) -> Message:
    return Message.from_role_and_content(Role.USER, content)


def parse_response_input(response_msg: ResponseInputParam) -> Message:
    role = response_msg["role"]
    content = response_msg["content"]
    if isinstance(content, str):
        msg = Message.from_role_and_content(role, content)
    else:
        contents = [TextContent(text=c["text"]) for c in content]
        msg = Message.from_role_and_contents(role, contents)
    return msg


def parse_response_output(output: ResponseOutputMessage) -> Message:
    role = output.role
    contents = [TextContent(text=c.text) for c in output.content]
    msg = Message.from_role_and_contents(role, contents)
    return msg


def render_for_completion(messages: list[Message]) -> list[int]:
    conversation = Conversation.from_messages(messages)
    token_ids = get_encoding().render_conversation_for_completion(
        conversation, Role.ASSISTANT)
    return token_ids


def get_stop_tokens_for_assistant_actions() -> list[int]:
    return get_encoding().stop_tokens_for_assistant_actions()


def get_streamable_parser_for_assistant() -> StreamableParser:
    return StreamableParser(get_encoding(), role=Role.ASSISTANT)


def parse_output_message(message: Message):
    output_items = []
    if message.channel == "analysis":
        for content in message.content:
            reasoning_item = ResponseReasoningItem(
                text=content.text,
                status=None,  # NOTE: Only the last output item has status.
            )
            output_items.append(reasoning_item)
    elif message.channel == "commentary":
        raise NotImplementedError("Tool call is not supported yet")
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
