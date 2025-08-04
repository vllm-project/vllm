# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import datetime
import json
from typing import Literal, Optional

from openai.types.responses import (ResponseInputParam, ResponseOutputMessage,
                                    ResponseOutputText)
from openai.types.responses.response_function_web_search import (
    ActionFind, ActionOpenPage, ActionSearch, ResponseFunctionWebSearch)
from openai_harmony import (Conversation, DeveloperContent,
                            HarmonyEncodingName, Message, ReasoningEffort,
                            Role, StreamableParser, SystemContent, TextContent,
                            load_harmony_encoding)

from typing import Any

from vllm.entrypoints.openai.protocol import ResponseReasoningItem, ResponseReasoningTextContent
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
    browser_tool: Optional[Any] = None,
    enable_python: bool = True,
    python_tool: Optional[Any] = None,
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
        assert browser_tool is not None
        sys_msg_content = sys_msg_content.with_tools(browser_tool.tool_config)
    if enable_python:
        assert python_tool is not None
        sys_msg_content = sys_msg_content.with_tools(python_tool.tool_config)
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
    if message.author.role != "assistant":
        # This is a message from a tool to the assistant (e.g., search result).
        # Don't include it in the final output for now. TODO: Handle this.
        return []

    output_items = []
    recipient = message.recipient
    if recipient is not None and recipient.startswith("browser."):
        if len(message.content) != 1:
            raise ValueError("Invalid number of contents in browser message")
        content = message.content[0]
        browser_call = json.loads(content.text)
        if recipient == "browser.search":
            action = ActionSearch(query=browser_call["query"], type="search")
        elif recipient == "browser.open":
            url = ""  # FIXME: browser_call["url"]
            action = ActionOpenPage(url=url, type="open_page")
        elif recipient == "browser.find":
            url = ""  # FIXME: browser_call["url"]
            action = ActionFind(pattern=browser_call["pattern"],
                                url=url,
                                type="find")
        else:
            raise ValueError(f"Unknown browser action: {recipient}")
        web_search_item = ResponseFunctionWebSearch(
            id=f"ws_{random_uuid()}",
            action=action,
            status="completed",
            type="web_search_call",
        )
        output_items.append(web_search_item)
    elif message.channel in ("analysis", "commentary"):
        for content in message.content:
            reasoning_item = ResponseReasoningItem(
                content=[ResponseReasoningTextContent(text=content.text)],
                status=None,
            )
            output_items.append(reasoning_item)
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


def parse_output_into_messages(token_ids: list[int]) -> list[Message]:
    parser = get_streamable_parser_for_assistant()
    for token_id in token_ids:
        parser.process(token_id)
    return parser.messages
