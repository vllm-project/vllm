# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import datetime
import json
from collections.abc import Iterable, Sequence
from typing import Literal, Optional, Union

from openai.types.responses import (ResponseFunctionToolCall,
                                    ResponseOutputItem, ResponseOutputMessage,
                                    ResponseOutputText, ResponseReasoningItem)
from openai.types.responses.response_function_web_search import (
    ActionFind, ActionOpenPage, ActionSearch, ResponseFunctionWebSearch)
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent)
from openai.types.responses.tool import Tool
from openai_harmony import (Author, Conversation, DeveloperContent,
                            HarmonyEncodingName, Message, ReasoningEffort,
                            Role, StreamableParser, SystemContent, TextContent,
                            ToolDescription, load_harmony_encoding)

from vllm.entrypoints.openai.protocol import ResponseInputOutputItem
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
    browser_description: Optional[str] = None,
    python_description: Optional[str] = None,
) -> Message:
    sys_msg_content = SystemContent.new()
    if model_identity is not None:
        sys_msg_content = sys_msg_content.with_model_identity(model_identity)
    if reasoning_effort is not None:
        sys_msg_content = sys_msg_content.with_reasoning_effort(
            REASONING_EFFORT[reasoning_effort])
    if start_date is None:
        # NOTE(woosuk): This brings non-determinism in vLLM. Be careful.
        start_date = datetime.datetime.now().strftime("%Y-%m-%d")
    sys_msg_content = sys_msg_content.with_conversation_start_date(start_date)
    if browser_description is not None:
        sys_msg_content = sys_msg_content.with_tools(browser_description)
    if python_description is not None:
        sys_msg_content = sys_msg_content.with_tools(python_description)
    sys_msg = Message.from_role_and_content(Role.SYSTEM, sys_msg_content)
    return sys_msg


def get_developer_message(instructions: Optional[str] = None,
                          tools: Optional[list[Tool]] = None) -> Message:
    dev_msg_content = DeveloperContent.new()
    if instructions is not None:
        dev_msg_content = dev_msg_content.with_instructions(instructions)
    if tools is not None:
        function_tools = []
        for tool in tools:
            if tool.type in ("web_search_preview", "code_interpreter"):
                # These are built-in tools that are added to the system message.
                pass
            elif tool.type == "function":
                function_tools.append(tool)
            else:
                raise ValueError(f"tool type {tool.type} not supported")
        if function_tools:
            function_tool_descriptions = [
                ToolDescription.new(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                ) for tool in function_tools
            ]
            dev_msg_content = dev_msg_content.with_function_tools(
                function_tool_descriptions)
    dev_msg = Message.from_role_and_content(Role.DEVELOPER, dev_msg_content)
    return dev_msg


def get_user_message(content: str) -> Message:
    return Message.from_role_and_content(Role.USER, content)


def parse_response_input(
    response_msg: ResponseInputOutputItem,
    prev_responses: list[Union[ResponseOutputItem, ResponseReasoningItem]]
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
            contents = [
                TextContent(text=text_prefix + c["text"]) for c in content
            ]
            msg = Message.from_role_and_contents(role, contents)
    elif response_msg["type"] == "function_call_output":
        call_id = response_msg["call_id"]
        call_response: Optional[ResponseFunctionToolCall] = None
        for prev_response in reversed(prev_responses):
            if isinstance(prev_response, ResponseFunctionToolCall
                          ) and prev_response.call_id == call_id:
                call_response = prev_response
                break
        if call_response is None:
            raise ValueError(f"No call message found for {call_id}")
        msg = Message.from_author_and_content(
            Author.new(Role.TOOL, f"functions.{call_response.name}"),
            response_msg["output"])
    elif response_msg["type"] == "reasoning":
        content = response_msg["content"]
        assert len(content) == 1
        msg = Message.from_role_and_content(Role.ASSISTANT, content[0]["text"])
    elif response_msg["type"] == "function_call":
        msg = Message.from_role_and_content(Role.ASSISTANT,
                                            response_msg["arguments"])
        msg = msg.with_channel("commentary")
        msg = msg.with_recipient(f"functions.{response_msg['name']}")
        msg = msg.with_content_type("json")
    else:
        raise ValueError(f"Unknown input type: {response_msg['type']}")
    return msg


def parse_chat_input(chat_msg) -> Message:
    role = chat_msg["role"]
    content = chat_msg["content"]
    if isinstance(content, str):
        contents = [TextContent(text=content)]
    else:
        # TODO: Support refusal.
        contents = [TextContent(text=c["text"]) for c in content]
    msg = Message.from_role_and_contents(role, contents)
    return msg


def render_for_completion(messages: list[Message]) -> list[int]:
    conversation = Conversation.from_messages(messages)
    token_ids = get_encoding().render_conversation_for_completion(
        conversation, Role.ASSISTANT)
    return token_ids


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
    if recipient is not None and recipient.startswith("browser."):
        if len(message.content) != 1:
            raise ValueError("Invalid number of contents in browser message")
        content = message.content[0]
        browser_call = json.loads(content.text)
        # TODO: translate to url properly!
        if recipient == "browser.search":
            action = ActionSearch(
                query=f"cursor:{browser_call.get('query', '')}", type="search")
        elif recipient == "browser.open":
            action = ActionOpenPage(
                url=f"cursor:{browser_call.get('url', '')}", type="open_page")
        elif recipient == "browser.find":
            action = ActionFind(pattern=browser_call["pattern"],
                                url=f"cursor:{browser_call.get('url', '')}",
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
    elif message.channel == "analysis":
        for content in message.content:
            reasoning_item = ResponseReasoningItem(
                id=f"rs_{random_uuid()}",
                summary=[],
                type="reasoning",
                content=[
                    ResponseReasoningTextContent(text=content.text,
                                                 type="reasoning_text")
                ],
                status=None,
            )
            output_items.append(reasoning_item)
    elif message.channel == "commentary":
        if message.recipient.startswith("functions."):
            function_name = message.recipient.split(".")[-1]
            for content in message.content:
                random_id = random_uuid()
                response_item = ResponseFunctionToolCall(
                    arguments=content.text,
                    call_id=f"call_{random_id}",
                    type="function_call",
                    name=function_name,
                    id=f"ft_{random_id}",
                )
                output_items.append(response_item)
        elif message.recipient.startswith(
                "python") or message.recipient.startswith("browser"):
            for content in message.content:
                reasoning_item = ResponseReasoningItem(
                    id=f"rs_{random_uuid()}",
                    summary=[],
                    type="reasoning",
                    content=[
                        ResponseReasoningTextContent(text=content.text,
                                                     type="reasoning_text")
                    ],
                    status=None,
                )
                output_items.append(reasoning_item)
        else:
            raise ValueError(f"Unknown recipient: {message.recipient}")
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


def parse_remaining_state(
        parser: StreamableParser) -> list[ResponseOutputItem]:
    if not parser.current_content:
        return []
    if parser.current_role != Role.ASSISTANT:
        return []
    current_recipient = parser.current_recipient
    if (current_recipient is not None
            and current_recipient.startswith("browser.")):
        return []

    if parser.current_channel == "analysis":
        reasoning_item = ResponseReasoningItem(
            id=f"rs_{random_uuid()}",
            summary=[],
            type="reasoning",
            content=[
                ResponseReasoningTextContent(text=parser.current_content,
                                             type="reasoning_text")
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
            status="completed",
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
        token_ids: Sequence[int]) -> tuple[Optional[str], Optional[str], bool]:
    parser = parse_output_into_messages(token_ids)
    output_msgs = parser.messages
    if len(output_msgs) == 0:
        # The generation has stopped during reasoning.
        is_tool_call = False
        reasoning_content = parser.current_content
        final_content = None
    elif len(output_msgs) == 1:
        # The generation has stopped during final message.
        is_tool_call = False
        reasoning_content = output_msgs[0].content[0].text
        final_content = parser.current_content
    else:
        if len(output_msgs) != 2:
            raise ValueError(
                "Expected 2 output messages (reasoning and final), "
                f"but got {len(output_msgs)}.")
        reasoning_msg, final_msg = output_msgs
        reasoning_content = reasoning_msg.content[0].text
        final_content = final_msg.content[0].text
        is_tool_call = final_msg.recipient is not None
    return reasoning_content, final_content, is_tool_call
