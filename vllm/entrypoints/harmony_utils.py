# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import datetime
import json
from collections.abc import Iterable, Sequence
from typing import Literal

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
from vllm.entrypoints.openai.protocol import (
    ChatCompletionToolsParam,
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

MCP_BUILTIN_TOOLS: set[str] = {
    "web_search_preview",
    "code_interpreter",
    "container",
}


def has_custom_tools(tool_types: set[str]) -> bool:
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
                "mcp",
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
        if role == "system":
            role = "developer"
            text_prefix = "Instructions:\n"
        else:
            text_prefix = ""
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
        msg = msg.with_recipient(f"functions.{response_msg['name']}")
        msg = msg.with_content_type("json")
    else:
        raise ValueError(f"Unknown input type: {response_msg['type']}")
    return msg


def parse_input_to_harmony_message(chat_msg) -> list[Message]:
    if not isinstance(chat_msg, dict):
        chat_msg = chat_msg.model_dump(exclude_none=True)

    role = chat_msg.get("role")
    tool_calls = chat_msg.get("tool_calls")
    if role == "assistant" and tool_calls:
        msgs: list[Message] = []
        for call in tool_calls:
            func = call.get("function", {})
            name = func.get("name", "")
            arguments = func.get("arguments", "") or ""
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments)
            
            msg = Message.from_role_and_content(Role.ASSISTANT, arguments)
            msg = msg.with_channel("commentary")
            msg = msg.with_recipient(f"functions.{name}")
            msg = msg.with_content_type("json")
            msgs.append(msg)
        return msgs

    if role == "tool":
        name = chat_msg.get("name", "")
        content = chat_msg.get("content", "") or ""
        if isinstance(content, list):
            content = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )

        msg = Message.from_author_and_content(
            Author.new(Role.TOOL, f"functions.{name}"), content
        ).with_channel("commentary")
        return [msg]

    content = chat_msg.get("content", "")
    if isinstance(content, str):
        contents = [TextContent(text=content)]
    else:
        contents = [TextContent(text=c.get("text", "")) for c in content]
    msg = Message.from_role_and_contents(role, contents)
    return [msg]


def construct_harmony_previous_input_messages(
    request: ResponsesRequest,
) -> list[OpenAIHarmonyMessage]:
    messages: list[OpenAIHarmonyMessage] = []
    if request.previous_input_messages:
        for message in request.previous_input_messages:
            if isinstance(message, OpenAIHarmonyMessage):
                message_role = message.author.role
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


def parse_output_message(message: Message) -> list[ResponseOutputItem]:
    if message.author.role != "assistant":
        return []

    output_items: list[ResponseOutputItem] = []
    recipient = message.recipient
    
    if recipient is not None and recipient.startswith("browser."):
        if len(message.content) != 1:
            raise ValueError("Invalid number of contents in browser message")
        content = message.content[0]
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
                pattern=browser_call["pattern"],
                url=f"cursor:{browser_call.get('url', '')}",
                type="find",
            )
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
                    ResponseReasoningTextContent(
                        text=content.text, type="reasoning_text"
                    )
                ],
                status=None,
            )
            output_items.append(reasoning_item)
            
    elif message.channel == "commentary":
        if recipient is not None and recipient.startswith("functions."):
            # FIX: Strict name sanitization to remove leaked tags like <|channel|>
            raw_name = recipient.split("functions.")[1]
            function_name = raw_name.split("<")[0].strip()
            
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
                
        elif recipient is not None and (
            recipient.startswith("python")
            or recipient.startswith("browser")
            or recipient.startswith("container")
        ):
            for content in message.content:
                reasoning_item = ResponseReasoningItem(
                    id=f"rs_{random_uuid()}",
                    summary=[],
                    type="reasoning",
                    content=[
                        ResponseReasoningTextContent(
                            text=content.text, type="reasoning_text"
                        )
                    ],
                    status=None,
                )
                output_items.append(reasoning_item)
        else:
            raise ValueError(f"Unknown recipient: {recipient}")
            
    elif message.channel == "final":
        contents = []
        for content in message.content:
            output_text = ResponseOutputText(
                text=content.text,
                annotations=[],  
                type="output_text",
                logprobs=None,
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


def parse_remaining_state(parser: StreamableParser) -> list[ResponseOutputItem]:
    if not parser.current_content:
        return []
    if parser.current_role != Role.ASSISTANT:
        return []
    current_recipient = parser.current_recipient
    if current_recipient is not None and current_recipient.startswith("browser."):
        return []

    if parser.current_channel == "analysis":
        reasoning_item = ResponseReasoningItem(
            id=f"rs_{random_uuid()}",
            summary=[],
            type="reasoning",
            content=[
                ResponseReasoningTextContent(
                    text=parser.current_content, type="reasoning_text"
                )
            ],
            status=None,
        )
        return [reasoning_item]
        
    elif parser.current_channel == "commentary":
        if current_recipient is not None and current_recipient.startswith("functions."):
            # FIX: Strict name sanitization here as well
            raw_name = current_recipient.split("functions.")[1]
            function_name = raw_name.split("<")[0].strip()
            
            random_id = random_uuid()
            response_item = ResponseFunctionToolCall(
                arguments=parser.current_content,
                call_id=f"call_{random_id}",
                type="function_call",
                name=function_name,
                id=f"fc_{random_id}",
            )
            return [response_item]
            
    elif parser.current_channel == "final":
        output_text = ResponseOutputText(
            text=parser.current_content,
            annotations=[],  
            type="output_text",
            logprobs=None, 
        )
        text_item = ResponseOutputMessage(
            id=f"msg_{random_uuid()}",
            content=[output_text],
            role="assistant",
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
    
    tokens = list(token_ids)
    if not tokens:
        return parser

    encoding = get_encoding()
    
    # FIX: Use allowed_special="all" to avoid Tokenizer error
    start_token = encoding.encode("<|start|>", allowed_special="all")[0]
    
    if tokens[0] != start_token:
        def get_id(text):
            return encoding.encode(text, allowed_special="all")[0]

        header_tokens = [
            start_token,
            get_id("assistant"),
            get_id("<|channel|>"),
            get_id("analysis"),
            get_id("<|message|>")
        ]
        tokens = header_tokens + tokens

    for token_id in tokens:
        try:
            parser.process(token_id)
        except Exception:
            break
            
    return parser


def parse_chat_output(
    token_ids: Sequence[int],
) -> tuple[str | None, str | None, bool]:
    parser = parse_output_into_messages(token_ids)
    output_msgs = parser.messages
    
    reasoning_parts = []
    final_content = None
    is_tool_call = False
    
    for msg in output_msgs:
        if msg.channel == "analysis":
            for content in msg.content:
                reasoning_parts.append(content.text)
        elif msg.channel == "final":
            for content in msg.content:
                final_content = content.text
        elif msg.channel == "commentary" and msg.recipient and msg.recipient.startswith("functions."):
            is_tool_call = True
            if not final_content:
                final_content = ""
            for content in msg.content:
                 final_content = content.text

    if parser.current_content:
        if parser.current_channel == "analysis":
             reasoning_parts.append(parser.current_content)
        elif parser.current_channel == "final":
             final_content = parser.current_content
        elif parser.current_channel == "commentary" and parser.current_recipient and parser.current_recipient.startswith("functions."):
             is_tool_call = True
             final_content = parser.current_content

    reasoning = "\n".join(reasoning_parts) if reasoning_parts else None

    return reasoning, final_content, is_tool_call