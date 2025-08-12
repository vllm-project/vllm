"""
Harmony utilities for GPT-OSS model support.
"""

import datetime
import json
import os
from collections.abc import Iterable, Sequence
from typing import Literal, Optional, Union

try:
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

    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False

from vllm.entrypoints.openai.protocol import ResponseInputOutputItem
from vllm.utils import random_uuid

# Global harmony encoding instance
_harmony_encoding = None

REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH if HARMONY_AVAILABLE else "high",
    "medium": ReasoningEffort.MEDIUM if HARMONY_AVAILABLE else "medium",
    "low": ReasoningEffort.LOW if HARMONY_AVAILABLE else "low",
}


def is_harmony_available() -> bool:
    """Check if openai-harmony is available."""
    return HARMONY_AVAILABLE


def get_encoding(name: str = "o200k_harmony") -> Optional[object]:
    """Get the harmony encoding instance."""
    global _harmony_encoding

    if not HARMONY_AVAILABLE:
        return None

    if _harmony_encoding is None:
        try:
            if HARMONY_AVAILABLE:
                _harmony_encoding = load_harmony_encoding(
                    HarmonyEncodingName.HARMONY_GPT_OSS
                )
            else:
                _harmony_encoding = load_harmony_encoding(name)
        except Exception as e:
            # Handle cases where harmony vocab might not be available
            # in air-gapped environments
            print(f"Warning: Could not load harmony encoding: {e}")
            return None

    return _harmony_encoding


def get_system_message(
    model_identity: Optional[str] = None,
    reasoning_effort: Optional[Literal["high", "medium", "low"]] = None,
    start_date: Optional[str] = None,
    browser_description: Optional[str] = None,
    python_description: Optional[str] = None,
) -> object:
    """Get system message for harmony format."""
    if not HARMONY_AVAILABLE:
        return {"role": "system", "content": "You are a helpful assistant."}

    sys_msg_content = SystemContent.new()
    if model_identity is not None:
        sys_msg_content = sys_msg_content.with_model_identity(model_identity)
    if reasoning_effort is not None:
        sys_msg_content = sys_msg_content.with_reasoning_effort(
            REASONING_EFFORT[reasoning_effort]
        )
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


def get_developer_message(
    instructions: Optional[str] = None, tools: Optional[list] = None
) -> object:
    """Get developer message for harmony format."""
    if not HARMONY_AVAILABLE:
        return {"role": "system", "content": instructions or ""}

    dev_msg_content = DeveloperContent.new()
    if instructions is not None:
        dev_msg_content = dev_msg_content.with_instructions(instructions)
    if tools is not None:
        function_tools = []
        for tool in tools:
            if hasattr(tool, "type") and tool.type in (
                "web_search_preview",
                "code_interpreter",
            ):
                # These are built-in tools that are added to the system message.
                pass
            elif hasattr(tool, "type") and tool.type == "function":
                function_tools.append(tool)
            else:
                # Handle dict-like tools
                if isinstance(tool, dict) and tool.get("type") == "function":
                    function_tools.append(tool)
        if function_tools:
            function_tool_descriptions = []
            for tool in function_tools:
                if hasattr(tool, "name"):
                    function_tool_descriptions.append(
                        ToolDescription.new(
                            name=tool.name,
                            description=tool.description,
                            parameters=tool.parameters,
                        )
                    )
                elif isinstance(tool, dict):
                    function_tool_descriptions.append(
                        ToolDescription.new(
                            name=tool.get("name", "unknown"),
                            description=tool.get("description", ""),
                            parameters=tool.get("parameters", {}),
                        )
                    )
            dev_msg_content = dev_msg_content.with_function_tools(
                function_tool_descriptions
            )
    dev_msg = Message.from_role_and_content(Role.DEVELOPER, dev_msg_content)
    return dev_msg


def get_user_message(content: str) -> object:
    """Get user message for harmony format."""
    if not HARMONY_AVAILABLE:
        return {"role": "user", "content": content}
    return Message.from_role_and_content(Role.USER, content)


def parse_response_input(response_msg: dict, prev_responses: list = None) -> object:
    """Parse response input into harmony format."""
    if not HARMONY_AVAILABLE:
        return {"role": "user", "content": str(response_msg)}

    if prev_responses is None:
        prev_responses = []

    if not isinstance(response_msg, dict):
        response_msg = (
            response_msg.model_dump() if hasattr(response_msg, "model_dump") else {}
        )

    if "type" not in response_msg or response_msg["type"] == "message":
        role = response_msg.get("role", "user")
        content = response_msg.get("content", "")
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
                TextContent(text=text_prefix + c.get("text", "")) for c in content
            ]
            msg = Message.from_role_and_contents(role, contents)
    elif response_msg["type"] == "function_call_output":
        call_id = response_msg["call_id"]
        call_response = None
        for prev_response in reversed(prev_responses):
            if hasattr(prev_response, "call_id") and prev_response.call_id == call_id:
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


def parse_chat_input(chat_msg) -> object:
    """Parse chat input into harmony format."""
    if not HARMONY_AVAILABLE:
        return chat_msg

    role = chat_msg.get("role", "user")
    content = chat_msg.get("content", "")
    if isinstance(content, str):
        contents = [TextContent(text=content)]
    else:
        # TODO: Support refusal.
        contents = [TextContent(text=c.get("text", "")) for c in content]
    msg = Message.from_role_and_contents(role, contents)
    return msg


def render_for_completion(messages: list) -> list[int]:
    """Render messages for completion."""
    if not HARMONY_AVAILABLE:
        return []

    conversation = Conversation.from_messages(messages)
    token_ids = get_encoding().render_conversation_for_completion(
        conversation, Role.ASSISTANT
    )
    return token_ids


def parse_output_message(message) -> list:
    """Parse a Harmony message into a list of output response items."""
    if not HARMONY_AVAILABLE:
        return []

    if message.author.role != "assistant":
        # This is a message from a tool to the assistant (e.g., search result).
        # Don't include it in the final output for now. This aligns with
        # OpenAI's behavior on models like o4-mini.
        return []

    output_items = []
    recipient = message.recipient
    if recipient is not None and recipient.startswith("browser."):
        if len(message.content) != 1:
            raise ValueError("Invalid number of contents in browser message")
        content = message.content[0]
        browser_call = json.loads(content.text)
        # TODO: translate to url properly!
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
        elif message.recipient.startswith("python") or message.recipient.startswith(
            "browser"
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


def parse_remaining_state(parser) -> list:
    """Parse remaining state from parser."""
    if not HARMONY_AVAILABLE:
        return []

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


def get_stop_tokens_for_assistant_actions():
    """Get stop tokens for assistant actions."""
    encoding = get_encoding()
    if encoding is None:
        return []

    try:
        return encoding.stop_tokens_for_assistant_actions()
    except AttributeError:
        # Fallback if method doesn't exist
        return []


def get_streamable_parser_for_assistant():
    """Get streamable parser for assistant role."""
    return StreamableParser(get_encoding(), role=Role.ASSISTANT)


def parse_output_into_messages(token_ids: Iterable[int]):
    """Parse output token ids into messages."""
    if not HARMONY_AVAILABLE:
        return None

    parser = get_streamable_parser_for_assistant()
    print(f"JAY DEBUG: Parsing output tokens: {token_ids}")
    print(f"JAY DEBUG: Parser: {parser}")
    for token_id in token_ids:
        parser.process(token_id)
    return parser


def parse_chat_output(
    token_ids: Sequence[int],
) -> tuple[Optional[str], Optional[str], bool]:
    """Parse chat output tokens into reasoning and final content."""
    parser = parse_output_into_messages(token_ids)
    if parser is None:
        return None, None, False

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
                f"but got {len(output_msgs)}."
            )
        reasoning_msg, final_msg = output_msgs
        reasoning_content = reasoning_msg.content[0].text
        final_content = final_msg.content[0].text
        is_tool_call = final_msg.recipient is not None

    return reasoning_content, final_content, is_tool_call


def encode_reasoning_token(token_type: str = "reasoning"):
    """Encode reasoning tokens for GPT-OSS."""
    encoding = get_encoding()
    if encoding is None:
        return []

    try:
        # This is a placeholder - actual implementation depends on harmony API
        return encoding.encode(f"<|{token_type}|>")
    except Exception:
        return []


def is_reasoning_token(token_id: int) -> bool:
    """Check if a token ID represents a reasoning token."""
    encoding = get_encoding()
    if encoding is None:
        return False

    try:
        # This is a placeholder - actual implementation depends on harmony API
        decoded = encoding.decode([token_id])
        return decoded.startswith("<|") and decoded.endswith("|>")
    except Exception:
        return False
