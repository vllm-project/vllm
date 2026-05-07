# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
# fmt: off

"""
DeepSeek-V4 Encoding

A self-contained implementation for encoding/decoding DeepSeek-V4 chat messages
with tool calling, thinking mode, and quick instruction task support.
"""

from typing import Any, Dict, List, Union, Optional, Tuple
import copy
import json

import regex as re

# ============================================================
# Special Tokens
# ============================================================

bos_token: str = "<｜begin▁of▁sentence｜>"
eos_token: str = "<｜end▁of▁sentence｜>"
thinking_start_token: str = "<think>"
thinking_end_token: str = "</think>"
dsml_token: str = "｜DSML｜"

USER_SP_TOKEN = "<｜User｜>"
ASSISTANT_SP_TOKEN = "<｜Assistant｜>"
LATEST_REMINDER_SP_TOKEN = "<｜latest_reminder｜>"

# Task special tokens for internal classification tasks
DS_TASK_SP_TOKENS = {
    "action": "<｜action｜>",
    "query": "<｜query｜>",
    "authority": "<｜authority｜>",
    "domain": "<｜domain｜>",
    "title": "<｜title｜>",
    "read_url": "<｜read_url｜>",
}
VALID_TASKS = set(DS_TASK_SP_TOKENS.keys())

# ============================================================
# Templates
# ============================================================

system_msg_template: str = "{content}"
user_msg_template: str = "{content}"
latest_reminder_msg_template: str = "{content}"
assistant_msg_template: str = "{reasoning}{content}{tool_calls}" + eos_token
assistant_msg_wo_eos_template: str = "{reasoning}{content}{tool_calls}"
thinking_template: str = "{reasoning}"

response_format_template: str = (
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}"
)
tool_call_template: str = (
    "<{dsml_token}invoke name=\"{name}\">\n{arguments}\n</{dsml_token}invoke>"
)
tool_calls_template = (
    "<{dsml_token}{tc_block_name}>\n{tool_calls}\n</{dsml_token}{tc_block_name}>"
)
tool_calls_block_name: str = "tool_calls"

tool_output_template: str = (
    "<tool_result>{content}</tool_result>"
)

REASONING_EFFORT_MAX = (
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
)

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}tool_calls>" block like the following:

<{dsml_token}tool_calls>
<{dsml_token}invoke name="$TOOL_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$TOOL_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""

# ============================================================
# Utility Functions
# ============================================================

def to_json(value: Any) -> str:
    """Serialize a value to JSON string."""
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return json.dumps(value, ensure_ascii=True)


def tools_from_openai_format(tools):
    """Extract function definitions from OpenAI-format tool list."""
    return [tool["function"] for tool in tools]


def tool_calls_from_openai_format(tool_calls):
    """Convert OpenAI-format tool calls to internal format."""
    return [
        {
            "name": tool_call["function"]["name"],
            "arguments": tool_call["function"]["arguments"],
        }
        for tool_call in tool_calls
    ]


def tool_calls_to_openai_format(tool_calls):
    """Convert internal tool calls to OpenAI format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool_call["name"],
                "arguments": tool_call["arguments"],
            }
        }
        for tool_call in tool_calls
    ]


def encode_arguments_to_dsml(tool_call: Dict[str, Any]) -> str:
    """
    Encode tool call arguments into DSML parameter format.

    Args:
        tool_call: Dict with "name" and "arguments" keys.

    Returns:
        DSML-formatted parameter string.
    """
    p_dsml_template = '<{dsml_token}parameter name="{key}" string="{is_str}">{value}</{dsml_token}parameter>'
    P_dsml_strs = []

    if isinstance(tool_call["arguments"], str):
        arguments = json.loads(tool_call["arguments"])
    else:
        arguments = tool_call["arguments"]

    for k, v in arguments.items():
        p_dsml_str = p_dsml_template.format(
            dsml_token=dsml_token,
            key=k,
            is_str="true" if isinstance(v, str) else "false",
            value=v if isinstance(v, str) else to_json(v),
        )
        P_dsml_strs.append(p_dsml_str)

    return "\n".join(P_dsml_strs)


def decode_dsml_to_arguments(tool_name: str, tool_args: Dict[str, Tuple[str, str]]) -> Dict[str, str]:
    """
    Decode DSML parameters back to a tool call dict.

    Args:
        tool_name: Name of the tool.
        tool_args: Dict mapping param_name -> (value, is_string_flag).

    Returns:
        Dict with "name" and "arguments" (JSON string) keys.
    """
    def _decode_value(key: str, value: str, string: str):
        if string == "true":
            value = to_json(value)
        return f"{to_json(key)}: {value}"

    tool_args_json = "{" + ", ".join([_decode_value(k, v, string=is_str) for k, (v, is_str) in tool_args.items()]) + "}"
    return dict(name=tool_name, arguments=tool_args_json)


def render_tools(tools: List[Dict[str, Union[str, Dict[str, Any]]]]) -> str:
    """
    Render tool schemas into the system prompt format.

    Args:
        tools: List of tool schema dicts (each with name, description, parameters).

    Returns:
        Formatted tools section string.
    """
    tools_json = [to_json(t) for t in tools]

    return TOOLS_TEMPLATE.format(
        tool_schemas="\n".join(tools_json),
        dsml_token=dsml_token,
        thinking_start_token=thinking_start_token,
        thinking_end_token=thinking_end_token,
    )


def find_last_user_index(messages: List[Dict[str, Any]]) -> int:
    """Find the index of the last user/developer message."""
    last_user_index = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") in ["user", "developer"]:
            last_user_index = idx
            break
    return last_user_index


# ============================================================
# Message Rendering
# ============================================================

def render_message(index: int, messages: List[Dict[str, Any]], thinking_mode: str, drop_thinking: bool = True, reasoning_effort: Optional[str] = None) -> str:
    """
    Render a single message at the given index into its encoded string form.

    This is the core function that converts each message in the conversation
    into the DeepSeek-V4 format.

    Args:
        index: Index of the message to render.
        messages: Full list of messages in the conversation.
        thinking_mode: Either "chat" or "thinking".
        drop_thinking: Whether to drop reasoning content from earlier turns.
        reasoning_effort: Optional reasoning effort level ("max", "high", or None).

    Returns:
        Encoded string for this message.
    """
    assert 0 <= index < len(messages)
    assert thinking_mode in ["chat", "thinking"], f"Invalid thinking_mode `{thinking_mode}`"

    prompt = ""
    msg = messages[index]
    last_user_idx = find_last_user_index(messages)

    role = msg.get("role")
    content = msg.get("content")
    tools = msg.get("tools")
    response_format = msg.get("response_format")
    tool_calls = msg.get("tool_calls")
    reasoning = msg.get("reasoning")
    wo_eos = msg.get("wo_eos", False)

    if tools:
        tools = tools_from_openai_format(tools)
    if tool_calls:
        tool_calls = tool_calls_from_openai_format(tool_calls)

    # Reasoning effort prefix (only at index 0 in thinking mode with max effort)
    assert reasoning_effort in ['max', None, 'high'], f"Invalid reasoning effort: {reasoning_effort}"
    if index == 0 and thinking_mode == "thinking" and reasoning_effort == 'max':
        prompt += REASONING_EFFORT_MAX

    if role == "system":
        prompt += system_msg_template.format(content=content or "")
        if tools:
            prompt += "\n\n" + render_tools(tools)
        if response_format:
            prompt += "\n\n" + response_format_template.format(schema=to_json(response_format))

    elif role == "developer":
        assert content, f"Invalid message for role `{role}`: {msg}"

        content_developer = USER_SP_TOKEN
        content_developer += content

        if tools:
            content_developer += "\n\n" + render_tools(tools)
        if response_format:
            content_developer += "\n\n" + response_format_template.format(schema=to_json(response_format))

        prompt += user_msg_template.format(content=content_developer)

    elif role == "user":
        prompt += USER_SP_TOKEN

        # Handle content blocks (tool results mixed with text)
        content_blocks = msg.get("content_blocks")
        if content_blocks:
            parts = []
            for block in content_blocks:
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(block.get("text", ""))
                elif block_type == "tool_result":
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, list):
                        text_parts = []
                        for b in tool_content:
                            if b.get("type") == "text":
                                text_parts.append(b.get("text", ""))
                            else:
                                text_parts.append(f"[Unsupported {b.get('type')}]")
                        tool_content = "\n\n".join(text_parts)
                    parts.append(tool_output_template.format(content=tool_content))
                else:
                    parts.append(f"[Unsupported {block_type}]")
            prompt += "\n\n".join(parts)
        else:
            prompt += content or ""

    elif role == "latest_reminder":
        prompt += LATEST_REMINDER_SP_TOKEN + latest_reminder_msg_template.format(content=content)

    elif role == "tool":
        raise NotImplementedError("deepseek_v4 merges tool messages into user; please preprocess with merge_tool_messages()")

    elif role == "assistant":
        thinking_part = ""
        tc_content = ""

        if tool_calls:
            tc_list = [
                tool_call_template.format(
                    dsml_token=dsml_token,
                    name=tc.get("name"),
                    arguments=encode_arguments_to_dsml(tc)
                )
                for tc in tool_calls
            ]
            tc_content += '\n\n' + tool_calls_template.format(
                dsml_token=dsml_token,
                tool_calls="\n".join(tc_list),
                tc_block_name=tool_calls_block_name,
            )

        summary_content = content or ""
        reasoning = reasoning or ""

        # Check if previous message has a task - if so, this is a task output (no thinking)
        prev_has_task = index - 1 >= 0 and messages[index - 1].get("task") is not None

        if thinking_mode == "thinking" and not prev_has_task:
            if not drop_thinking or index > last_user_idx:
                thinking_part = thinking_template.format(reasoning=reasoning) + thinking_end_token
            else:
                thinking_part = ""

        if wo_eos:
            prompt += assistant_msg_wo_eos_template.format(
                reasoning=thinking_part,
                content=summary_content,
                tool_calls=tc_content,
            )
        else:
            prompt += assistant_msg_template.format(
                reasoning=thinking_part,
                content=summary_content,
                tool_calls=tc_content,
            )
    else:
        raise NotImplementedError(f"Unknown role: {role}")

    # Append transition tokens based on what follows
    if index + 1 < len(messages) and messages[index + 1].get("role") not in ["assistant", "latest_reminder"]:
        return prompt

    task = messages[index].get("task")
    if task is not None:
        # Task special token for internal classification tasks
        assert task in VALID_TASKS, f"Invalid task: '{task}'. Valid tasks are: {list(VALID_TASKS)}"
        task_sp_token = DS_TASK_SP_TOKENS[task]

        if task != "action":
            # Non-action tasks: append task sp token directly after the message
            prompt += task_sp_token
        else:
            # Action task: append Assistant + thinking token + action sp token
            prompt += ASSISTANT_SP_TOKEN
            prompt += thinking_end_token if thinking_mode != "thinking" else thinking_start_token
            prompt += task_sp_token

    elif messages[index].get("role") in ["user", "developer"]:
        # Normal generation: append Assistant + thinking token
        prompt += ASSISTANT_SP_TOKEN
        if not drop_thinking and thinking_mode == "thinking":
            prompt += thinking_start_token
        elif drop_thinking and thinking_mode == "thinking" and index >= last_user_idx:
            prompt += thinking_start_token
        else:
            prompt += thinking_end_token

    return prompt


# ============================================================
# Preprocessing
# ============================================================

def merge_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge tool messages into the preceding user message using content_blocks format.

    DeepSeek-V4 does not have a standalone "tool" role; instead, tool results
    are encoded as <tool_result> blocks within user messages.

    This function converts a standard OpenAI-format conversation (with separate
    "tool" role messages) into V4 format where tool results are merged into
    user messages.

    Args:
        messages: List of message dicts in OpenAI format.

    Returns:
        Processed message list with tool messages merged into user messages.
    """
    merged: List[Dict[str, Any]] = []

    for msg in messages:
        msg = copy.deepcopy(msg)
        role = msg.get("role")

        if role == "tool":
            # Convert tool message to a user message with tool_result block
            tool_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            # Merge into previous message if it's already a user (merged tool)
            if merged and merged[-1].get("role") == "user" and "content_blocks" in merged[-1]:
                merged[-1]["content_blocks"].append(tool_block)
            else:
                merged.append({
                    "role": "user",
                    "content_blocks": [tool_block],
                })
        elif role == "user":
            text_block = {"type": "text", "text": msg.get("content", "")}
            if merged and merged[-1].get("role") == "user" and "content_blocks" in merged[-1] and merged[-1].get("task") is None:
                merged[-1]["content_blocks"].append(text_block)
            else:
                new_msg = {
                    "role": "user",
                    "content": msg.get("content", ""),
                    "content_blocks": [text_block],
                }
                # Preserve extra fields (task, wo_eos, mask, etc.)
                for key in ("task", "wo_eos", "mask"):
                    if key in msg:
                        new_msg[key] = msg[key]
                merged.append(new_msg)
        else:
            merged.append(msg)

    return merged


def sort_tool_results_by_call_order(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort tool_result blocks within user messages by the order of tool_calls
    in the preceding assistant message.

    Args:
        messages: Preprocessed message list (after merge_tool_messages).

    Returns:
        Message list with sorted tool result blocks.
    """
    last_tool_call_order: Dict[str, int] = {}

    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            last_tool_call_order = {}
            for idx, tc in enumerate(msg["tool_calls"]):
                tc_id = tc.get("id") or tc.get("function", {}).get("id", "")
                if tc_id:
                    last_tool_call_order[tc_id] = idx

        elif role == "user" and msg.get("content_blocks"):
            tool_blocks = [b for b in msg["content_blocks"] if b.get("type") == "tool_result"]
            if len(tool_blocks) > 1 and last_tool_call_order:
                sorted_blocks = sorted(
                    tool_blocks,
                    key=lambda b: last_tool_call_order.get(b.get("tool_use_id", ""), 0)
                )
                sorted_idx = 0
                new_blocks = []
                for block in msg["content_blocks"]:
                    if block.get("type") == "tool_result":
                        new_blocks.append(sorted_blocks[sorted_idx])
                        sorted_idx += 1
                    else:
                        new_blocks.append(block)
                msg["content_blocks"] = new_blocks

    return messages


# ============================================================
# Main Encoding Function
# ============================================================

def encode_messages(
    messages: List[Dict[str, Any]],
    thinking_mode: str,
    context: Optional[List[Dict[str, Any]]] = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: Optional[str] = None,
) -> str:
    """
    Encode a list of messages into the DeepSeek-V4 prompt format.

    This is the main entry point for encoding conversations. It handles:
    - BOS token insertion
    - Thinking mode with optional reasoning content dropping
    - Tool message merging into user messages
    - Multi-turn conversation context

    Args:
        messages: List of message dicts to encode.
        thinking_mode: Either "chat" or "thinking".
        context: Optional preceding context messages (already encoded prefix).
        drop_thinking: If True, drop reasoning from earlier assistant turns
                      (only keep reasoning for messages after the last user message).
        add_default_bos_token: Whether to prepend BOS token at conversation start.
        reasoning_effort: Optional reasoning effort level ("max", "high", or None).

    Returns:
        The encoded prompt string.
    """
    context = context if context else []

    # Preprocess: merge tool messages and sort tool results
    messages = merge_tool_messages(messages)
    messages = sort_tool_results_by_call_order(context + messages)[len(context):]
    if context:
        context = merge_tool_messages(context)
        context = sort_tool_results_by_call_order(context)

    full_messages = context + messages

    prompt = bos_token if add_default_bos_token and len(context) == 0 else ""

    # Resolve drop_thinking: if any message has tools defined, don't drop thinking
    effective_drop_thinking = drop_thinking
    if any(m.get("tools") for m in full_messages):
        effective_drop_thinking = False

    if thinking_mode == "thinking" and effective_drop_thinking:
        full_messages = _drop_thinking_messages(full_messages)
        # After dropping, recalculate how many messages to render
        # (context may have shrunk too)
        num_to_render = len(full_messages) - len(_drop_thinking_messages(context))
        context_len = len(full_messages) - num_to_render
    else:
        num_to_render = len(messages)
        context_len = len(context)

    for idx in range(num_to_render):
        prompt += render_message(
            idx + context_len,
            full_messages,
            thinking_mode=thinking_mode,
            drop_thinking=effective_drop_thinking,
            reasoning_effort=reasoning_effort,
        )

    return prompt


def _drop_thinking_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Drop reasoning and non-essential messages before the last user message.

    Behavior:
    - Messages with role in ["user", "system", "tool", "latest_reminder"] are always kept.
    - Messages at or after the last user index are always kept.
    - Assistant messages before the last user get reasoning removed.
    - Developer messages before the last user are dropped entirely.
    """
    last_user_idx = find_last_user_index(messages)
    result = []
    keep_roles = {"user", "system", "tool", "latest_reminder", "direct_search_results"}

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role in keep_roles or idx >= last_user_idx:
            result.append(msg)
        elif role == "assistant":
            msg = copy.copy(msg)
            msg.pop("reasoning", None)
            result.append(msg)
        # developer and other roles before last_user_idx are dropped

    return result


# ============================================================
# Parsing (Decoding model output)
# ============================================================

def _read_until_stop(index: int, text: str, stop: List[str]) -> Tuple[int, str, Optional[str]]:
    """
    Read text from index until one of the stop strings is found.

    Returns:
        Tuple of (new_index, content_before_stop, matched_stop_string_or_None).
    """
    min_pos = len(text)
    matched_stop = None

    for s in stop:
        pos = text.find(s, index)
        if pos != -1 and pos < min_pos:
            min_pos = pos
            matched_stop = s

    if matched_stop:
        content = text[index:min_pos]
        return min_pos + len(matched_stop), content, matched_stop
    else:
        content = text[index:]
        return len(text), content, None


def parse_tool_calls(index: int, text: str) -> Tuple[int, Optional[str], List[Dict[str, str]]]:
    """
    Parse DSML tool calls from text starting at the given index.

    Args:
        index: Starting position in text.
        text: The full text to parse.

    Returns:
        Tuple of (new_index, last_stop_token, list_of_tool_call_dicts).
        Each tool call dict has "name" and "arguments" keys.
    """
    tool_calls: List[Dict[str, Any]] = []
    stop_token = None
    tool_calls_end_token = f"</{dsml_token}{tool_calls_block_name}>"

    while index < len(text):
        index, content_before, stop_token = _read_until_stop(index, text, [f"<{dsml_token}invoke", tool_calls_end_token])
        if content_before != ">\n":
            raise ValueError(f"Tool call format error: expected '>\\n' but got '{content_before}'")

        if stop_token == tool_calls_end_token:
            break

        if stop_token is None:
            raise ValueError("Missing special token in tool calls")

        index, tool_name_content, stop_token = _read_until_stop(index, text, [f"<{dsml_token}parameter", f"</{dsml_token}invoke"])

        p_tool_name = re.findall(r'^\s*name="(.*?)">\n$', tool_name_content, flags=re.DOTALL)
        if len(p_tool_name) != 1:
            raise ValueError(f"Tool name format error: '{tool_name_content}'")
        tool_name = p_tool_name[0]

        tool_args: Dict[str, Tuple[str, str]] = {}
        while stop_token == f"<{dsml_token}parameter":
            index, param_content, stop_token = _read_until_stop(index, text, [f"/{dsml_token}parameter"])

            param_kv = re.findall(r'^ name="(.*?)" string="(true|false)">(.*?)<$', param_content, flags=re.DOTALL)
            if len(param_kv) != 1:
                raise ValueError(f"Parameter format error: '{param_content}'")
            param_name, string, param_value = param_kv[0]

            if param_name in tool_args:
                raise ValueError(f"Duplicate parameter name: '{param_name}'")
            tool_args[param_name] = (param_value, string)

            index, content, stop_token = _read_until_stop(index, text, [f"<{dsml_token}parameter", f"</{dsml_token}invoke"])
            if content != ">\n":
                raise ValueError(f"Parameter format error: expected '>\\n' but got '{content}'")

        tool_call = decode_dsml_to_arguments(tool_name=tool_name, tool_args=tool_args)
        tool_calls.append(tool_call)

    return index, stop_token, tool_calls


def parse_message_from_completion_text(text: str, thinking_mode: str) -> Dict[str, Any]:
    """
    Parse a model completion text into a structured assistant message.

    This function takes the raw text output from the model (a single assistant turn)
    and extracts:
    - reasoning (thinking block)
    - content (summary/response)
    - tool_calls (if any)

    NOTE: This function is designed to parse only correctly formatted strings and
    will raise ValueError for malformed output.

    Args:
        text: The raw completion text (including EOS token).
        thinking_mode: Either "chat" or "thinking".

    Returns:
        Dict with keys: "role", "content", "reasoning", "tool_calls".
        tool_calls are in OpenAI format.
    """
    summary_content, reasoning = "", ""
    tool_calls: List[Dict[str, str]] = []
    index, stop_token = 0, None
    tool_calls_start_token = f"\n\n<{dsml_token}{tool_calls_block_name}"

    is_thinking = thinking_mode == "thinking"
    is_tool_calling = False

    if is_thinking:
        index, content_delta, stop_token = _read_until_stop(index, text, [thinking_end_token, tool_calls_start_token])
        reasoning = content_delta
        if stop_token != thinking_end_token:
            raise ValueError("Invalid thinking format: missing </think>")

    index, content_delta, stop_token = _read_until_stop(index, text, [eos_token, tool_calls_start_token])
    summary_content = content_delta
    if stop_token == tool_calls_start_token:
        is_tool_calling = True
    else:
        if stop_token != eos_token:
            raise ValueError("Invalid format: missing EOS token")

    if is_tool_calling:
        index, stop_token, tool_calls = parse_tool_calls(index, text)

        index, tool_ends_text, stop_token = _read_until_stop(index, text, [eos_token])
        if tool_ends_text:
            raise ValueError("Unexpected content after tool calls")

    if len(text) != index or stop_token not in [eos_token, None]:
        raise ValueError("Unexpected content at end")

    for sp_token in [bos_token, eos_token, thinking_start_token, thinking_end_token, dsml_token]:
        if sp_token in summary_content or sp_token in reasoning:
            raise ValueError(f"Unexpected special token '{sp_token}' in content")

    return {
        "role": "assistant",
        "content": summary_content,
        "reasoning": reasoning,
        "tool_calls": tool_calls_to_openai_format(tool_calls)
    }

# fmt: on
