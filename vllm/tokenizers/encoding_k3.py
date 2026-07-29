# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026 Moonshot AI
#
# Source: https://huggingface.co/moonshotai/Kimi-K3/blob/main/encoding_k3.py
# Licensed under the Kimi K3 License;
"""Kimi K3 XTML encoding helpers.

This module keeps chat rendering in Python.
Callers that need token IDs should consume ``EncodeSegment`` objects directly:
structural markers may be encoded as tiktoken special tokens, while user/tool
text and attribute values are encoded as ordinary text.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

OPEN_TOKEN = "<|open|>"
CLOSE_TOKEN = "<|close|>"
SEP_TOKEN = "<|sep|>"
END_OF_MSG_TOKEN = "<|end_of_msg|>"
IMAGE_PLACEHOLDER = "<|kimi_image_placeholder|>"

_VALID_THINKING_EFFORTS = {"low", "high", "max"}


@dataclass(frozen=True)
class EncodeSegment:
    text: str
    allow_special: bool = False


class _ImagePromptState:
    def __init__(self, image_prompts: list[str] | None = None):
        self.image_prompts = image_prompts
        self.index = 0

    def next_prompt(self) -> str:
        if self.image_prompts is None:
            return IMAGE_PLACEHOLDER
        if self.index >= len(self.image_prompts):
            raise ValueError("More image placeholders than image prompts.")
        prompt = self.image_prompts[self.index]
        self.index += 1
        return prompt

    def assert_consumed(self) -> None:
        if self.image_prompts is None:
            return
        if self.index != len(self.image_prompts):
            raise ValueError(
                f"image prompt count {len(self.image_prompts)} != "
                f"consumed placeholder count {self.index}"
            )


def _segment(text: Any, *, allow_special: bool = False) -> list[EncodeSegment]:
    text = str(text)
    if not text:
        return []
    return [EncodeSegment(text, allow_special=allow_special)]


def _control(text: str) -> list[EncodeSegment]:
    return _segment(text, allow_special=True)


def _text(text: Any) -> list[EncodeSegment]:
    return _segment(text, allow_special=False)


def _append_text(
    segments: list[EncodeSegment],
    text: Any,
    image_state: _ImagePromptState,
) -> None:
    text = str(text)
    if text == "":
        return
    if image_state.image_prompts is None or IMAGE_PLACEHOLDER not in text:
        segments.extend(_text(text))
        return

    parts = text.split(IMAGE_PLACEHOLDER)
    for i, part in enumerate(parts):
        segments.extend(_text(part))
        if i < len(parts) - 1:
            segments.extend(_segment(image_state.next_prompt(), allow_special=True))


def _escape_attr_value(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace('"', "&quot;")


def _attr(key: str, value: Any) -> list[EncodeSegment]:
    return (
        _text(f" {key}") + _text('="') + _text(_escape_attr_value(value)) + _text('"')
    )


def _open_tag(tag: str, attrs: Iterable[tuple[str, Any]] = ()) -> list[EncodeSegment]:
    segments: list[EncodeSegment] = []
    segments.extend(_control(OPEN_TOKEN))
    segments.extend(_text(tag))
    for key, value in attrs:
        segments.extend(_attr(key, value))
    segments.extend(_control(SEP_TOKEN))
    return segments


def _close_tag(tag: str) -> list[EncodeSegment]:
    segments: list[EncodeSegment] = []
    segments.extend(_control(CLOSE_TOKEN))
    segments.extend(_text(tag))
    segments.extend(_control(SEP_TOKEN))
    return segments


def _end_of_msg() -> list[EncodeSegment]:
    return _control(END_OF_MSG_TOKEN)


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _xtml_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if value is None:
        return "null"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if isinstance(value, str):
        return "string"
    if _is_mapping(value):
        return "object"
    return "array"


def _xtml_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_response_schema(response_format: Any) -> Any:
    if response_format is None:
        return None

    json_schema = _get_value(response_format, "json_schema")
    if json_schema is None:
        return None

    if isinstance(json_schema, dict):
        return json_schema.get(
            "schema",
            json_schema.get("json_schema", json_schema),
        )

    schema = _get_value(json_schema, "schema")
    if schema is not None:
        return schema

    schema = _get_value(json_schema, "json_schema")
    if schema is not None:
        return schema

    return json_schema


def deep_sort_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: deep_sort_dict(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [deep_sort_dict(item) for item in obj]
    return obj


def normalize_tool_arguments(arguments: Any) -> tuple[dict[str, Any], str | None]:
    if arguments is None:
        return {}, None
    if isinstance(arguments, dict):
        return arguments, None
    if isinstance(arguments, str):
        if not arguments.strip():
            return {}, None
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}, arguments
        if not isinstance(parsed, dict):
            raise ValueError("Kimi K3 tool call arguments must be a JSON object.")
        return parsed, None
    raise TypeError(
        "Kimi K3 tool call arguments must be a dict or a JSON object string."
    )


def normalize_message(message: Any) -> Any:
    if not isinstance(message, dict):
        return message

    normalized = dict(message)

    tools = normalized.get("tools")
    if tools is not None:
        normalized["tools"] = deep_sort_dict(tools)

    tool_calls = normalized.get("tool_calls")
    if not tool_calls:
        return normalized

    normalized_calls = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            normalized_calls.append(tool_call)
            continue

        tc = dict(tool_call)
        function = tc.get("function")
        if isinstance(function, dict):
            fn = dict(function)
            arguments, json_block = normalize_tool_arguments(fn.get("arguments"))
            fn["arguments"] = arguments
            if json_block is None:
                fn.pop("_xtml_json_block", None)
            else:
                fn["_xtml_json_block"] = json_block
            tc["function"] = fn
        else:
            arguments, json_block = normalize_tool_arguments(tc.get("arguments"))
            tc["arguments"] = arguments
            if json_block is None:
                tc.pop("_xtml_json_block", None)
            else:
                tc["_xtml_json_block"] = json_block
        normalized_calls.append(tc)

    normalized["tool_calls"] = normalized_calls
    return normalized


def normalize_conversation(conversation: Any) -> Any:
    if not isinstance(conversation, list):
        return conversation

    def normalize_messages(messages: list[Any]) -> list[Any]:
        return [normalize_message(message) for message in messages]

    if conversation and isinstance(conversation[0], list):
        return [normalize_messages(messages) for messages in conversation]
    return normalize_messages(conversation)


def _tool_call_id_index(tool_calls: Any) -> dict:
    """Map assistant ``tool_calls[].id`` to ``(1-based position, function name)``.

    The position mirrors the chat template's enumeration over ``tool_calls``
    (every entry advances the position, even an id-less one). Duplicate ids keep
    their first occurrence.
    """
    index: dict = {}
    if not isinstance(tool_calls, list):
        return index
    for position, tool_call in enumerate(tool_calls, start=1):
        if not isinstance(tool_call, dict):
            continue
        call_id = tool_call.get("id")
        if call_id is None:
            continue
        key = str(call_id)
        if key in index:
            continue
        function = tool_call.get("function")
        name = (
            function.get("name")
            if isinstance(function, dict)
            else tool_call.get("name")
        )
        index[key] = (position, name)
    return index


def normalize_xtml_tool_result_messages(messages: list[Any]) -> list[Any]:
    """Re-sort K3 XTML tool results into assistant ``tool_calls`` order.

    Serving frameworks generally deliver tool results already in call order. A
    direct Transformers caller, however, may pass OpenAI-style tool messages in any
    order, so each run of consecutive tool messages is matched against the most
    recent preceding assistant ``tool_calls`` by opaque ``tool_call_id`` ==
    ``tool_calls[].id`` (K3 drops the ``func:index`` format requirement) and
    sorted by the matched 1-based position. The matched call is authoritative,
    so each matched message's ``tool`` is set to that call's function name --
    this keeps an explicit (and possibly stale) ``tool``/``name`` from drifting
    out of sync with the reordered position. ``index`` is still derived from the
    rendered position by the chat template. A run that cannot be fully matched is
    left untouched. Re-running is idempotent.

    This function is side-effect free: matched tool messages are shallow-copied
    before their ``tool``/``name`` is rewritten, and every other message is
    appended to the output as-is. The input list and its message objects are
    never mutated.
    """
    if not isinstance(messages, list):
        return messages

    output: list[Any] = []
    current_index: dict = {}
    i = 0
    n = len(messages)

    while i < n:
        message = messages[i]

        if isinstance(message, dict) and message.get("role") == "assistant":
            tool_calls = message.get("tool_calls")
            current_index = _tool_call_id_index(tool_calls) if tool_calls else {}
            output.append(message)
            i += 1
            continue

        if not isinstance(message, dict) or message.get("role") != "tool":
            output.append(message)
            i += 1
            continue

        run: list[tuple] = []  # (position, original_offset, message, name)
        unresolved = False
        offset = 0
        while (
            i < n
            and isinstance(messages[i], dict)
            and messages[i].get("role") == "tool"
        ):
            tool_message = messages[i]
            call_id = tool_message.get("tool_call_id", tool_message.get("id"))
            matched = current_index.get(str(call_id)) if call_id is not None else None
            if matched is None:
                unresolved = True
                run.append((None, offset, tool_message, None))
            else:
                position, name = matched
                run.append((position, offset, tool_message, name))
            offset += 1
            i += 1

        if unresolved:
            output.extend(item[2] for item in run)
        else:
            run.sort(key=lambda item: (item[0], item[1]))
            for _, _, tool_message, name in run:
                if name is None:
                    output.append(tool_message)
                    continue
                # The id-matched call is authoritative: align tool (and any
                # explicit name) so the rendered XTML tool attribute cannot
                # disagree with the reordered position. Copy first so the
                # caller's message object is never mutated.
                resolved = dict(tool_message)
                resolved["tool"] = name
                if "name" in resolved:
                    resolved["name"] = name
                output.append(resolved)

    return output


def is_batched_conversation(conversation: Any) -> bool:
    return (
        isinstance(conversation, list)
        and bool(conversation)
        and isinstance(conversation[0], list)
    )


def _render_content_segments(
    content: Any,
    image_state: _ImagePromptState,
) -> list[EncodeSegment]:
    segments: list[EncodeSegment] = []
    if isinstance(content, str):
        _append_text(segments, content, image_state)
    elif content is not None:
        for part in content:
            if part["type"] in ["image", "image_url"]:
                segments.extend(_segment(image_state.next_prompt(), allow_special=True))
            else:
                _append_text(segments, part["text"], image_state)
    return segments


def _internal_system_message(message_type: str, body: str) -> list[EncodeSegment]:
    segments: list[EncodeSegment] = []
    segments.extend(_open_tag("message", [("role", "system"), ("type", message_type)]))
    segments.extend(_text(body.strip()))
    segments.extend(_close_tag("message"))
    segments.extend(_end_of_msg())
    return segments


def _render_assistant_segments(
    message: dict[str, Any],
    image_state: _ImagePromptState,
    thinking: bool = True,
) -> list[EncodeSegment]:
    segments: list[EncodeSegment] = []
    # The <think> channel is structural: in thinking mode every assistant
    # message carries the open/close tags even when there is no reasoning
    # content to fill in. In non-thinking mode the channel is dropped
    # entirely.
    if thinking:
        reasoning_content = message.get("reasoning_content") or message.get("reasoning")
        segments.extend(_open_tag("think"))
        if reasoning_content is not None and str(reasoning_content).strip():
            _append_text(segments, reasoning_content, image_state)
        segments.extend(_close_tag("think"))

    segments.extend(_open_tag("response"))
    segments.extend(_render_content_segments(message.get("content"), image_state))
    segments.extend(_close_tag("response"))

    tool_calls = message.get("tool_calls")
    if tool_calls:
        segments.extend(_open_tag("tools"))
        for index, tool_call in enumerate(tool_calls, start=1):
            fn = tool_call.get("function", tool_call)
            segments.extend(_open_tag("call", [("tool", fn["name"]), ("index", index)]))
            args = fn.get("arguments", {})
            json_block = fn.get("_xtml_json_block")
            if json_block is not None:
                segments.extend(_open_tag("json", [("type", "object")]))
                _append_text(segments, json_block, image_state)
                segments.extend(_close_tag("json"))
            elif _is_mapping(args):
                for key, value in args.items():
                    segments.extend(
                        _open_tag(
                            "argument",
                            [("key", key), ("type", _xtml_type(value))],
                        )
                    )
                    _append_text(segments, _xtml_value(value), image_state)
                    segments.extend(_close_tag("argument"))
            segments.extend(_close_tag("call"))
        segments.extend(_close_tag("tools"))

    return segments


def _render_tool_declare(tools: Any, *, dynamic: bool = False) -> list[EncodeSegment]:
    if dynamic:
        body = (
            "## New Tools Available\n"
            "The system dynamically extends the toolset via lazy-loading.\n"
            "You have access to all existing and extended tools.\n"
            "Here are the specs for the extended tools.\n\n"
            "```json\n"
            f"{_json_compact(tools)}\n"
            "```"
        )
    else:
        body = (
            "# Tools\n"
            "Here are the available tools, described in JSONSchema.\n\n"
            "```json\n"
            f"{_json_compact(tools)}\n"
            "```"
        )
    segments: list[EncodeSegment] = []
    segments.extend(
        _open_tag("message", [("role", "system"), ("type", "tool-declare")])
    )
    segments.extend(_text(body))
    segments.extend(_close_tag("message"))
    segments.extend(_end_of_msg())
    return segments


def build_chat_segments(
    messages: list[Any],
    tools: list[dict] | None = None,
    *,
    add_generation_prompt: bool = True,
    thinking: bool = True,
    image_prompts: list[str] | None = None,
    **kwargs: Any,
) -> list[EncodeSegment]:
    # Re-sort tool results by tool_call_id at the lowest layer so every caller
    # (processor or direct tokenizer) gets correctly ordered XTML. The helper is
    # side-effect free, so the caller's message objects are left untouched.
    messages = normalize_xtml_tool_result_messages(messages)
    messages = normalize_conversation(messages)
    tools = deep_sort_dict(tools)

    kwargs = dict(kwargs)
    response_format = kwargs.get("response_format")
    if "response_schema" not in kwargs:
        response_schema = extract_response_schema(response_format)
        if response_schema is not None:
            kwargs["response_schema"] = response_schema
    if kwargs.get("response_schema") is not None:
        kwargs["response_schema"] = deep_sort_dict(kwargs["response_schema"])

    image_state = _ImagePromptState(image_prompts)
    segments: list[EncodeSegment] = []

    tool_calls = None
    tool_index = 0

    if tools:
        segments.extend(_render_tool_declare(tools))

    thinking_effort = kwargs.get("thinking_effort")
    if thinking and thinking_effort is not None:
        assert thinking_effort in _VALID_THINKING_EFFORTS, (
            f"Unsupported thinking_effort={thinking_effort!r}; "
            f"supported values are {sorted(_VALID_THINKING_EFFORTS)}."
        )
    if thinking and thinking_effort in _VALID_THINKING_EFFORTS:
        segments.extend(
            _internal_system_message(
                "thinking-effort",
                "`thinking_effort` guides on how much to think in your "
                "thinking channel (not including the response channel), "
                "supported values include `low`, `medium`, `high`, and `max`.\n"
                f"Now the system is invoked with `thinking_effort={thinking_effort}`.",
            )
        )

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue

        role = message["role"]
        if role == "user":
            attrs = [("role", "user")]
            if message.get("name"):
                attrs.append(("name", message["name"]))
            segments.extend(_open_tag("message", attrs))
            segments.extend(
                _render_content_segments(message.get("content"), image_state)
            )
            segments.extend(_close_tag("message"))
            segments.extend(_end_of_msg())
        elif role == "system" and message.get("tools"):
            segments.extend(_render_tool_declare(message["tools"], dynamic=True))
        elif role == "system":
            attrs = [("role", "system")]
            if message.get("name"):
                attrs.append(("name", message["name"]))
            segments.extend(_open_tag("message", attrs))
            segments.extend(
                _render_content_segments(message.get("content"), image_state)
            )
            segments.extend(_close_tag("message"))
            segments.extend(_end_of_msg())
        elif role == "tool":
            tool_index += 1
            tool_name = message.get("tool", message.get("name"))
            if (
                tool_name is None
                and tool_calls is not None
                and tool_index <= len(tool_calls)
            ):
                tc = tool_calls[tool_index - 1]
                fn = tc.get("function", tc)
                tool_name = fn["name"]
            if tool_name is None:
                raise ValueError(
                    "Kimi K3 tool messages need a resolvable tool name: "
                    "carry `tool`/`name`, or match a preceding assistant "
                    "tool_call by order."
                )
            segments.extend(
                _open_tag(
                    "message",
                    [("role", "tool"), ("tool", tool_name), ("index", tool_index)],
                )
            )
            segments.extend(
                _render_content_segments(message.get("content"), image_state)
            )
            segments.extend(_close_tag("message"))
            segments.extend(_end_of_msg())
        elif role == "assistant":
            tool_calls = message.get("tool_calls")
            tool_index = 0
            attrs = [("role", "assistant")]
            if message.get("name"):
                attrs.append(("name", message["name"]))
            segments.extend(_open_tag("message", attrs))
            segments.extend(_render_assistant_segments(message, image_state, thinking))
            segments.extend(_close_tag("message"))
            segments.extend(_end_of_msg())

    tool_choice = kwargs.get("tool_choice")
    if tool_choice == "required":
        segments.extend(
            _internal_system_message(
                "tool-choice",
                "The system is invoked with `tool_choice=required`.\n"
                "You MUST call tools in the next message.",
            )
        )
    elif tool_choice == "none":
        segments.extend(
            _internal_system_message(
                "tool-choice",
                "The system is invoked with `tool_choice=none`.\n"
                "You MUST NOT call any tools in the next message.",
            )
        )

    rf = kwargs.get("response_format")
    rf_type = _get_value(rf, "type", rf) if isinstance(rf, dict) else rf
    if rf_type == "json_object":
        segments.extend(
            _internal_system_message(
                "response-format",
                "The system is invoked with `response_format=json_object`.\n"
                "Your response must be raw JSON data without markdown code "
                "blocks (```json) or any additional formatting.",
            )
        )
    elif rf_type == "json_schema":
        schema = _json_compact(kwargs.get("response_schema"))
        segments.extend(
            _internal_system_message(
                "response-format",
                "The system is invoked with `response_format=json_schema`.\n"
                "Your response must be raw JSON data without markdown code "
                "blocks (```json) or any additional formatting.\n"
                "The JSON data must match the following schema:\n"
                f"```json\n{schema}\n```",
            )
        )

    if add_generation_prompt:
        segments.extend(_open_tag("message", [("role", "assistant")]))
        segments.extend(_open_tag("think" if thinking else "response"))

    image_state.assert_consumed()
    return segments
