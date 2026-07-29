# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026 Moonshot AI
#
# Derived from moonshotai/Kimi-K3 encoding_k3.py at commit c5d1dd4.
# Licensed under the Kimi K3 License; see LICENSE.kimi-k3 in this directory.
"""Kimi K3 XTML chat encoding."""

from __future__ import annotations

import json
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
        if self.image_prompts is not None and self.index != len(self.image_prompts):
            raise ValueError(
                f"image prompt count {len(self.image_prompts)} != "
                f"consumed placeholder count {self.index}"
            )


def _segment(text: Any, *, allow_special: bool = False) -> list[EncodeSegment]:
    text = str(text)
    return [EncodeSegment(text, allow_special)] if text else []


def _control(text: str) -> list[EncodeSegment]:
    return _segment(text, allow_special=True)


def _text(text: Any) -> list[EncodeSegment]:
    return _segment(text)


def _append_text(
    segments: list[EncodeSegment],
    text: Any,
    image_state: _ImagePromptState,
) -> None:
    text = str(text)
    if not text:
        return
    if image_state.image_prompts is None or IMAGE_PLACEHOLDER not in text:
        segments.extend(_text(text))
        return
    parts = text.split(IMAGE_PLACEHOLDER)
    for index, part in enumerate(parts):
        segments.extend(_text(part))
        if index < len(parts) - 1:
            segments.extend(_segment(image_state.next_prompt(), allow_special=True))


def _escape_attr_value(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace('"', "&quot;")


def _open_tag(
    tag: str,
    attrs: list[tuple[str, Any]] | tuple[tuple[str, Any], ...] = (),
) -> list[EncodeSegment]:
    segments = _control(OPEN_TOKEN) + _text(tag)
    for key, value in attrs:
        segments += _text(f' {key}="{_escape_attr_value(value)}"')
    return segments + _control(SEP_TOKEN)


def _close_tag(tag: str) -> list[EncodeSegment]:
    return _control(CLOSE_TOKEN) + _text(tag) + _control(SEP_TOKEN)


def _end_of_msg() -> list[EncodeSegment]:
    return _control(END_OF_MSG_TOKEN)


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _deep_sort(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _deep_sort(value) for key, value in sorted(obj.items())}
    if isinstance(obj, list):
        return [_deep_sort(value) for value in obj]
    return obj


def _normalize_arguments(arguments: Any) -> tuple[dict[str, Any], str | None]:
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


def _normalize_message(message: Any) -> Any:
    if not isinstance(message, dict):
        return message
    normalized = dict(message)
    if (tools := normalized.get("tools")) is not None:
        normalized["tools"] = _deep_sort(tools)
    calls = normalized.get("tool_calls")
    if not calls:
        return normalized
    normalized_calls = []
    for call in calls:
        call = dict(call)
        function = dict(call.get("function", call))
        arguments, json_block = _normalize_arguments(function.get("arguments"))
        function["arguments"] = arguments
        if json_block is not None:
            function["_xtml_json_block"] = json_block
        if "function" in call:
            call["function"] = function
        else:
            call = function
        normalized_calls.append(call)
    normalized["tool_calls"] = normalized_calls
    return normalized


def normalize_xtml_tool_result_messages(messages: list[Any]) -> list[Any]:
    output: list[Any] = []
    call_index: dict[str, tuple[int, str | None]] = {}
    index = 0
    while index < len(messages):
        message = messages[index]
        if isinstance(message, dict) and message.get("role") == "assistant":
            call_index = {}
            for position, call in enumerate(message.get("tool_calls") or [], start=1):
                function = call.get("function", call)
                name = function.get("name")
                call_id = call.get("id")
                if call_id is not None:
                    call_index.setdefault(str(call_id), (position, name))
            output.append(message)
            index += 1
            continue
        if not isinstance(message, dict) or message.get("role") != "tool":
            output.append(message)
            index += 1
            continue
        run = []
        unresolved = False
        while (
            index < len(messages)
            and isinstance(messages[index], dict)
            and messages[index].get("role") == "tool"
        ):
            tool_message = messages[index]
            call_id = tool_message.get("tool_call_id", tool_message.get("id"))
            target = call_index.get(str(call_id)) if call_id is not None else None
            if target is None:
                unresolved = True
                run.append((0, tool_message))
            else:
                position, name = target
                resolved = dict(tool_message)
                if name is not None:
                    resolved["tool"] = name
                    if "name" in resolved:
                        resolved["name"] = name
                run.append((position, resolved))
            index += 1
        if unresolved:
            output.extend(message for _, message in run)
        else:
            ordered = sorted(run, key=lambda item: item[0])
            output.extend(message for _, message in ordered)
    return output


def is_batched_conversation(conversation: Any) -> bool:
    return (
        isinstance(conversation, list)
        and bool(conversation)
        and isinstance(conversation[0], list)
    )


def _render_content(
    content: Any,
    image_state: _ImagePromptState,
) -> list[EncodeSegment]:
    segments: list[EncodeSegment] = []
    if isinstance(content, str):
        _append_text(segments, content, image_state)
    elif content is not None:
        for part in content:
            if part["type"] in ("image", "image_url"):
                segments.extend(_segment(image_state.next_prompt(), allow_special=True))
            else:
                _append_text(segments, part["text"], image_state)
    return segments


def _internal_system_message(kind: str, body: str) -> list[EncodeSegment]:
    return (
        _open_tag("message", [("role", "system"), ("type", kind)])
        + _text(body.strip())
        + _close_tag("message")
        + _end_of_msg()
    )


def _xtml_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    return "array"


def _render_assistant(
    message: dict[str, Any],
    image_state: _ImagePromptState,
    thinking: bool,
) -> list[EncodeSegment]:
    segments: list[EncodeSegment] = []
    if thinking:
        segments += _open_tag("think")
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning is not None and str(reasoning).strip():
            _append_text(segments, reasoning, image_state)
        segments += _close_tag("think")
    segments += _open_tag("response")
    segments += _render_content(message.get("content"), image_state)
    segments += _close_tag("response")
    calls = message.get("tool_calls") or []
    if calls:
        segments += _open_tag("tools")
        for index, call in enumerate(calls, start=1):
            function = call.get("function", call)
            segments += _open_tag(
                "call", [("tool", function["name"]), ("index", index)]
            )
            json_block = function.get("_xtml_json_block")
            if json_block is not None:
                segments += _open_tag("json", [("type", "object")])
                _append_text(segments, json_block, image_state)
                segments += _close_tag("json")
            else:
                for key, value in function.get("arguments", {}).items():
                    segments += _open_tag(
                        "argument", [("key", key), ("type", _xtml_type(value))]
                    )
                    rendered = (
                        value
                        if isinstance(value, str)
                        else json.dumps(value, ensure_ascii=False)
                    )
                    _append_text(segments, rendered, image_state)
                    segments += _close_tag("argument")
            segments += _close_tag("call")
        segments += _close_tag("tools")
    return segments


def _render_tool_declare(tools: Any, *, dynamic: bool = False) -> list[EncodeSegment]:
    if dynamic:
        body = (
            "## New Tools Available\n"
            "The system dynamically extends the toolset via lazy-loading.\n"
            "You have access to all existing and extended tools.\n"
            "Here are the specs for the extended tools.\n\n"
            f"```json\n{_json_compact(tools)}\n```"
        )
    else:
        body = (
            "# Tools\nHere are the available tools, described in JSONSchema.\n\n"
            f"```json\n{_json_compact(tools)}\n```"
        )
    return _internal_system_message("tool-declare", body)


def _response_schema(response_format: Any) -> Any:
    json_schema = _get_value(response_format, "json_schema")
    if json_schema is None:
        return None
    return _get_value(
        json_schema,
        "schema",
        _get_value(json_schema, "json_schema", json_schema),
    )


def build_chat_segments(
    messages: list[Any],
    tools: list[dict] | None = None,
    *,
    add_generation_prompt: bool = True,
    thinking: bool = True,
    image_prompts: list[str] | None = None,
    **kwargs: Any,
) -> list[EncodeSegment]:
    messages = [
        _normalize_message(message)
        for message in normalize_xtml_tool_result_messages(messages)
    ]
    tools = _deep_sort(tools)
    image_state = _ImagePromptState(image_prompts)
    segments: list[EncodeSegment] = []
    if tools:
        segments += _render_tool_declare(tools)

    effort = kwargs.get("thinking_effort")
    if thinking and effort is not None and effort not in _VALID_THINKING_EFFORTS:
        raise ValueError(
            f"Unsupported thinking_effort={effort!r}; "
            f"supported values are {sorted(_VALID_THINKING_EFFORTS)}."
        )
    if thinking and effort:
        segments += _internal_system_message(
            "thinking-effort",
            "`thinking_effort` guides on how much to think in your thinking "
            "channel (not including the response channel), supported values "
            "include `low`, `medium`, `high`, and `max`.\n"
            f"Now the system is invoked with `thinking_effort={effort}`.",
        )

    prior_calls = None
    tool_index = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message["role"]
        if role == "system" and message.get("tools"):
            segments += _render_tool_declare(message["tools"], dynamic=True)
        elif role in ("user", "system"):
            attrs = [("role", role)]
            if message.get("name"):
                attrs.append(("name", message["name"]))
            segments += _open_tag("message", attrs)
            segments += _render_content(message.get("content"), image_state)
            segments += _close_tag("message") + _end_of_msg()
        elif role == "assistant":
            prior_calls = message.get("tool_calls")
            tool_index = 0
            segments += _open_tag("message", [("role", "assistant")])
            segments += _render_assistant(message, image_state, thinking)
            segments += _close_tag("message") + _end_of_msg()
        elif role == "tool":
            tool_index += 1
            tool_name = message.get("tool", message.get("name"))
            if tool_name is None and prior_calls and tool_index <= len(prior_calls):
                tool_name = prior_calls[tool_index - 1].get(
                    "function", prior_calls[tool_index - 1]
                )["name"]
            if tool_name is None:
                raise ValueError("Kimi K3 tool message has no resolvable tool name.")
            segments += _open_tag(
                "message",
                [("role", "tool"), ("tool", tool_name), ("index", tool_index)],
            )
            segments += _render_content(message.get("content"), image_state)
            segments += _close_tag("message") + _end_of_msg()

    tool_choice = kwargs.get("tool_choice")
    if tool_choice in ("required", "none"):
        requirement = (
            "call tools" if tool_choice == "required" else "NOT call any tools"
        )
        segments += _internal_system_message(
            "tool-choice",
            f"The system is invoked with `tool_choice={tool_choice}`.\n"
            f"You MUST {requirement} in the next message.",
        )

    response_format = kwargs.get("response_format")
    response_type = _get_value(response_format, "type", response_format)
    if response_type == "json_object":
        segments += _internal_system_message(
            "response-format",
            "The system is invoked with `response_format=json_object`.\n"
            "Your response must be raw JSON data without markdown code blocks "
            "(```json) or any additional formatting.",
        )
    elif response_type == "json_schema":
        schema = _deep_sort(_response_schema(response_format))
        segments += _internal_system_message(
            "response-format",
            "The system is invoked with `response_format=json_schema`.\n"
            "Your response must be raw JSON data without markdown code blocks "
            "(```json) or any additional formatting.\n"
            "The JSON data must match the following schema:\n"
            f"```json\n{_json_compact(schema)}\n```",
        )

    if add_generation_prompt:
        segments += _open_tag("message", [("role", "assistant")])
        segments += _open_tag("think" if thinking else "response")
    image_state.assert_consumed()
    return segments
