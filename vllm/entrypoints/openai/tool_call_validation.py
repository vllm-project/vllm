# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validation shared by the OpenAI and Anthropic tool-calling surfaces."""

from collections.abc import Iterable
from typing import Any

from vllm.exceptions import VLLMValidationError


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _invalid_tool_result(field: str, value: Any) -> VLLMValidationError:
    if not value:
        return VLLMValidationError(
            f"{field} must be provided for a tool result.", parameter=field
        )
    return VLLMValidationError(
        f"{field} '{value}' does not match any prior tool call.", parameter=field
    )


def validate_chat_tool_call_ids(
    messages: Iterable[Any], *, field: str = "tool_call_id"
) -> None:
    """Reject tool results whose call ID is absent or has no prior call."""
    call_ids: set[str] = set()
    for message in messages:
        role = _get(message, "role")
        if role == "assistant":
            for tool_call in _get(message, "tool_calls", ()) or ():
                call_id = _get(tool_call, "id")
                if call_id:
                    call_ids.add(call_id)
        elif role == "tool":
            result_id = _get(message, field)
            if not result_id or result_id not in call_ids:
                raise _invalid_tool_result(field, result_id)


def validate_responses_tool_call_ids(
    *,
    previous_messages: Iterable[Any] = (),
    previous_outputs: Iterable[Any] = (),
    input_items: Iterable[Any] = (),
    field: str = "call_id",
) -> None:
    """Validate Responses tool results against the effective prior history."""
    call_ids: set[str] = set()
    for message in previous_messages:
        if _get(message, "role") != "assistant":
            continue
        for tool_call in _get(message, "tool_calls", ()) or ():
            call_id = _get(tool_call, "id")
            if call_id:
                call_ids.add(call_id)

    def visit(items: Iterable[Any]) -> None:
        for item in items:
            item_type = _get(item, "type")
            role = _get(item, "role")
            if item_type in ("function_call", "custom_tool_call"):
                call_id = _get(item, field)
                if call_id:
                    call_ids.add(call_id)
            elif item_type in ("function_call_output", "custom_tool_call_output"):
                result_id = _get(item, field)
                if not result_id or result_id not in call_ids:
                    raise _invalid_tool_result(field, result_id)
            elif role == "assistant":
                for tool_call in _get(item, "tool_calls", ()) or ():
                    call_id = _get(tool_call, "id")
                    if call_id:
                        call_ids.add(call_id)
            elif role == "tool":
                result_id = _get(item, "tool_call_id")
                if not result_id or result_id not in call_ids:
                    raise _invalid_tool_result("tool_call_id", result_id)

    visit(previous_outputs)
    visit(input_items)
