# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace

from vllm.exceptions import VLLMValidationError
from vllm.renderers.chat_utils import ChatCompletionMessageParam, ConversationMessage

from .hf import HfRenderer
from .inputs import DictPrompt
from .params import ChatParams


def _order_tool_results(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    ordered = []
    i = 0
    while i < len(messages):
        message = messages[i]
        if message.get("role") != "tool":
            tool_calls = message.get("tool_calls")
            if message.get("role") == "assistant" and tool_calls:
                calls = list(tool_calls)
                if any(not isinstance(call, Mapping) for call in calls):
                    raise VLLMValidationError(
                        "Tool calls must be objects.", parameter="messages"
                    )
                message = dict(message, tool_calls=deepcopy(calls))
            ordered.append(message)
            i += 1
            continue

        previous = ordered[-1] if ordered else {}
        previous_calls = (
            previous.get("tool_calls") if previous.get("role") == "assistant" else None
        )
        if not previous_calls:
            raise VLLMValidationError(
                "Tool results require preceding assistant tool calls.",
                parameter="messages",
            )
        ids = [call.get("id") for call in previous_calls]
        if any(not isinstance(call_id, str) or not call_id for call_id in ids):
            raise VLLMValidationError(
                "Tool call IDs must be nonempty strings.", parameter="messages"
            )
        valid_ids = set(ids)
        if len(valid_ids) != len(ids):
            raise VLLMValidationError(
                "Tool call IDs must be unique within an assistant turn.",
                parameter="messages",
            )
        results = {}
        while i < len(messages) and messages[i].get("role") == "tool":
            result = messages[i]
            call_id = result.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in valid_ids:
                raise VLLMValidationError(
                    "Tool result must reference a preceding tool call ID.",
                    parameter="messages",
                )
            if call_id in results:
                raise VLLMValidationError(
                    "Duplicate tool result ID.", parameter="messages"
                )
            results[call_id] = result
            i += 1
        if len(results) != len(ids):
            raise VLLMValidationError(
                "Each preceding tool call requires one result.", parameter="messages"
            )
        ordered.extend(results[call_id] for call_id in ids)
    return ordered


def _exclude_tools_for_none(params: ChatParams) -> ChatParams:
    if params.chat_template is not None or params.tool_choice != "none":
        return params
    return replace(
        params, chat_template_kwargs={**params.chat_template_kwargs, "tools": []}
    )


class GigaChat4Renderer(HfRenderer):
    """Normalize tool results for GigaChat4's positional native chat template."""

    def render_messages(
        self, messages: list[ChatCompletionMessageParam], params: ChatParams
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        params = _exclude_tools_for_none(params)
        if params.chat_template is None:
            messages = _order_tool_results(messages)
        return super().render_messages(messages, params)

    async def render_messages_async(
        self, messages: list[ChatCompletionMessageParam], params: ChatParams
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        params = _exclude_tools_for_none(params)
        if params.chat_template is None:
            messages = _order_tool_results(messages)
        return await super().render_messages_async(messages, params)
