# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from typing import Any, cast

from vllm.config import VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.media.connector import merge_media_io_kwargs
from vllm.tokenizers.hf import HfTokenizer
from vllm.utils.async_utils import make_async

from .base import BaseRenderer
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams

# Keep the original image mode (including the alpha channel) for K3 instead of
# flattening images onto a background color. Server-level (--media-io-kwargs)
# and request-level media_io_kwargs still take precedence over this default.
_K3_MEDIA_IO_DEFAULTS: dict[str, dict[str, Any]] = {"image": {"image_mode": None}}
_K3_THINKING_EFFORTS = ("low", "high", "max")


def _merge_k3_media_io_kwargs(
    media_io_kwargs: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    return merge_media_io_kwargs(_K3_MEDIA_IO_DEFAULTS, media_io_kwargs)


def _convert_developer_to_system(
    conversation: list[ConversationMessage],
) -> list[ConversationMessage]:
    """Map ``developer`` messages to ``system`` for K3's chat encoding.

    K3's ``encoding_k3`` only recognizes ``system`` and silently drops other
    roles. A system message carrying ``tools`` is rendered as a dynamic tool
    declare with its content ignored, so a developer message with BOTH
    content and tools is split in two (declare first, content second),
    matching the Rust frontend.
    """
    converted: list[ConversationMessage] = []
    for msg in conversation:
        if msg["role"] != "developer":
            converted.append(msg)
            continue
        content = msg.get("content")
        has_content = bool(
            content
            if isinstance(content, str)
            else (content is not None and content != [])
        )
        if has_content and msg.get("tools"):
            converted.append(
                {"role": "system", "tools": msg["tools"]}  # type: ignore[misc]
            )
            converted.append(
                {
                    **{k: v for k, v in msg.items() if k != "tools"},
                    "role": "system",
                }  # type: ignore[misc]
            )
        else:
            converted.append({**msg, "role": "system"})  # type: ignore[misc]
    return converted


def _drop_null_tool_fields(tools: Any) -> Any:
    """Drop keys with ``None`` values at the tool and function level.

    The serving layer's ``model_dump()`` keeps unset optional fields as
    explicit ``null``, which the platform tokenism omits -- the nulls would
    diverge the rendered prompt. Only the two pydantic-model levels are
    touched; schema content under ``parameters`` is left alone.
    """
    if not isinstance(tools, list):
        return tools
    cleaned = []
    for tool in tools:
        if not isinstance(tool, dict):
            cleaned.append(tool)
            continue
        tool = {k: v for k, v in tool.items() if v is not None}
        function = tool.get("function")
        if isinstance(function, dict):
            tool["function"] = {k: v for k, v in function.items() if v is not None}
        cleaned.append(tool)
    return cleaned


def _preserve_malformed_tool_arguments(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    """Wrap unparsable tool-call ``arguments`` as JSON string literals.

    ``parse_chat_messages`` rejects malformed JSON arguments with a 400,
    while K3's encoding tolerates them via a raw-text fallback. Wrapping the
    string as a JSON string literal survives that ``loads`` (round-tripping
    to the original string), so the text reaches K3's encoding byte-exact.
    """
    preserved: list[ChatCompletionMessageParam] = []
    for message in messages:
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not isinstance(tool_calls, list):
            preserved.append(message)
            continue
        new_calls = []
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                function = tool_call.get("function")
                if isinstance(function, dict):
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        if not arguments.strip():
                            # Whitespace-only arguments read as a non-empty
                            # string downstream and fail json.loads; K3 treats
                            # them as empty arguments.
                            tool_call = {
                                **tool_call,
                                "function": {**function, "arguments": "{}"},
                            }
                        else:
                            try:
                                json.loads(arguments)
                            except json.JSONDecodeError:
                                tool_call = {
                                    **tool_call,
                                    "function": {
                                        **function,
                                        "arguments": json.dumps(arguments),
                                    },
                                }
            new_calls.append(tool_call)
        preserved.append({**message, "tool_calls": new_calls})  # type: ignore[typeddict-item]
    return preserved


def _dump_k3_template_value(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json", exclude_none=True)

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()

    return value


def _apply_k3_thinking_kwargs(kwargs: dict[str, Any]) -> None:
    if (enable_thinking := kwargs.pop("enable_thinking", None)) is not None:
        kwargs.setdefault("thinking", enable_thinking)

    reasoning_effort = kwargs.pop("reasoning_effort", None)
    if reasoning_effort == "none":
        kwargs.setdefault("thinking", False)
    elif reasoning_effort is not None:
        kwargs.setdefault("thinking_effort", reasoning_effort)

    thinking_effort = kwargs.get("thinking_effort")
    if thinking_effort is not None and thinking_effort not in _K3_THINKING_EFFORTS:
        supported = ", ".join(_K3_THINKING_EFFORTS)
        raise VLLMValidationError(
            f"Kimi K3 supports thinking_effort values: {supported}",
            parameter="thinking_effort",
            value=thinking_effort,
        )


def _normalize_k3_tool_messages(
    conversation: list[ConversationMessage],
) -> list[dict[str, Any]]:
    """Reorder tool-result messages to match assistant tool_call order.

    Supports matching by ``tool_call_id`` or by the synthetic
    ``"{tool}:{zero_based_index}"`` alias. When any tool message in a
    block cannot be resolved, the whole block is left in its original
    order (graceful fallback).

    Returns a new list; caller-owned message dicts are not mutated.
    """
    normalized: list[dict[str, Any]] = []
    i = 0

    while i < len(conversation):
        message = conversation[i]
        normalized.append(dict(message))
        i += 1

        if message.get("role") != "assistant":
            continue

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            continue

        # Build the lookup table from the assistant's tool_calls.
        targets_by_id: dict[str, tuple[int, str]] = {}
        for position, tool_call in enumerate(tool_calls):
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not isinstance(name, str) or not name:
                continue

            call_target = (position, name)
            aliases = [f"{name}:{position}"]
            if tool_call_id := tool_call.get("id"):
                aliases.insert(0, str(tool_call_id))
            for alias in aliases:
                targets_by_id.setdefault(alias, call_target)

        # Collect consecutive tool messages.
        block_start = i
        while i < len(conversation) and conversation[i].get("role") == "tool":
            i += 1
        if i == block_start:
            continue

        # Try to resolve every tool message in the block.
        resolved: list[tuple[int, int, dict[str, Any]]] = []
        for order, tool_message in enumerate(conversation[block_start:i]):
            tool_call_id = tool_message.get("tool_call_id")
            resolved_target = (
                targets_by_id.get(str(tool_call_id))
                if tool_call_id is not None
                else None
            )
            if resolved_target is None:
                normalized.extend(dict(item) for item in conversation[block_start:i])
                break

            position, name = resolved_target
            enriched = dict(tool_message)
            enriched["tool"] = name
            enriched["index"] = position + 1
            resolved.append((position, order, enriched))
        else:
            resolved.sort(key=lambda item: (item[0], item[1]))
            normalized.extend(item for _, _, item in resolved)

    return normalized


class KimiK3Renderer(BaseRenderer[HfTokenizer]):
    """Render chat prompts with Kimi K3's Python XTML encoding.

    K3 ships no Jinja chat template; its tokenizer renders messages through
    ``encoding_k3`` instead. We tokenize eagerly so the structural markers keep
    their special-token ids while user- and tool-supplied text stays ordinary.
    """

    def __init__(self, config: VllmConfig, tokenizer: HfTokenizer | None) -> None:
        super().__init__(config, tokenizer)

        self._apply_chat_template_async = make_async(
            self._apply_chat_template, executor=self._executor
        )

    def _apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        params: ChatParams,
    ) -> list[int]:
        # Tokenize eagerly: K3 encodes structural markers as special tokens and
        # user/tool text as ordinary tokens, so we cannot defer to a plain
        # re-tokenization of the rendered string downstream.
        kwargs = params.get_apply_chat_template_kwargs()
        _apply_k3_thinking_kwargs(kwargs)
        if params.tool_choice not in (None, "auto"):
            kwargs["tool_choice"] = _dump_k3_template_value(params.tool_choice)
        if params.response_format is not None:
            kwargs["response_format"] = _dump_k3_template_value(params.response_format)
        if (tools := kwargs.get("tools")) is not None:
            kwargs["tools"] = _drop_null_tool_fields(tools)
        kwargs["tokenize"] = True
        return self.get_tokenizer().apply_chat_template(conversation, **kwargs)

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = parse_chat_messages(
            _preserve_malformed_tool_arguments(messages),
            self.model_config,
            content_format="string",
            media_io_kwargs=_merge_k3_media_io_kwargs(params.media_io_kwargs),
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        rendered_conversation = _normalize_k3_tool_messages(
            _convert_developer_to_system(conversation)
        )
        prompt = parse_dec_only_prompt(
            self._apply_chat_template(rendered_conversation, params)
        )
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return cast(list[ConversationMessage], rendered_conversation), prompt

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            _preserve_malformed_tool_arguments(messages),
            self.model_config,
            content_format="string",
            media_io_kwargs=_merge_k3_media_io_kwargs(params.media_io_kwargs),
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        rendered_conversation = _normalize_k3_tool_messages(
            _convert_developer_to_system(conversation)
        )
        token_ids = await self._apply_chat_template_async(rendered_conversation, params)
        prompt = parse_dec_only_prompt(token_ids)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return cast(list[ConversationMessage], rendered_conversation), prompt
