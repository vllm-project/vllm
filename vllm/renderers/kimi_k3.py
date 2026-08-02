# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
from .inputs import DecoderOnlyDictPrompt, DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams

# Keep the original image mode (including the alpha channel) for K3 instead of
# flattening images onto a background color. Server-level (--media-io-kwargs)
# and request-level media_io_kwargs still take precedence over this default.
_K3_MEDIA_IO_DEFAULTS: dict[str, dict[str, Any]] = {"image": {"image_mode": None}}
_K3_THINKING_EFFORTS = ("low", "high", "max")

# XTML channel-open stubs that encoding_k3 may append as a generation prefix.
# Engine keeps these ids; API usage.prompt_tokens excludes them when set below.
_K3_CHANNEL_OPEN_STUBS = (
    "<|open|>think<|sep|>",
    "<|open|>response<|sep|>",
)


def _merge_k3_media_io_kwargs(
    media_io_kwargs: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    return merge_media_io_kwargs(_K3_MEDIA_IO_DEFAULTS, media_io_kwargs)


def _k3_generation_prefix_len(tokenizer: HfTokenizer, token_ids: list[int]) -> int:
    """Return length of a trailing channel-open stub, else 0."""
    for stub in _K3_CHANNEL_OPEN_STUBS:
        stub_ids = tokenizer.encode(stub, add_special_tokens=False)
        n = len(stub_ids)
        if n and len(token_ids) >= n and token_ids[-n:] == stub_ids:
            return n
    return 0


def _prompt_with_generation_prefix_len(
    tokenizer: HfTokenizer, token_ids: list[int]
) -> DecoderOnlyDictPrompt:
    prompt = parse_dec_only_prompt(token_ids)
    prefix_len = _k3_generation_prefix_len(tokenizer, token_ids)
    if prefix_len:
        prompt["generation_prefix_len"] = prefix_len
    return prompt


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
        kwargs["tokenize"] = True
        return self.get_tokenizer().apply_chat_template(conversation, **kwargs)

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=_merge_k3_media_io_kwargs(params.media_io_kwargs),
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        rendered_conversation = _normalize_k3_tool_messages(conversation)
        token_ids = self._apply_chat_template(rendered_conversation, params)
        prompt = _prompt_with_generation_prefix_len(self.get_tokenizer(), token_ids)
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
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=_merge_k3_media_io_kwargs(params.media_io_kwargs),
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        rendered_conversation = _normalize_k3_tool_messages(conversation)
        token_ids = await self._apply_chat_template_async(rendered_conversation, params)
        prompt = _prompt_with_generation_prefix_len(self.get_tokenizer(), token_ids)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return cast(list[ConversationMessage], rendered_conversation), prompt
