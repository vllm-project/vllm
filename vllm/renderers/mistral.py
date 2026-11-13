# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from typing import Any

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.renderers.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.utils.async_utils import make_async

from .base import BaseRenderer
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams

logger = init_logger(__name__)


def safe_apply_chat_template(
    tokenizer: MistralTokenizer,
    messages: list[ChatCompletionMessageParam],
    **kwargs,
) -> str | list[int]:
    from mistral_common.exceptions import MistralCommonException

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    # mistral-common uses assert statements to stop processing of input
    # if input does not comply with the expected format.
    # We convert those assertion errors to ValueErrors so they can be
    # properly caught in the preprocessing_input step
    except (AssertionError, MistralCommonException) as e:
        raise ValueError(str(e)) from e

    # External library exceptions can sometimes occur despite the framework's
    # internal exception management capabilities.
    except Exception as e:
        # Log and report any library-related exceptions for further
        # investigation.
        logger.exception(
            "An error occurred in `mistral_common` while applying chat template"
        )
        raise ValueError(str(e)) from e


_VALID_TOOL_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
_INVALID_TOOL_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]")


def _sanitize_tool_call_names(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    """Clean the tool call names that mistral-common rejects.

    Invalid characters are dropped (`invalid_tool_call` if none is left).
    Messages without an invalid name are returned unchanged.
    """
    sanitized: list[ChatCompletionMessageParam] = []
    for message in messages:
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not isinstance(tool_calls, list):
            sanitized.append(message)
            continue
        fixed = []
        changed = False
        for call in tool_calls:
            function = call.get("function") if isinstance(call, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            if isinstance(name, str) and not _VALID_TOOL_NAME.fullmatch(name):
                clean = _INVALID_TOOL_NAME_CHARS.sub("", name)[:64]
                function = {**function, "name": clean or "invalid_tool_call"}
                call = {**call, "function": function}
                changed = True
            fixed.append(call)
        if changed:
            message = {**message, "tool_calls": fixed}  # type: ignore[assignment]
        sanitized.append(message)
    return sanitized


def _get_chat_template_kwargs(params: ChatParams) -> dict[str, Any]:
    kwargs = params.get_apply_chat_template_kwargs()
    if params.response_format is not None:
        rf = params.response_format.model_dump(
            mode="json", by_alias=True, exclude_none=True
        )
        # mistral-common only accepts JSON objects and JSON schemas.
        if rf["type"] == "json_object" or "schema" in rf.get("json_schema", {}):
            kwargs["response_format"] = rf
    return kwargs


class MistralRenderer(BaseRenderer[MistralTokenizer]):
    def __init__(
        self,
        config: VllmConfig,
        tokenizer: MistralTokenizer | None,
    ) -> None:
        super().__init__(config, tokenizer)

        self._apply_chat_template_async = make_async(
            safe_apply_chat_template, executor=self._executor
        )

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        tokenizer = self.get_tokenizer()
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        prompt_raw = safe_apply_chat_template(
            tokenizer,
            _sanitize_tool_call_names(messages),
            **_get_chat_template_kwargs(params),
        )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        tokenizer = self.get_tokenizer()
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        prompt_raw = await self._apply_chat_template_async(
            tokenizer,
            _sanitize_tool_call_names(messages),
            **_get_chat_template_kwargs(params),
        )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt
