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
from vllm.logger import init_logger
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.utils.async_utils import make_async

from .base import BaseRenderer
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams

logger = init_logger(__name__)


def _adapt_tool_images_for_mistral(
    messages: list[ChatCompletionMessageParam], tokenizer_version: int
) -> list[ChatCompletionMessageParam]:
    """Keep tool results consecutive for Mistral tokenizers before v15.

    Mistral Common accepts multimodal tool content starting with tokenizer v15.
    Older tokenizers reject image chunks on `role=tool`. Move those images to
    the following user turn only at the renderer boundary, after every
    consecutive tool result has been emitted. This avoids reintroducing a user
    message between parallel tool results while preserving the image input.
    """
    if tokenizer_version >= 15:
        return messages

    adapted: list[ChatCompletionMessageParam] = []
    pending_images: list[dict[str, Any]] = []

    for message in messages:
        content = message.get("content")
        if message.get("role") == "tool":
            if isinstance(content, list):
                tool_content = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        pending_images.append(part)
                    else:
                        tool_content.append(part)
                adapted.append(
                    cast(
                        ChatCompletionMessageParam,
                        {**message, "content": tool_content},
                    )
                )
            else:
                adapted.append(message)
            continue

        if pending_images:
            if message.get("role") == "user":
                if isinstance(content, list):
                    user_content = [*pending_images, *content]
                elif isinstance(content, str):
                    user_content = [
                        *pending_images,
                        {"type": "text", "text": content},
                    ]
                else:
                    user_content = pending_images.copy()
                adapted.append(
                    cast(
                        ChatCompletionMessageParam,
                        {**message, "content": user_content},
                    )
                )
            else:
                adapted.append(
                    cast(
                        ChatCompletionMessageParam,
                        {"role": "user", "content": pending_images.copy()},
                    )
                )
                adapted.append(message)
            pending_images.clear()
        else:
            adapted.append(message)

    if pending_images:
        adapted.append(
            cast(
                ChatCompletionMessageParam,
                {"role": "user", "content": pending_images},
            )
        )

    return adapted


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
        messages = _adapt_tool_images_for_mistral(messages, tokenizer.version)
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        prompt_raw = safe_apply_chat_template(
            tokenizer,
            messages,
            **params.get_apply_chat_template_kwargs(),
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
        messages = _adapt_tool_images_for_mistral(messages, tokenizer.version)
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        prompt_raw = await self._apply_chat_template_async(
            tokenizer,
            messages,
            **params.get_apply_chat_template_kwargs(),
        )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt
