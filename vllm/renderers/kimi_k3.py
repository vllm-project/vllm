# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from vllm.config import VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
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


def _merge_k3_media_io_kwargs(
    media_io_kwargs: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    return merge_media_io_kwargs(_K3_MEDIA_IO_DEFAULTS, media_io_kwargs)


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
        conversation: list[ConversationMessage],
        params: ChatParams,
    ) -> list[int]:
        # Tokenize eagerly: K3 encodes structural markers as special tokens and
        # user/tool text as ordinary tokens, so we cannot defer to a plain
        # re-tokenization of the rendered string downstream.
        kwargs = params.get_apply_chat_template_kwargs()
        # Translate the standard enable_thinking/reasoning_effort kwargs to
        # K3's native thinking/thinking_effort (native kwargs take precedence).
        if (enable_thinking := kwargs.pop("enable_thinking", None)) is not None:
            kwargs.setdefault("thinking", enable_thinking)
        if (reasoning_effort := kwargs.pop("reasoning_effort", None)) is not None:
            kwargs.setdefault("thinking_effort", reasoning_effort)
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

        prompt = parse_dec_only_prompt(self._apply_chat_template(conversation, params))
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
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=_merge_k3_media_io_kwargs(params.media_io_kwargs),
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        token_ids = await self._apply_chat_template_async(conversation, params)
        prompt = parse_dec_only_prompt(token_ids)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt
