# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.inputs import EmbedsPrompt, TextPrompt, TokensPrompt
from vllm.logger import init_logger
from vllm.tokenizers import cached_get_tokenizer
from vllm.tokenizers.grok2 import Grok2Tokenizer

from .params import ChatParams
from .protocol import RendererLike

logger = init_logger(__name__)


class Grok2Renderer(RendererLike):
    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        tokenizer_kwargs: dict[str, Any],
    ) -> "RendererLike":
        return cls(config, tokenizer_kwargs)

    def __init__(
        self,
        config: ModelConfig,
        tokenizer_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()

        self.config = config

        if config.skip_tokenizer_init:
            tokenizer = None
        else:
            tokenizer = cached_get_tokenizer(
                tokenizer_cls=Grok2Tokenizer,
                **tokenizer_kwargs,
            )

        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> Grok2Tokenizer | None:
        return self._tokenizer

    def get_tokenizer(self) -> Grok2Tokenizer:
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer not available when `skip_tokenizer_init=True`")

        return tokenizer

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], TextPrompt | TokensPrompt | EmbedsPrompt]:
        tokenizer = self.get_tokenizer()
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.config,
            content_format="string",
        )

        prompt_raw = tokenizer.apply_chat_template(
            conversation=conversation,
            messages=messages,
            **params.get_apply_chat_template_kwargs(),
        )

        prompt = self.render_completion(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], TextPrompt | TokensPrompt | EmbedsPrompt]:
        tokenizer = self.get_tokenizer()
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.config,
            content_format="string",
        )

        prompt_raw = tokenizer.apply_chat_template(
            conversation=conversation,
            messages=messages,
            **params.get_apply_chat_template_kwargs(),
        )

        prompt = self.render_completion(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt
