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
from vllm.tokenizers import TokenizerLike

from .params import ChatParams
from .protocol import BaseRenderer

logger = init_logger(__name__)


class TerratorchRenderer(BaseRenderer):
    @classmethod
    def from_config(
        cls,
        config: "ModelConfig",
        tokenizer_kwargs: dict[str, Any],
    ) -> "BaseRenderer":
        return cls(config)

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        if not config.skip_tokenizer_init:
            raise ValueError("Terratorch renderer requires `skip_tokenizer_init=True`")

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return None

    def get_tokenizer(self) -> TokenizerLike:
        raise ValueError("Tokenizer not available for Terratorch renderer")

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], TextPrompt | TokensPrompt | EmbedsPrompt]:
        model_config = self.config

        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            model_config,
            content_format="string",
        )

        prompt = self.render_completion([1])  # Dummy token IDs
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
        model_config = self.config

        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            model_config,
            content_format="string",
        )

        prompt = self.render_completion([1])  # Dummy token IDs
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt
