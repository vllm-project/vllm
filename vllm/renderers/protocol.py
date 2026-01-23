# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any, Protocol

from vllm.inputs import TextPrompt, TokensPrompt
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.entrypoints.chat_utils import (
        ChatCompletionMessageParam,
        ConversationMessage,
    )


class RendererLike(Protocol):
    @classmethod
    def from_config(
        cls,
        config: "ModelConfig",
        tokenizer_kwargs: dict[str, Any],
    ) -> "RendererLike":
        raise NotImplementedError

    @property
    def tokenizer(self) -> TokenizerLike | None:
        raise NotImplementedError

    def get_tokenizer(self) -> TokenizerLike:
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer not available when `skip_tokenizer_init=True`")

        return tokenizer

    def render_messages(
        self,
        messages: list["ChatCompletionMessageParam"],
        **kwargs,
    ) -> tuple[list["ConversationMessage"], TextPrompt | TokensPrompt]:
        raise NotImplementedError

    async def render_messages_async(
        self,
        messages: list["ChatCompletionMessageParam"],
        **kwargs,
    ) -> tuple[list["ConversationMessage"], TextPrompt | TokensPrompt]:
        return self.render_messages(messages, **kwargs)
