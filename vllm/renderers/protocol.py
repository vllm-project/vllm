# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any, Protocol

import torch

from vllm.inputs import EmbedsPrompt, TextPrompt, TokensPrompt
from vllm.tokenizers import TokenizerLike
from vllm.utils.async_utils import AsyncMicrobatchTokenizer
from vllm.utils.collection_utils import is_list_of

from .embed_utils import safe_load_prompt_embeds
from .params import TokenizeParams

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.entrypoints.chat_utils import (
        ChatCompletionMessageParam,
        ConversationMessage,
    )


class RendererLike(Protocol):
    config: "ModelConfig"
    _async_tokenizer: AsyncMicrobatchTokenizer

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

    def get_async_tokenizer(self) -> AsyncMicrobatchTokenizer:
        # Lazy initialization since offline LLM doesn't use async
        if not hasattr(self, "_async_tokenizer"):
            self._async_tokenizer = AsyncMicrobatchTokenizer(self.get_tokenizer())

        return self._async_tokenizer

    # Step 1: Convert raw inputs to prompts
    def render_completion(
        self,
        prompt_raw: str | list[int] | torch.Tensor,
    ) -> TextPrompt | TokensPrompt | EmbedsPrompt:
        if isinstance(prompt_raw, str):
            encoder_config = self.config.encoder_config or {}
            if encoder_config.get("do_lower_case", False):
                prompt_raw = prompt_raw.lower()

            return TextPrompt(prompt=prompt_raw)
        if isinstance(prompt_raw, list):
            return TokensPrompt(prompt_token_ids=prompt_raw)

        embeds = safe_load_prompt_embeds(self.config, prompt_raw)
        return EmbedsPrompt(prompt_embeds=embeds)

    def render_completions(
        self,
        prompt_input: str | list[str] | list[int] | list[list[int]] | None = None,
        prompt_embeds: bytes | list[bytes] | None = None,
    ) -> list[TextPrompt | TokensPrompt | EmbedsPrompt]:
        prompts_raw = list[str | list[int] | torch.Tensor]()

        if prompt_input is not None:
            if isinstance(prompt_input, str) or is_list_of(prompt_input, int):
                prompts_raw.append(prompt_input)
            else:
                prompts_raw.extend(prompt_input)

        if prompt_embeds is not None:
            if isinstance(prompt_embeds, torch.Tensor):
                prompt_embeds.append(prompt_embeds)
            else:
                prompts_raw.extend(prompt_embeds)

        return [self.render_completion(prompt) for prompt in prompts_raw]

    async def render_completions_async(
        self,
        prompt_input: str | list[str] | list[int] | list[list[int]] | None = None,
        prompt_embeds: bytes | list[bytes] | None = None,
    ) -> list[TextPrompt | TokensPrompt | EmbedsPrompt]:
        return self.render_completions(prompt_input, prompt_embeds)

    def render_messages(
        self,
        messages: list["ChatCompletionMessageParam"],
        **kwargs,
    ) -> tuple[list["ConversationMessage"], TextPrompt | TokensPrompt | EmbedsPrompt]:
        raise NotImplementedError

    async def render_messages_async(
        self,
        messages: list["ChatCompletionMessageParam"],
        **kwargs,
    ) -> tuple[list["ConversationMessage"], TextPrompt | TokensPrompt | EmbedsPrompt]:
        return self.render_messages(messages, **kwargs)

    # Step 2: Tokenize prompts if necessary
    def tokenize_prompt(
        self,
        prompt: TextPrompt | TokensPrompt | EmbedsPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt | EmbedsPrompt:
        if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
            prompt = params.apply_pre_tokenization(prompt)

            tokenizer = self.get_tokenizer()
            prompt_token_ids = tokenizer.encode(
                prompt["prompt"],
                truncation=params.truncate_prompt_tokens is not None,
                max_length=params.truncate_prompt_tokens,
                add_special_tokens=params.add_special_tokens,
            )

            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids, **prompt)

        if params.needs_detokenization and "prompt" not in prompt:
            if "prompt_token_ids" not in prompt:
                raise RuntimeError("Cannot run detokenization on embeddings")

            tokenizer = self.get_tokenizer()
            prompt_text = tokenizer.decode(prompt["prompt_token_ids"])
            prompt["prompt"] = prompt_text

        return params.apply_post_tokenization(prompt)

    async def tokenize_prompt_async(
        self,
        prompt: TextPrompt | TokensPrompt | EmbedsPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt | EmbedsPrompt:
        if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
            prompt = params.apply_pre_tokenization(prompt)

            tokenizer = self.get_async_tokenizer()
            prompt_token_ids = await tokenizer.encode(
                prompt["prompt"],
                truncation=params.truncate_prompt_tokens is not None,
                max_length=params.truncate_prompt_tokens,
                add_special_tokens=params.add_special_tokens,
            )

            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids, **prompt)

        if params.needs_detokenization and "prompt" not in prompt:
            if "prompt_token_ids" not in prompt:
                raise RuntimeError("Cannot run detokenization on embeddings")

            tokenizer = self.get_async_tokenizer()
            prompt_text = await tokenizer.decode(prompt["prompt_token_ids"])
            prompt["prompt"] = prompt_text

        return params.apply_post_tokenization(prompt)
