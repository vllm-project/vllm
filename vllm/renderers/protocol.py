# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from vllm.inputs import EmbedsPrompt, TextPrompt, TokensPrompt
from vllm.tokenizers import TokenizerLike
from vllm.utils.async_utils import AsyncMicrobatchTokenizer

from .embed_utils import safe_load_prompt_embeds
from .inputs import DictPrompt, EncoderDecoderTokPrompt, TokPrompt
from .params import ChatParams, TokenizeParams

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.entrypoints.chat_utils import (
        ChatCompletionMessageParam,
        ConversationMessage,
    )


class BaseRenderer(ABC):
    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: "ModelConfig",
        tokenizer_kwargs: dict[str, Any],
    ) -> "BaseRenderer":
        raise NotImplementedError

    def __init__(self, config: "ModelConfig") -> None:
        super().__init__()

        self.config = config

        # Lazy initialization since offline LLM doesn't use async
        self._async_tokenizer: AsyncMicrobatchTokenizer | None = None

    @property
    @abstractmethod
    def tokenizer(self) -> TokenizerLike | None:
        raise NotImplementedError

    def get_tokenizer(self) -> TokenizerLike:
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer not available when `skip_tokenizer_init=True`")

        return tokenizer

    def get_async_tokenizer(self) -> AsyncMicrobatchTokenizer:
        if self._async_tokenizer is None:
            self._async_tokenizer = AsyncMicrobatchTokenizer(self.get_tokenizer())

        return self._async_tokenizer

    # Step 1: Convert raw inputs to prompts
    def render_completion(
        self,
        prompt: DictPrompt | bytes,
    ) -> DictPrompt:
        if isinstance(prompt, bytes):
            embeds = safe_load_prompt_embeds(self.config, prompt)
            prompt = EmbedsPrompt(prompt_embeds=embeds)

        return prompt

    def render_completions(
        self,
        prompts: Sequence[DictPrompt | bytes],
    ) -> list[DictPrompt]:
        if len(prompts) == 0:
            raise ValueError("You must pass at least one prompt")

        return [self.render_completion(prompt) for prompt in prompts]

    async def render_completions_async(
        self,
        prompts: Sequence[DictPrompt | bytes],
    ) -> list[DictPrompt]:
        return self.render_completions(prompts)

    @abstractmethod
    def render_messages(
        self,
        messages: list["ChatCompletionMessageParam"],
        params: ChatParams,
    ) -> tuple[list["ConversationMessage"], DictPrompt]:
        raise NotImplementedError

    async def render_messages_async(
        self,
        messages: list["ChatCompletionMessageParam"],
        params: ChatParams,
    ) -> tuple[list["ConversationMessage"], DictPrompt]:
        return self.render_messages(messages, params)

    # Step 2: Tokenize prompts if necessary
    def _tokenize_prompt(
        self,
        prompt: TextPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt:
        tokenizer = self.get_tokenizer()
        prompt_token_ids = tokenizer.encode(
            prompt["prompt"],
            **params.get_encode_kwargs(),
        )

        return TokensPrompt(prompt_token_ids=prompt_token_ids, **prompt)

    async def _tokenize_prompt_async(
        self,
        prompt: TextPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt:
        tokenizer = self.get_async_tokenizer()
        prompt_token_ids = await tokenizer.encode(
            prompt["prompt"],
            **params.get_encode_kwargs(),
        )

        return TokensPrompt(prompt_token_ids=prompt_token_ids, **prompt)

    def _detokenize_prompt(self, prompt: TokensPrompt) -> TokensPrompt:
        tokenizer = self.get_tokenizer()
        prompt["prompt"] = tokenizer.decode(prompt["prompt_token_ids"])

        return prompt

    async def _detokenize_prompt_async(self, prompt: TokensPrompt) -> TokensPrompt:
        tokenizer = self.get_async_tokenizer()
        prompt["prompt"] = await tokenizer.decode(prompt["prompt_token_ids"])

        return prompt

    def tokenize_prompt(
        self,
        prompt: DictPrompt,
        params: TokenizeParams,
    ) -> TokPrompt:
        if "encoder_prompt" in prompt:
            enc_prompt, dec_prompt = (
                self.tokenize_prompt(prompt["encoder_prompt"], params),
                (
                    None
                    if prompt["decoder_prompt"] is None
                    else self.tokenize_prompt(prompt["decoder_prompt"], params)
                ),
            )

            return EncoderDecoderTokPrompt(
                encoder_prompt=enc_prompt,
                decoder_prompt=dec_prompt,
            )

        if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
            prompt = params.apply_pre_tokenization(self.tokenizer, prompt)
            prompt = self._tokenize_prompt(prompt, params)

        if params.needs_detokenization and "prompt" not in prompt:
            if "prompt_token_ids" not in prompt:
                raise RuntimeError("Cannot run detokenization on embeddings")

            prompt = self._detokenize_prompt(prompt)

        return params.apply_post_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]

    def tokenize_prompts(
        self,
        prompts: Sequence[DictPrompt],
        params: TokenizeParams,
    ) -> list[TokPrompt]:
        return [self.tokenize_prompt(prompt, params) for prompt in prompts]

    async def tokenize_prompt_async(
        self,
        prompt: DictPrompt,
        params: TokenizeParams,
    ) -> TokPrompt:
        if "encoder_prompt" in prompt:
            enc_prompt, dec_prompt = await asyncio.gather(
                self.tokenize_prompt_async(prompt["encoder_prompt"], params),
                (
                    asyncio.sleep(0)
                    if prompt["decoder_prompt"] is None
                    else self.tokenize_prompt_async(prompt["decoder_prompt"], params)
                ),
            )

            return EncoderDecoderTokPrompt(
                encoder_prompt=enc_prompt,
                decoder_prompt=dec_prompt,
            )

        if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
            prompt = params.apply_pre_tokenization(self.tokenizer, prompt)
            prompt = await self._tokenize_prompt_async(prompt, params)

        if params.needs_detokenization and "prompt" not in prompt:
            if "prompt_token_ids" not in prompt:
                raise RuntimeError("Cannot run detokenization on embeddings")

            prompt = await self._detokenize_prompt_async(prompt)

        return params.apply_post_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]

    async def tokenize_prompts_async(
        self,
        prompts: Sequence[DictPrompt],
        params: TokenizeParams,
    ) -> list[TokPrompt]:
        return await asyncio.gather(
            *(self.tokenize_prompt_async(prompt, params) for prompt in prompts)
        )
