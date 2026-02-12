# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, overload

from vllm.inputs import EmbedsPrompt, TextPrompt, TokensPrompt
from vllm.tokenizers import TokenizerLike
from vllm.utils.async_utils import AsyncMicrobatchTokenizer

from .embed_utils import safe_load_prompt_embeds
from .inputs import (
    DictPrompt,
    EncoderDecoderDictPrompt,
    EncoderDecoderTokPrompt,
    TokPrompt,
)
from .inputs.preprocess import extract_target_prompt
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
    def render_prompt(
        self,
        prompt: DictPrompt | bytes,
    ) -> DictPrompt:
        if isinstance(prompt, bytes):
            embeds = safe_load_prompt_embeds(self.config, prompt)
            prompt = EmbedsPrompt(prompt_embeds=embeds)

        return prompt

    def render_prompts(
        self,
        prompts: Sequence[DictPrompt | bytes],
    ) -> list[DictPrompt]:
        if len(prompts) == 0:
            raise ValueError("You must pass at least one prompt")

        return [self.render_prompt(prompt) for prompt in prompts]

    async def render_prompts_async(
        self,
        prompts: Sequence[DictPrompt | bytes],
    ) -> list[DictPrompt]:
        return self.render_prompts(prompts)

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

    def _tokenize_enc_dec_prompt(
        self,
        prompt: EncoderDecoderDictPrompt,
        params: TokenizeParams,
    ) -> EncoderDecoderTokPrompt:
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

    async def _tokenize_enc_dec_prompt_async(
        self,
        prompt: EncoderDecoderDictPrompt,
        params: TokenizeParams,
    ) -> EncoderDecoderTokPrompt:
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

    @overload
    def tokenize_prompt(
        self,
        prompt: TextPrompt | TokensPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt: ...

    @overload
    def tokenize_prompt(  # type: ignore[misc]
        self,
        prompt: EmbedsPrompt,
        params: TokenizeParams,
    ) -> EmbedsPrompt: ...

    @overload
    def tokenize_prompt(  # type: ignore[misc]
        self,
        prompt: EncoderDecoderDictPrompt,
        params: TokenizeParams,
    ) -> EncoderDecoderTokPrompt: ...

    def tokenize_prompt(
        self,
        prompt: DictPrompt,
        params: TokenizeParams,
    ) -> TokPrompt:
        if "encoder_prompt" in prompt:
            return self._tokenize_enc_dec_prompt(prompt, params)  # type: ignore[arg-type]

        if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
            prompt = params.apply_pre_tokenization(self.tokenizer, prompt)
            prompt = self._tokenize_prompt(prompt, params)

        if params.needs_detokenization and "prompt" not in prompt:
            if "prompt_token_ids" not in prompt:
                raise RuntimeError("Cannot run detokenization on embeddings")

            prompt = self._detokenize_prompt(prompt)  # type: ignore[arg-type]

        return params.apply_post_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]

    def tokenize_prompts(
        self,
        prompts: Sequence[DictPrompt],
        params: TokenizeParams,
    ) -> list[TokPrompt]:
        return [self.tokenize_prompt(prompt, params) for prompt in prompts]

    @overload
    async def tokenize_prompt_async(
        self,
        prompt: TextPrompt | TokensPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt: ...

    @overload
    async def tokenize_prompt_async(  # type: ignore[misc]
        self,
        prompt: EmbedsPrompt,
        params: TokenizeParams,
    ) -> EmbedsPrompt: ...

    @overload
    async def tokenize_prompt_async(  # type: ignore[misc]
        self,
        prompt: EncoderDecoderDictPrompt,
        params: TokenizeParams,
    ) -> EncoderDecoderTokPrompt: ...

    async def tokenize_prompt_async(
        self,
        prompt: DictPrompt,
        params: TokenizeParams,
    ) -> TokPrompt:
        if "encoder_prompt" in prompt:
            return await self._tokenize_enc_dec_prompt_async(prompt, params)  # type: ignore[arg-type]

        if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
            prompt = params.apply_pre_tokenization(self.tokenizer, prompt)
            prompt = await self._tokenize_prompt_async(prompt, params)

        if params.needs_detokenization and "prompt" not in prompt:
            if "prompt_token_ids" not in prompt:
                raise RuntimeError("Cannot run detokenization on embeddings")

            prompt = await self._detokenize_prompt_async(prompt)  # type: ignore[arg-type]

        return params.apply_post_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]

    async def tokenize_prompts_async(
        self,
        prompts: Sequence[DictPrompt],
        params: TokenizeParams,
    ) -> list[TokPrompt]:
        return await asyncio.gather(
            *(self.tokenize_prompt_async(prompt, params) for prompt in prompts)
        )

    # Step 3: Add extra keys to the prompts
    def _apply_prompt_extras(
        self,
        prompts: Sequence[DictPrompt | TokPrompt],
        prompt_extras: dict[str, Any] | None,
    ):
        if not prompt_extras:
            return

        for prompt in prompts:
            target_prompt = extract_target_prompt(self.config, prompt)
            target_prompt.update(prompt_extras)  # type: ignore[arg-type]

    # Top-level methods
    def render_cmpl(
        self,
        prompts: Sequence[DictPrompt | bytes],
        tok_params: TokenizeParams,
        *,
        prompt_extras: dict[str, Any] | None = None,
    ):
        dict_prompts = self.render_prompts(prompts)

        # NOTE: Some MM models have non-default `add_special_tokens`
        # so we handle tokenization in multi-modal processor
        if self.config.is_multimodal_model:
            self._apply_prompt_extras(dict_prompts, prompt_extras)
            return dict_prompts

        tok_prompts = self.tokenize_prompts(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        # TODO: Apply multi-modal processor
        return tok_prompts

    async def render_cmpl_async(
        self,
        prompts: Sequence[DictPrompt | bytes],
        tok_params: TokenizeParams,
        *,
        prompt_extras: dict[str, Any] | None = None,
    ):
        dict_prompts = await self.render_prompts_async(prompts)

        # NOTE: MM data cannot be passed to online Completions API
        # so we don't have the special case that is in the offline version
        tok_prompts = await self.tokenize_prompts_async(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        # TODO: Apply multi-modal processor
        return tok_prompts

    def render_chat(
        self,
        conversations: Sequence[list["ChatCompletionMessageParam"]],
        chat_params: ChatParams,
        tok_params: TokenizeParams,
        *,
        prompt_extras: dict[str, Any] | None = None,
    ):
        rendered = [
            self.render_messages(conversation, chat_params)
            for conversation in conversations
        ]

        out_conversations = list[list["ConversationMessage"]]()
        dict_prompts = list[DictPrompt]()
        for conv, prompt in rendered:
            out_conversations.append(conv)
            dict_prompts.append(prompt)

        tok_prompts = self.tokenize_prompts(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        # TODO: Apply multi-modal processor
        return out_conversations, tok_prompts

    async def render_chat_async(
        self,
        conversations: Sequence[list["ChatCompletionMessageParam"]],
        chat_params: ChatParams,
        tok_params: TokenizeParams,
        *,
        prompt_extras: dict[str, Any] | None = None,
    ):
        rendered = [
            self.render_messages_async(conversation, chat_params)
            for conversation in conversations
        ]

        out_conversations = list[list["ConversationMessage"]]()
        dict_prompts = list[DictPrompt]()
        for conv, prompt in await asyncio.gather(*rendered):
            out_conversations.append(conv)
            dict_prompts.append(prompt)

        tok_prompts = await self.tokenize_prompts_async(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        # TODO: Apply multi-modal processor
        return out_conversations, tok_prompts
