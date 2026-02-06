# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, overload

from vllm.inputs.data import (
    EmbedsPrompt,
    PromptType,
    SingletonPrompt,
    TextPrompt,
    TokensPrompt,
)
from vllm.renderers.inputs import DictPromptType, SingletonDictPrompt
from vllm.renderers.inputs.parse import parse_dec_only_prompt, parse_enc_dec_prompt
from vllm.tokenizers import TokenizerLike
from vllm.utils.async_utils import AsyncMicrobatchTokenizer

from .embed_utils import safe_load_prompt_embeds
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
    @overload
    def render_completion(
        self,
        prompt: SingletonPrompt | bytes,
    ) -> SingletonDictPrompt: ...

    @overload
    def render_completion(  # type: ignore[misc]
        self,
        prompt: PromptType,
    ) -> DictPromptType: ...

    def render_completion(
        self,
        prompt: PromptType | bytes,
    ) -> DictPromptType:
        if isinstance(prompt, bytes):
            embeds = safe_load_prompt_embeds(self.config, prompt)
            prompt = EmbedsPrompt(prompt_embeds=embeds)

        if self.config.is_encoder_decoder:
            return parse_enc_dec_prompt(prompt)

        return parse_dec_only_prompt(prompt)

    @overload
    def render_completions(
        self,
        prompts: Sequence[SingletonPrompt | bytes],
    ) -> list[SingletonDictPrompt]: ...

    @overload
    def render_completions(  # type: ignore[misc]
        self,
        prompts: Sequence[PromptType],
    ) -> list[DictPromptType]: ...

    def render_completions(
        self,
        prompts: Sequence[PromptType | bytes],
    ) -> list[SingletonDictPrompt] | list[DictPromptType]:
        if len(prompts) == 0:
            raise ValueError("You must pass at least one prompt")

        return [self.render_completion(prompt) for prompt in prompts]

    @overload
    async def render_completions_async(
        self,
        prompts: Sequence[SingletonPrompt | bytes],
    ) -> list[SingletonDictPrompt]: ...

    @overload
    async def render_completions_async(  # type: ignore[misc]
        self,
        prompts: Sequence[PromptType],
    ) -> list[DictPromptType]: ...

    async def render_completions_async(
        self,
        prompts: Sequence[PromptType | bytes],
    ) -> list[SingletonDictPrompt] | list[DictPromptType]:
        return self.render_completions(prompts)

    @abstractmethod
    def render_messages(
        self,
        messages: list["ChatCompletionMessageParam"],
        params: ChatParams,
    ) -> tuple[list["ConversationMessage"], SingletonDictPrompt]:
        raise NotImplementedError

    async def render_messages_async(
        self,
        messages: list["ChatCompletionMessageParam"],
        params: ChatParams,
    ) -> tuple[list["ConversationMessage"], SingletonDictPrompt]:
        return self.render_messages(messages, params)

    # Step 2: Tokenize prompts if necessary
    def tokenize_prompt(
        self,
        prompt: TextPrompt | TokensPrompt | EmbedsPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt | EmbedsPrompt:
        if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
            prompt = params.apply_pre_tokenization(self.tokenizer, prompt)

            tokenizer = self.get_tokenizer()
            prompt_token_ids = tokenizer.encode(
                prompt["prompt"],
                **params.get_encode_kwargs(),
            )

            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids, **prompt)  # type: ignore[typeddict-unknown-key]

        if params.needs_detokenization and "prompt" not in prompt:
            if "prompt_token_ids" not in prompt:
                raise RuntimeError("Cannot run detokenization on embeddings")

            tokenizer = self.get_tokenizer()
            prompt_text = tokenizer.decode(prompt["prompt_token_ids"])  # type: ignore[typeddict-item]
            prompt["prompt"] = prompt_text  # type: ignore[typeddict-unknown-key]

        return params.apply_post_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]

    def tokenize_prompts(
        self,
        prompts: Sequence[TextPrompt | TokensPrompt | EmbedsPrompt],
        params: TokenizeParams,
    ) -> list[TokensPrompt | EmbedsPrompt]:
        return [self.tokenize_prompt(prompt, params) for prompt in prompts]

    async def tokenize_prompt_async(
        self,
        prompt: TextPrompt | TokensPrompt | EmbedsPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt | EmbedsPrompt:
        if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
            prompt = params.apply_pre_tokenization(self.tokenizer, prompt)

            tokenizer = self.get_async_tokenizer()
            prompt_token_ids = await tokenizer.encode(
                prompt["prompt"],
                **params.get_encode_kwargs(),
            )

            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids, **prompt)  # type: ignore[typeddict-unknown-key]

        if params.needs_detokenization and "prompt" not in prompt:
            if "prompt_token_ids" not in prompt:
                raise RuntimeError("Cannot run detokenization on embeddings")

            tokenizer = self.get_async_tokenizer()
            prompt_text = await tokenizer.decode(prompt["prompt_token_ids"])  # type: ignore[typeddict-item]
            prompt["prompt"] = prompt_text  # type: ignore[typeddict-unknown-key]

        return params.apply_post_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]

    async def tokenize_prompts_async(
        self,
        prompts: Sequence[TextPrompt | TokensPrompt | EmbedsPrompt],
        params: TokenizeParams,
    ) -> list[TokensPrompt | EmbedsPrompt]:
        return await asyncio.gather(
            *(self.tokenize_prompt_async(prompt, params) for prompt in prompts)
        )
