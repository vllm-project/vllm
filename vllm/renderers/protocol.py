# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from typing import TYPE_CHECKING, Any, Protocol

from vllm.inputs import EmbedsPrompt, TextPrompt, TokensPrompt
from vllm.tokenizers import TokenizerLike
from vllm.utils.async_utils import AsyncMicrobatchTokenizer
from vllm.utils.collection_utils import is_list_of

from .embed_utils import safe_load_prompt_embeds
from .params import ChatParams, TokenizeParams

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
        prompt_raw: str | list[int] | bytes,
    ) -> TextPrompt | TokensPrompt | EmbedsPrompt:
        error_msg = "Each prompt must be a string or an array of tokens"

        if isinstance(prompt_raw, str):
            return TextPrompt(prompt=prompt_raw)

        if isinstance(prompt_raw, list):
            if not is_list_of(prompt_raw, int):
                raise TypeError(error_msg)

            return TokensPrompt(prompt_token_ids=prompt_raw)

        if isinstance(prompt_raw, bytes):
            embeds = safe_load_prompt_embeds(self.config, prompt_raw)
            return EmbedsPrompt(prompt_embeds=embeds)

        raise TypeError(error_msg)

    def render_completions(
        self,
        prompt_input: str | list[str] | list[int] | list[list[int]] | None = None,
        prompt_embeds: bytes | list[bytes] | None = None,
    ) -> list[TextPrompt | TokensPrompt | EmbedsPrompt]:
        prompts_raw = list[str | list[int] | bytes]()

        if prompt_embeds is not None:  # embeds take higher priority
            if isinstance(prompt_embeds, bytes):
                prompts_raw.append(prompt_embeds)
            else:
                prompts_raw.extend(prompt_embeds)

        if prompt_input is not None:
            if isinstance(prompt_input, str) or (
                len(prompt_input) > 0 and is_list_of(prompt_input, int)
            ):
                prompts_raw.append(prompt_input)  # type: ignore[arg-type]
            else:
                prompts_raw.extend(prompt_input)  # type: ignore[arg-type]

        if len(prompts_raw) == 0:
            raise ValueError("You must pass at least one prompt")

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
        params: ChatParams,
    ) -> tuple[list["ConversationMessage"], TextPrompt | TokensPrompt | EmbedsPrompt]:
        raise NotImplementedError

    async def render_messages_async(
        self,
        messages: list["ChatCompletionMessageParam"],
        params: ChatParams,
    ) -> tuple[list["ConversationMessage"], TextPrompt | TokensPrompt | EmbedsPrompt]:
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

            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids, **prompt)

        if params.needs_detokenization and "prompt" not in prompt:
            if "prompt_token_ids" not in prompt:
                raise RuntimeError("Cannot run detokenization on embeddings")

            tokenizer = self.get_tokenizer()
            prompt_text = tokenizer.decode(prompt["prompt_token_ids"])  # type: ignore[typeddict-item]
            prompt["prompt"] = prompt_text  # type: ignore[typeddict-unknown-key]

        return params.apply_post_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]

    def tokenize_prompts(
        self,
        prompts: list[TextPrompt | TokensPrompt | EmbedsPrompt],
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

            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids, **prompt)

        if params.needs_detokenization and "prompt" not in prompt:
            if "prompt_token_ids" not in prompt:
                raise RuntimeError("Cannot run detokenization on embeddings")

            tokenizer = self.get_async_tokenizer()
            prompt_text = await tokenizer.decode(prompt["prompt_token_ids"])  # type: ignore[typeddict-item]
            prompt["prompt"] = prompt_text  # type: ignore[typeddict-unknown-key]

        return params.apply_post_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]

    async def tokenize_prompts_async(
        self,
        prompts: list[TextPrompt | TokensPrompt | EmbedsPrompt],
        params: TokenizeParams,
    ) -> list[TokensPrompt | EmbedsPrompt]:
        return await asyncio.gather(
            *(self.tokenize_prompt_async(prompt, params) for prompt in prompts)
        )
