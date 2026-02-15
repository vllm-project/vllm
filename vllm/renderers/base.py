# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, overload

from typing_extensions import TypeVar

from vllm.inputs import EmbedsPrompt, TextPrompt, TokensPrompt
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.utils.async_utils import AsyncMicrobatchTokenizer
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.metrics.stats import MultiModalCacheStats

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
    from vllm.config import VllmConfig
    from vllm.entrypoints.chat_utils import (
        ChatCompletionMessageParam,
        ConversationMessage,
    )
    from vllm.multimodal.cache import BaseMultiModalProcessorCache
    from vllm.multimodal.processing import BaseMultiModalProcessor

logger = init_logger(__name__)


_T = TypeVar("_T", bound=TokenizerLike, default=TokenizerLike)


class BaseRenderer(ABC, Generic[_T]):
    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: "VllmConfig",
        tokenizer_kwargs: dict[str, Any],
    ) -> "BaseRenderer":
        raise NotImplementedError

    def __init__(self, config: "VllmConfig", tokenizer: _T | None) -> None:
        super().__init__()

        self.config = config
        self.model_config = config.model_config

        self.tokenizer = tokenizer

        # Lazy initialization since offline LLM doesn't use async
        self._async_tokenizer: AsyncMicrobatchTokenizer | None = None

        self.mm_processor: BaseMultiModalProcessor | None = None
        self._mm_cache_stats: MultiModalCacheStats | None = None
        if config.model_config.is_multimodal_model:
            from vllm.multimodal import MULTIMODAL_REGISTRY as mm_registry

            mm_processor_cache = mm_registry.processor_cache_from_config(config)

            with set_default_torch_num_threads():
                self.mm_processor = mm_registry.create_processor(
                    config.model_config,
                    config.observability_config,
                    tokenizer=tokenizer,
                    cache=mm_processor_cache,
                )

            if mm_processor_cache:
                self._mm_cache_stats = MultiModalCacheStats()

    def get_tokenizer(self) -> _T:
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer not available when `skip_tokenizer_init=True`")

        return tokenizer

    def get_async_tokenizer(self) -> AsyncMicrobatchTokenizer:
        if self._async_tokenizer is None:
            self._async_tokenizer = AsyncMicrobatchTokenizer(self.get_tokenizer())

        return self._async_tokenizer

    def get_mm_processor(self) -> "BaseMultiModalProcessor":
        if self.mm_processor is None:
            raise ValueError("Multi-modal processor not available for text-only models")

        return self.mm_processor

    @property
    def mm_processor_cache(self) -> "BaseMultiModalProcessorCache | None":
        if self.mm_processor is None:
            return None

        return self.mm_processor.cache

    def stat_mm_cache(self) -> MultiModalCacheStats | None:
        mm_cache_stats = self._mm_cache_stats
        if mm_cache_stats is None:
            return None

        self._mm_cache_stats = MultiModalCacheStats()

        return mm_cache_stats

    def update_mm_cache_stats(self) -> None:
        mm_processor_cache = self.mm_processor_cache
        mm_cache_stats = self._mm_cache_stats

        if mm_processor_cache and mm_cache_stats:
            delta = mm_processor_cache.make_stats(delta=True)
            mm_cache_stats.record(delta.total, delta.hits)

    def clear_mm_cache(self) -> None:
        mm_processor_cache = self.mm_processor_cache
        if mm_processor_cache is not None:
            mm_processor_cache.clear_cache()

        if self._mm_cache_stats is not None:
            self._mm_cache_stats.reset = True

    def shutdown(self) -> None:
        mm_processor_cache = self.mm_processor_cache
        if mm_processor_cache is not None:
            mm_processor_cache.close()

    def get_bos_token_id(self) -> int | None:
        if self.tokenizer is None:
            logger.warning_once(
                "Using None for BOS token id because tokenizer is not initialized"
            )
            return None

        return self.tokenizer.bos_token_id

    def get_eos_token_id(self) -> int | None:
        if self.tokenizer is None:
            logger.warning_once(
                "Using None for EOS token id because tokenizer is not initialized"
            )
            return None

        return self.tokenizer.eos_token_id

    @cached_property
    def default_cmpl_tok_params(self) -> TokenizeParams:
        mm_processor = self.mm_processor
        if mm_processor is not None:
            return mm_processor.info.default_tok_params

        model_config = self.model_config
        encoder_config = model_config.encoder_config or {}

        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=True,
        )

    @cached_property
    def default_chat_tok_params(self) -> TokenizeParams:
        mm_processor = self.mm_processor
        if mm_processor is not None:
            return mm_processor.info.default_tok_params

        model_config = self.model_config
        encoder_config = model_config.encoder_config or {}

        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=False,
        )

    # Step 1: Convert raw inputs to prompts
    def render_prompt(
        self,
        prompt: DictPrompt | bytes,
    ) -> DictPrompt:
        if isinstance(prompt, bytes):
            embeds = safe_load_prompt_embeds(self.model_config, prompt)
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
            target_prompt = extract_target_prompt(self.model_config, prompt)
            target_prompt.update(prompt_extras)  # type: ignore[arg-type]

    # Top-level methods
    def render_cmpl(
        self,
        prompts: Sequence[DictPrompt | bytes],
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
    ):
        if tok_params is None:
            tok_params = self.default_cmpl_tok_params

        dict_prompts = self.render_prompts(prompts)
        tok_prompts = self.tokenize_prompts(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        # TODO: Apply multi-modal processor
        return tok_prompts

    async def render_cmpl_async(
        self,
        prompts: Sequence[DictPrompt | bytes],
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
    ):
        if tok_params is None:
            tok_params = self.default_cmpl_tok_params

        dict_prompts = await self.render_prompts_async(prompts)
        tok_prompts = await self.tokenize_prompts_async(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        # TODO: Apply multi-modal processor
        return tok_prompts

    def render_chat(
        self,
        conversations: Sequence[list["ChatCompletionMessageParam"]],
        chat_params: ChatParams,
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
    ):
        if tok_params is None:
            tok_params = self.default_chat_tok_params

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
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
    ):
        if tok_params is None:
            tok_params = self.default_chat_tok_params

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
