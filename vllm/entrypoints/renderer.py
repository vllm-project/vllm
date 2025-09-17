# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated, Optional, Union

import pybase64
import torch
from pydantic import Field

from vllm.config import ModelConfig
from vllm.inputs.data import EmbedsPrompt as EngineEmbedsPrompt
from vllm.inputs.data import TokensPrompt as EngineTokensPrompt
from vllm.inputs.parse import parse_and_batch_prompt
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import AsyncMicrobatchTokenizer


@dataclass(frozen=True)
class RenderConfig:
    """Configuration to control how prompts are prepared."""

    max_length: Optional[int] = None
    """Maximum allowable total input token length. If provided,
    token inputs longer than this raise ``ValueError``."""

    truncate_prompt_tokens: Optional[int] = None
    """Number of tokens to keep. ``None`` means no truncation.
    ``0`` yields an empty list (and skips embeds).
    ``-1`` maps to ``model_config.max_model_len``."""

    add_special_tokens: Optional[bool] = True
    """Whether to add model-specific special tokens during tokenization."""

    cache_salt: Optional[str] = None
    """String to disambiguate prefix cache entries."""

    needs_detokenization: Optional[bool] = False
    """If True, detokenize IDs back to text for inclusion in outputs."""


class BaseRenderer(ABC):
    """
    Base class for unified input processing and rendering.
    
    The Renderer serves as a unified input processor that consolidates
    tokenization, chat template formatting, and multimodal input handling
    into a single component.
    It converts high-level API requests (OpenAI-style JSON) into token IDs and
    multimodal features ready for engine consumption.
    
    Key responsibilities:
    - Convert text prompts to token sequences with proper special tokens
    - Apply chat templates and format conversations
    - Handle multimodal inputs (images, audio, etc.) when applicable
    - Manage prompt truncation and length validation
    - Provide clean separation between API layer and engine core
    """

    def __init__(
        self,
        model_config: ModelConfig,
        tokenizer: Optional[AnyTokenizer] = None,
    ):
        super().__init__()
        self.model_config = model_config
        self.tokenizer = tokenizer

    @abstractmethod
    async def render_prompt(
        self,
        *,
        prompt_or_prompts: Union[str, list[str], list[int], list[list[int]]],
        config: "RenderConfig",
    ) -> list[EngineTokensPrompt]:
        """
        Convert text or token inputs into engine-ready TokensPrompt objects.

        This method accepts text or token inputs and produces a
        list of [`TokensPrompt`][vllm.inputs.data.TokensPrompt] objects
        for the engine.

        Args:
            prompt_or_prompts: One of:
                - ``str``: Single text prompt.
                - ``list[str]``: Batch of text prompts.
                - ``list[int]``: Single pre-tokenized sequence.
                - ``list[list[int]]``: Batch of pre-tokenized sequences.
            config: Render configuration controlling how prompts are prepared
                (e.g., tokenization and length handling). 

        Returns:
            list[EngineTokensPrompt]: Engine-ready token prompts.

        Raises:
            ValueError: If input formats are invalid or length limits exceeded.
        """
        raise NotImplementedError

    @abstractmethod
    async def render_prompt_and_embeds(
        self,
        *,
        prompt_or_prompts: Optional[Union[str, list[str], list[int],
                                          list[list[int]]]] = None,
        prompt_embeds: Optional[Union[bytes, list[bytes]]] = None,
        config: "RenderConfig",
    ) -> list[Union[EngineTokensPrompt, EngineEmbedsPrompt]]:
        """
        Convert text/token and/or base64-encoded embeddings inputs into
        engine-ready prompt objects using a unified RenderConfig.

        At least one of ``prompt_or_prompts`` or ``prompt_embeds`` must be
        provided and non-empty. If both are omitted or empty (e.g., empty
        string and empty list), a ``ValueError`` is raised.

        Args:
            prompt_or_prompts: Text or token inputs to include.
            prompt_embeds: Base64-encoded bytes (or list thereof) containing a
                torch-saved tensor to be used as prompt embeddings.
            config: Render configuration controlling how prompts are prepared
                (e.g., tokenization and length handling). 

        Returns:
            list[Union[EngineTokensPrompt, EngineEmbedsPrompt]]:
                Engine-ready prompt objects.

        Raises:
            ValueError: If both ``prompt_or_prompts`` and ``prompt_embeds``
                are omitted or empty (decoder prompt cannot be empty), or if
                length limits are exceeded.
        """
        raise NotImplementedError

    @classmethod
    def load_prompt_embeds(
        cls,
        prompt_embeds: Union[bytes, list[bytes]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=0)]] = None,
        cache_salt: Optional[str] = None,
    ) -> list[EngineEmbedsPrompt]:
        """Load and validate base64-encoded embeddings into prompt objects."""

        def _load_and_validate_embed(embed: bytes) -> EngineEmbedsPrompt:
            tensor = torch.load(
                io.BytesIO(pybase64.b64decode(embed, validate=True)),
                weights_only=True,
                map_location=torch.device("cpu"),
            )
            assert isinstance(tensor, torch.Tensor) and tensor.dtype in (
                torch.float32,
                torch.bfloat16,
                torch.float16,
            )
            tensor = tensor.to_dense()
            if tensor.dim() > 2:
                tensor = tensor.squeeze(0)
                assert tensor.dim() == 2
            if truncate_prompt_tokens is not None:
                tensor = tensor[-truncate_prompt_tokens:]
            embeds_prompt = EngineEmbedsPrompt(prompt_embeds=tensor)
            if cache_salt is not None:
                embeds_prompt["cache_salt"] = cache_salt
            return embeds_prompt

        if isinstance(prompt_embeds, list):
            return [_load_and_validate_embed(embed) for embed in prompt_embeds]

        return [_load_and_validate_embed(prompt_embeds)]


class CompletionRenderer(BaseRenderer):

    def __init__(
        self,
        model_config: ModelConfig,
        tokenizer: Optional[AnyTokenizer] = None,
        async_tokenizer_pool: Optional[dict[AnyTokenizer,
                                            AsyncMicrobatchTokenizer]] = None,
    ):
        super().__init__(model_config, tokenizer)
        self.async_tokenizer_pool = async_tokenizer_pool
        self.async_tokenizer: Optional[AsyncMicrobatchTokenizer] = None

    async def render_prompt(
        self,
        *,
        prompt_or_prompts: Union[str, list[str], list[int], list[list[int]]],
        config: "RenderConfig",
    ) -> list[EngineTokensPrompt]:
        """Implementation of prompt rendering for completion-style requests.
        
        Uses async tokenizer pooling for improved performance. See base class
        for detailed parameter documentation.
        """
        truncate_prompt_tokens = self._validate_and_normalize_truncate_tokens(
            config.truncate_prompt_tokens, config.max_length)
        if truncate_prompt_tokens == 0:
            return []

        # Parse and batch the input prompts
        batch_inputs = parse_and_batch_prompt(prompt_or_prompts)

        tasks = []
        for prompt_input in batch_inputs:
            if prompt_input["is_tokens"] is True:
                # Token input
                # Note: detokenization is needed when echo is enabled,
                # where the input token IDs are decoded back to text.
                task = self._maybe_detokenize(prompt_input["content"],
                                              config.max_length,
                                              truncate_prompt_tokens,
                                              config.cache_salt,
                                              config.needs_detokenization)
            else:
                # Text input
                task = self._tokenize(prompt_input["content"],
                                      config.max_length,
                                      truncate_prompt_tokens,
                                      config.add_special_tokens,
                                      config.cache_salt)
            tasks.append(task)

        # Wait for all text tokenization to finish
        if tasks:
            tokenized_text_prompts = await asyncio.gather(*tasks)
            return tokenized_text_prompts

        return []

    async def render_prompt_and_embeds(
        self,
        *,
        prompt_or_prompts: Optional[Union[str, list[str], list[int],
                                          list[list[int]]]] = None,
        prompt_embeds: Optional[Union[bytes, list[bytes]]] = None,
        config: "RenderConfig",
    ) -> list[Union[EngineTokensPrompt, EngineEmbedsPrompt]]:
        """
        Render text/token prompts and/or precomputed embedding prompts. At
        least one of `prompt_or_prompts` or `prompt_embeds` must be provided.
        """
        truncate_prompt_tokens = self._validate_and_normalize_truncate_tokens(
            config.truncate_prompt_tokens, config.max_length)
        if truncate_prompt_tokens == 0:
            return []

        rendered: list[Union[EngineTokensPrompt, EngineEmbedsPrompt]] = []

        if prompt_embeds is not None:
            rendered.extend(
                self.load_prompt_embeds(prompt_embeds, truncate_prompt_tokens,
                                        config.cache_salt))
        if prompt_or_prompts is None or prompt_or_prompts == "":
            return rendered

        token_prompts = await self.render_prompt(
            prompt_or_prompts=prompt_or_prompts,
            config=config,
        )
        rendered.extend(token_prompts)

        return rendered

    def _validate_and_normalize_truncate_tokens(
        self,
        truncate_prompt_tokens: Optional[int],
        max_length: Optional[int],
    ) -> Optional[int]:
        """Validate and normalize truncate_prompt_tokens parameter."""
        if truncate_prompt_tokens is None:
            return None

        if truncate_prompt_tokens == 0:
            return 0

        if truncate_prompt_tokens < 0:
            truncate_prompt_tokens = self.model_config.max_model_len

        if max_length is not None and truncate_prompt_tokens > max_length:
            raise ValueError(
                f"truncate_prompt_tokens ({truncate_prompt_tokens}) "
                f"cannot be greater than max_length ({max_length}). "
                f"Please select a smaller truncation size.")

        return truncate_prompt_tokens

    def _maybe_apply_truncation(
            self, token_ids: list[int],
            truncate_prompt_tokens: Optional[int]) -> list[int]:
        """Apply truncation to token sequence."""
        if truncate_prompt_tokens is None:
            return token_ids
        if truncate_prompt_tokens >= len(token_ids):
            return token_ids

        return token_ids[-truncate_prompt_tokens:]

    async def _tokenize(
        self,
        text: str,
        max_length: Optional[int],
        truncate_prompt_tokens: Optional[int],
        add_special_tokens: Optional[bool],
        cache_salt: Optional[str],
    ) -> EngineTokensPrompt:
        """Tokenize text input asynchronously."""
        async_tokenizer = self._get_async_tokenizer()

        # Handle encoder-specific preprocessing
        if (self.model_config.encoder_config is not None
                and self.model_config.encoder_config.get(
                    "do_lower_case", False)):
            text = text.lower()

        # Tokenize texts
        if truncate_prompt_tokens is None:
            encoded = await async_tokenizer(
                text, add_special_tokens=add_special_tokens)
        else:
            encoded = await async_tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                truncation=True,
                max_length=truncate_prompt_tokens)

        return self._create_tokens_prompt(encoded.input_ids, max_length,
                                          cache_salt, text)

    async def _maybe_detokenize(
        self,
        token_ids: list[int],
        max_length: Optional[int],
        truncate_prompt_tokens: Optional[int],
        cache_salt: Optional[str],
        needs_detokenization: Optional[bool] = False,
    ) -> EngineTokensPrompt:
        """Optionally detokenize token IDs and build a tokens prompt."""
        token_ids = self._maybe_apply_truncation(token_ids,
                                                 truncate_prompt_tokens)

        prompt = None
        if needs_detokenization is True:
            async_tokenizer = self._get_async_tokenizer()
            prompt = await async_tokenizer.decode(token_ids)

        return self._create_tokens_prompt(token_ids=token_ids,
                                          max_length=max_length,
                                          cache_salt=cache_salt,
                                          prompt=prompt)

    def _get_async_tokenizer(self) -> AsyncMicrobatchTokenizer:
        """Get or create async tokenizer using shared pool."""
        async_tokenizer = self.async_tokenizer
        if async_tokenizer is not None:
            return async_tokenizer

        tokenizer = self.tokenizer
        if self.tokenizer is None:
            raise ValueError(
                "No tokenizer available for text input processing")

        if self.async_tokenizer_pool is None:
            async_tokenizer = AsyncMicrobatchTokenizer(tokenizer)
        else:
            async_tokenizer = self.async_tokenizer_pool.get(tokenizer)
            if async_tokenizer is None:
                async_tokenizer = AsyncMicrobatchTokenizer(tokenizer)
                self.async_tokenizer_pool[tokenizer] = async_tokenizer
        self.async_tokenizer = async_tokenizer
        return async_tokenizer

    def _create_tokens_prompt(
        self,
        token_ids: list[int],
        max_length: Optional[int] = None,
        cache_salt: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> EngineTokensPrompt:
        """Create validated EngineTokensPrompt."""
        if max_length is not None and len(token_ids) > max_length:
            raise ValueError(
                f"This model's maximum context length is {max_length} tokens. "
                f"However, your request has {len(token_ids)} input tokens. "
                "Please reduce the length of the input messages.")

        tokens_prompt = EngineTokensPrompt(prompt_token_ids=token_ids)
        if cache_salt is not None:
            tokens_prompt["cache_salt"] = cache_salt
        if prompt is not None:
            tokens_prompt["prompt"] = prompt
        return tokens_prompt
