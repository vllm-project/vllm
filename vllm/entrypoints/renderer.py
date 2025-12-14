# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated

import pybase64
import torch
from pydantic import Field

from vllm.config import ModelConfig
from vllm.inputs.data import EmbedsPrompt, TextPrompt, TokensPrompt
from vllm.inputs.parse import get_prompt_components, parse_raw_prompts
from vllm.tokenizers import TokenizerLike
from vllm.utils.async_utils import AsyncMicrobatchTokenizer


@dataclass(frozen=True)
class RenderConfig:
    """Configuration to control how prompts are prepared."""

    max_length: int | None = None
    """Maximum allowable total input token length. If provided,
    token inputs longer than this raise `ValueError`."""

    truncate_prompt_tokens: int | None = None
    """Number of tokens to keep. `None` means no truncation.
    `0` yields an empty list (and skips embeds).
    `-1` maps to `model_config.max_model_len`."""

    add_special_tokens: bool = True
    """Whether to add model-specific special tokens during tokenization."""

    cache_salt: str | None = None
    """String to disambiguate prefix cache entries."""

    needs_detokenization: bool | None = False
    """If True, detokenize IDs back to text for inclusion in outputs."""

    def verify_truncate_prompt_tokens(self, model_config: ModelConfig) -> int | None:
        """Validate and normalize `truncate_prompt_tokens` parameter."""
        truncate_prompt_tokens = self.truncate_prompt_tokens
        if truncate_prompt_tokens is None:
            return None

        if truncate_prompt_tokens == 0:
            return 0

        if truncate_prompt_tokens < 0:
            truncate_prompt_tokens = model_config.max_model_len

        max_length = self.max_length
        if max_length is not None and truncate_prompt_tokens > max_length:  # type: ignore[operator]
            raise ValueError(
                f"{truncate_prompt_tokens=} cannot be greater than "
                f"{max_length=}. Please select a smaller truncation size."
            )

        return truncate_prompt_tokens


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
        tokenizer: TokenizerLike | None = None,
    ):
        super().__init__()
        self.model_config = model_config
        self.tokenizer = tokenizer

    @abstractmethod
    async def render_prompt(
        self,
        *,
        prompt_or_prompts: str | list[str] | list[int] | list[list[int]],
        config: RenderConfig,
    ) -> list[TokensPrompt]:
        """
        Convert text or token inputs into engine-ready TokensPrompt objects.

        This method accepts text or token inputs and produces a
        list of [`TokensPrompt`][vllm.inputs.data.TokensPrompt] objects
        for the engine.

        Args:
            prompt_or_prompts: One of:
                - `str`: Single text prompt.
                - `list[str]`: Batch of text prompts.
                - `list[int]`: Single pre-tokenized sequence.
                - `list[list[int]]`: Batch of pre-tokenized sequences.
            config: Render configuration controlling how prompts are prepared
                (e.g., tokenization and length handling).

        Returns:
            list[TokensPrompt]: Engine-ready token prompts.

        Raises:
            ValueError: If input formats are invalid or length limits exceeded.
        """
        raise NotImplementedError

    @abstractmethod
    async def render_prompt_and_embeds(
        self,
        *,
        prompt_or_prompts: str | list[str] | list[int] | list[list[int]] | None = None,
        prompt_embeds: bytes | list[bytes] | None = None,
        config: RenderConfig,
    ) -> list[TokensPrompt | EmbedsPrompt]:
        """
        Convert text/token and/or base64-encoded embeddings inputs into
        engine-ready prompt objects using a unified RenderConfig.

        At least one of `prompt_or_prompts` or `prompt_embeds` must be
        provided and non-empty. If both are omitted or empty (e.g., empty
        string and empty list), a `ValueError` is raised.

        Args:
            prompt_or_prompts: Text or token inputs to include.
            prompt_embeds: Base64-encoded bytes (or list thereof) containing a
                torch-saved tensor to be used as prompt embeddings.
            config: Render configuration controlling how prompts are prepared
                (e.g., tokenization and length handling).

        Returns:
            list[Union[TokensPrompt, EmbedsPrompt]]:
                Engine-ready prompt objects.

        Raises:
            ValueError: If both `prompt_or_prompts` and `prompt_embeds`
                are omitted or empty (decoder prompt cannot be empty), or if
                length limits are exceeded.
        """
        raise NotImplementedError

    def load_prompt_embeds(
        self,
        prompt_embeds: bytes | list[bytes],
        truncate_prompt_tokens: Annotated[int, Field(ge=0)] | None = None,
        cache_salt: str | None = None,
    ) -> list[EmbedsPrompt]:
        """Load and validate base64-encoded embeddings into prompt objects."""
        if not self.model_config.enable_prompt_embeds:
            raise ValueError(
                "You must set `--enable-prompt-embeds` to input `prompt_embeds`."
            )

        def _load_and_validate_embed(embed: bytes) -> EmbedsPrompt:
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
            embeds_prompt = EmbedsPrompt(prompt_embeds=tensor)
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
        tokenizer: TokenizerLike | None = None,
        async_tokenizer_pool: dict[TokenizerLike, AsyncMicrobatchTokenizer]
        | None = None,
    ):
        super().__init__(model_config, tokenizer)
        self.async_tokenizer_pool = async_tokenizer_pool
        self.async_tokenizer: AsyncMicrobatchTokenizer | None = None

    async def render_prompt(
        self,
        *,
        prompt_or_prompts: str | list[str] | list[int] | list[list[int]],
        config: RenderConfig,
    ) -> list[TokensPrompt]:
        """Implementation of prompt rendering for completion-style requests.

        Uses async tokenizer pooling for improved performance. See base class
        for detailed parameter documentation.
        """
        truncate_prompt_tokens = config.verify_truncate_prompt_tokens(self.model_config)
        if truncate_prompt_tokens == 0:
            return []

        tasks = (
            self._create_prompt(
                prompt_input,
                config=config,
                truncate_prompt_tokens=truncate_prompt_tokens,
            )
            for prompt_input in parse_raw_prompts(prompt_or_prompts)
        )

        return await asyncio.gather(*tasks)

    async def render_prompt_and_embeds(
        self,
        *,
        prompt_or_prompts: str | list[str] | list[int] | list[list[int]] | None = None,
        prompt_embeds: bytes | list[bytes] | None = None,
        config: RenderConfig,
    ) -> list[TokensPrompt | EmbedsPrompt]:
        """
        Render text/token prompts and/or precomputed embedding prompts. At
        least one of `prompt_or_prompts` or `prompt_embeds` must be provided.
        """
        truncate_prompt_tokens = config.verify_truncate_prompt_tokens(self.model_config)
        if truncate_prompt_tokens == 0:
            return []

        rendered: list[TokensPrompt | EmbedsPrompt] = []

        if prompt_embeds is not None:
            rendered.extend(
                self.load_prompt_embeds(
                    prompt_embeds, truncate_prompt_tokens, config.cache_salt
                )
            )
        if prompt_or_prompts is None or prompt_or_prompts == "":
            return rendered

        token_prompts = await self.render_prompt(
            prompt_or_prompts=prompt_or_prompts,
            config=config,
        )
        rendered.extend(token_prompts)

        return rendered

    def _maybe_apply_truncation(
        self, token_ids: list[int], truncate_prompt_tokens: int | None
    ) -> list[int]:
        """Apply truncation to token sequence."""
        if truncate_prompt_tokens is None:
            return token_ids
        if truncate_prompt_tokens >= len(token_ids):
            return token_ids

        return token_ids[-truncate_prompt_tokens:]

    async def _create_prompt(
        self,
        prompt_input: TextPrompt | TokensPrompt,
        config: RenderConfig,
        truncate_prompt_tokens: int | None,
    ) -> TokensPrompt:
        prompt, prompt_token_ids, _ = get_prompt_components(prompt_input)

        if prompt_token_ids is not None:
            # NOTE: detokenization is needed when echo is enabled,
            # where the input token IDs are decoded back to text.
            return await self._create_prompt_from_token_ids(
                prompt_token_ids,
                config.max_length,
                truncate_prompt_tokens,
                config.cache_salt,
                config.needs_detokenization,
            )

        if prompt is not None:
            return await self._create_prompt_from_text(
                prompt,
                config.max_length,
                truncate_prompt_tokens,
                config.add_special_tokens,
                config.cache_salt,
            )

        # TODO: Also handle embeds prompt using this method
        raise NotImplementedError

    async def _create_prompt_from_text(
        self,
        text: str,
        max_length: int | None,
        truncate_prompt_tokens: int | None,
        add_special_tokens: bool,
        cache_salt: str | None,
    ) -> TokensPrompt:
        """Tokenize text input asynchronously."""
        async_tokenizer = self._get_async_tokenizer()

        # Handle encoder-specific preprocessing
        if (
            self.model_config.encoder_config is not None
            and self.model_config.encoder_config.get("do_lower_case", False)
        ):
            text = text.lower()

        # Tokenize texts
        if truncate_prompt_tokens is None:
            encoded = await async_tokenizer(text, add_special_tokens=add_special_tokens)
        else:
            encoded = await async_tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                truncation=True,
                max_length=truncate_prompt_tokens,
            )

        return self._create_tokens_prompt(
            encoded.input_ids, max_length, cache_salt, text
        )

    async def _create_prompt_from_token_ids(
        self,
        token_ids: list[int],
        max_length: int | None,
        truncate_prompt_tokens: int | None,
        cache_salt: str | None,
        needs_detokenization: bool | None = False,
    ) -> TokensPrompt:
        """Optionally detokenize token IDs and build a tokens prompt."""
        token_ids = self._maybe_apply_truncation(token_ids, truncate_prompt_tokens)

        prompt = None
        if needs_detokenization:
            async_tokenizer = self._get_async_tokenizer()
            prompt = await async_tokenizer.decode(token_ids)

        return self._create_tokens_prompt(
            token_ids=token_ids,
            max_length=max_length,
            cache_salt=cache_salt,
            prompt=prompt,
        )

    def _get_async_tokenizer(self) -> AsyncMicrobatchTokenizer:
        """Get or create async tokenizer using shared pool."""
        async_tokenizer = self.async_tokenizer
        if async_tokenizer is not None:
            return async_tokenizer

        tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("No tokenizer available for text input processing")

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
        max_length: int | None = None,
        cache_salt: str | None = None,
        prompt: str | None = None,
    ) -> TokensPrompt:
        """Create validated TokensPrompt."""
        if max_length is not None and len(token_ids) > max_length:
            raise ValueError(
                f"This model's maximum context length is {max_length} tokens. "
                f"However, your request has {len(token_ids)} input tokens. "
                "Please reduce the length of the input messages."
            )

        tokens_prompt = TokensPrompt(prompt_token_ids=token_ids)
        if cache_salt is not None:
            tokens_prompt["cache_salt"] = cache_salt
        if prompt is not None:
            tokens_prompt["prompt"] = prompt
        return tokens_prompt
