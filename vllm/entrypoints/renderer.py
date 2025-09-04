# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from abc import ABC, abstractmethod
from typing import Annotated, Optional, Union

from pydantic import Field

from vllm.config import ModelConfig
from vllm.inputs.data import TokensPrompt as EngineTokensPrompt
from vllm.inputs.parse import parse_and_batch_prompt
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import AsyncMicrobatchTokenizer


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
        prompt_or_prompts: Union[str, list[str], list[int], list[list[int]]],
        max_length: Optional[int] = None,
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=-1)]] = None,
        add_special_tokens: Optional[bool] = True,
        cache_salt: Optional[str] = None,
    ) -> list[EngineTokensPrompt]:
        """
        Convert input prompts into tokenized format for engine processing.
        
        This is the core method that transforms various input formats into
        standardized TokensPrompt objects. Implementations should handle
        tokenization, special token insertion, truncation, and validation
        according to model requirements.
        
        Args:
            prompt_or_prompts: Input data in various formats:
                - str: Single text prompt
                - list[str]: Batch of text prompts  
                - list[int]: Pre-tokenized sequence
                - list[list[int]]: Batch of pre-tokenized sequences
            max_length: Maximum sequence length (endpoint-specific behavior)
            truncate_prompt_tokens: Truncate to last N tokens
                (None=no truncation, 0=empty)
            add_special_tokens: Add model-specific tokens (e.g., [CLS], [SEP])
                to text inputs
            cache_salt: Optional string to disambiguate cached prompts
            
        Returns:
            list[EngineTokensPrompt]: Tokenized prompts ready for engine 
                consumption
            
        Raises:
            ValueError: If input format is invalid or length limits exceeded
        """
        raise NotImplementedError


class CompletionRenderer(BaseRenderer):

    def __init__(
        self,
        model_config: ModelConfig,
        tokenizer: Optional[AnyTokenizer] = None,
        async_tokenizer_pool: Optional[dict[AnyTokenizer,
                                            AsyncMicrobatchTokenizer]] = None,
    ):
        super().__init__(model_config, tokenizer)
        self.async_tokenizer_pool = async_tokenizer_pool or {}
        self.async_tokenizer: Optional[AsyncMicrobatchTokenizer] = None

    async def render_prompt(
        self,
        prompt_or_prompts: Union[str, list[str], list[int], list[list[int]]],
        max_length: Optional[int] = None,
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=-1)]] = None,
        add_special_tokens: Optional[bool] = True,
        cache_salt: Optional[str] = None,
    ) -> list[EngineTokensPrompt]:
        """Implementation of prompt rendering for completion-style requests.
        
        Uses async tokenizer pooling for improved performance. See base class
        for detailed parameter documentation.
        """
        if truncate_prompt_tokens is not None:
            if max_length is not None:
                assert 0 <= truncate_prompt_tokens <= max_length
            if truncate_prompt_tokens == 0:
                return []

        # Parse and batch the input prompts
        batch_inputs = parse_and_batch_prompt(prompt_or_prompts)

        rendered_prompts: list[EngineTokensPrompt] = []
        tokenize_tasks = []
        for prompt_input in batch_inputs:
            if prompt_input["is_tokens"] is True:
                # Token input
                token_ids = self._maybe_apply_truncation(
                    prompt_input["content"], truncate_prompt_tokens)
                rendered_prompts.append(
                    self._create_tokens_prompt(token_ids, max_length,
                                               cache_salt))
            else:
                # Text input
                tokenize_task = asyncio.create_task(
                    self._tokenize(prompt_input["content"], max_length,
                                   truncate_prompt_tokens, add_special_tokens,
                                   cache_salt))
                tokenize_tasks.append(tokenize_task)

        # Wait for all text tokenization to finish
        if tokenize_tasks:
            tokenized_text_prompts = await asyncio.gather(*tokenize_tasks)
            rendered_prompts.extend(tokenized_text_prompts)

        return rendered_prompts

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
                                          cache_salt)

    def _get_async_tokenizer(self) -> AsyncMicrobatchTokenizer:
        """Get or create async tokenizer using shared pool."""
        if self.async_tokenizer is not None:
            return self.async_tokenizer
        if self.tokenizer is None:
            raise ValueError(
                "No tokenizer available for text input processing")

        # Check shared pool first
        if self.tokenizer in self.async_tokenizer_pool:
            return self.async_tokenizer_pool[self.tokenizer]

        # Create new async tokenizer and add to pool
        self.async_tokenizer = AsyncMicrobatchTokenizer(self.tokenizer)
        self.async_tokenizer_pool[self.tokenizer] = self.async_tokenizer
        return self.async_tokenizer

    def _create_tokens_prompt(
        self,
        token_ids: list[int],
        max_length: Optional[int] = None,
        cache_salt: Optional[str] = None,
    ) -> EngineTokensPrompt:
        """Create validated EngineTokensPrompt."""
        if max_length is not None and len(token_ids) > max_length:
            raise ValueError(
                f"This maximum context length is {max_length} tokens. "
                f"However, your request has {len(token_ids)} input tokens. "
                "Please reduce the length of the input messages.")

        tokens_prompt = EngineTokensPrompt(prompt_token_ids=token_ids)
        if cache_salt is not None:
            tokens_prompt["cache_salt"] = cache_salt
        return tokens_prompt
