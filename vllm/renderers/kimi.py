# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Renderer for Kimi models using KimiTokenizer."""

from typing import Any

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.tokenizers import cached_get_tokenizer
from vllm.tokenizers.kimi import KimiTokenizer

from .hf import HfRenderer

logger = init_logger(__name__)


class KimiRenderer(HfRenderer):
    """Renderer for Kimi models.

    Inherits from HfRenderer but uses KimiTokenizer instead of CachedHfTokenizer
    to properly handle TikTokenTokenizer-based models.
    """

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        tokenizer_kwargs: dict[str, Any],
    ) -> "KimiRenderer":
        return cls(config, tokenizer_kwargs)

    def __init__(
        self,
        config: ModelConfig,
        tokenizer_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(config)

        # Initialize without calling HfRenderer.__init__ to avoid creating
        # the HF tokenizer. We'll create KimiTokenizer instead.

        if config.skip_tokenizer_init:
            tokenizer = None
        else:
            tokenizer = cached_get_tokenizer(
                tokenizer_cls=KimiTokenizer,
                **tokenizer_kwargs,
            )

        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> KimiTokenizer | None:
        return self._tokenizer

    def get_tokenizer(self) -> KimiTokenizer:
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer not available when `skip_tokenizer_init=True`")
        return tokenizer
