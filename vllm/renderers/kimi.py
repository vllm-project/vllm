# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Renderer for Kimi models using KimiTokenizer."""

from typing import Any

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.tokenizers import cached_get_tokenizer
from vllm.tokenizers.kimi import KimiTokenizer

from .hf import HfRenderer
from .protocol import BaseRenderer

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
        BaseRenderer.__init__(self, config)

        self.use_unified_vision_chunk = getattr(
            config.hf_config, "use_unified_vision_chunk", False
        )

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
        self._kimia_prompt_prefix = "<|im_kimia_user_msg_start|>"
        self._kimia_prompt_suffix = "<|im_msg_end|><|im_kimia_assistant_msg_start|>"

    @property
    def tokenizer(self) -> KimiTokenizer | None:
        return self._tokenizer

    def get_tokenizer(self) -> KimiTokenizer:
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer not available when `skip_tokenizer_init=True`")
        return tokenizer

    def render_prompt(self, prompt):
        prompt = super().render_prompt(prompt)
        if (
            isinstance(prompt, dict)
            and "prompt" in prompt
            and "prompt_token_ids" not in prompt
        ):
            prompt_text = prompt["prompt"]
            if isinstance(prompt_text, str) and not self._is_kimia_wrapped(prompt_text):
                prompt["prompt"] = (
                    f"{self._kimia_prompt_prefix}{prompt_text}"
                    f"{self._kimia_prompt_suffix}"
                )
        return prompt

    def _is_kimia_wrapped(self, prompt_text: str) -> bool:
        return (
            self._kimia_prompt_prefix in prompt_text
            or "<|im_kimia_assistant_msg_start|>" in prompt_text
            or "<|im_msg_end|>" in prompt_text
        )
