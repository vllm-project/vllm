# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, cast

from vllm.config import VllmConfig
from vllm.tokenizers.kimi_audio import KimiAudioTokenizer
from vllm.tokenizers.registry import get_tokenizer

from .hf import HfRenderer, HfTokenizer


class KimiAudioRenderer(HfRenderer):
    """Renderer for Kimi-Audio models.

    This renderer uses HfRenderer internally with a custom TikToken tokenizer.
    """

    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        config: VllmConfig,
        tokenizer_kwargs: dict[str, Any],
    ) -> "HfRenderer":
        """Create an HfRenderer instance for Kimi-Audio models."""
        model_config = config.model_config
        if model_config.skip_tokenizer_init:
            tokenizer = None
        else:
            # Extract tokenizer_name from kwargs (already processed by
            # tokenizer_args_from_config for ModelScope/GGUF/etc)
            tokenizer_name = tokenizer_kwargs.pop(
                "tokenizer_name", model_config.tokenizer
            )
            # Remove tokenizer_cls from kwargs to avoid duplicate argument
            tokenizer_kwargs = {
                k: v for k, v in tokenizer_kwargs.items() if k != "tokenizer_cls"
            }
            # Use get_tokenizer directly instead of cached_get_tokenizer
            # (KimiAudioTokenizer doesn't work with get_cached_tokenizer)
            tokenizer = cast(
                HfTokenizer,
                get_tokenizer(
                    tokenizer_name,
                    tokenizer_cls=KimiAudioTokenizer,  # type: ignore[arg-type]
                    **tokenizer_kwargs,
                ),
            )

        return HfRenderer(config, tokenizer)
