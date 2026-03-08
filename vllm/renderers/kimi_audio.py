# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, cast

from vllm.config import VllmConfig
from vllm.tokenizers import cached_get_tokenizer
from vllm.tokenizers.kimi_audio import KimiAudioTokenizer

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
            tokenizer = cast(
                HfTokenizer,
                cached_get_tokenizer(
                    tokenizer_cls=KimiAudioTokenizer,  # type: ignore[arg-type]
                    **tokenizer_kwargs,
                ),
            )

        return HfRenderer(config, tokenizer)
