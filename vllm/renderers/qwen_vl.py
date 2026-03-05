# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from vllm.config import VllmConfig
from vllm.tokenizers import cached_get_tokenizer
from vllm.tokenizers.qwen_vl import QwenVLTokenizer

from .base import BaseRenderer
from .hf import HfRenderer


class QwenVLRenderer(BaseRenderer[QwenVLTokenizer]):
    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        config: VllmConfig,
        tokenizer_kwargs: dict[str, Any],
    ) -> "HfRenderer":
        model_config = config.model_config
        if model_config.skip_tokenizer_init:
            tokenizer = None
        else:
            tokenizer = cached_get_tokenizer(
                tokenizer_cls=QwenVLTokenizer,
                **tokenizer_kwargs,
            )

        return HfRenderer(config, tokenizer)
