# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, cast

from vllm.config import ModelConfig
from vllm.renderers.hf import HfRenderer
from vllm.tokenizers import cached_get_tokenizer
from vllm.tokenizers.tiktoken import TikTokenTokenizer


class TikTokenRenderer(HfRenderer):
    def __init__(
        self,
        config: ModelConfig,
        tokenizer_kwargs: dict[str, Any],
    ) -> None:
        # Skip HfRenderer.__init__ to avoid loading CachedHfTokenizer
        super(HfRenderer, self).__init__(config)

        self.use_unified_vision_chunk = getattr(
            config.hf_config, "use_unified_vision_chunk", False
        )

        if config.skip_tokenizer_init:
            tokenizer = None
        else:
            tokenizer = cast(
                TikTokenTokenizer,
                cached_get_tokenizer(
                    tokenizer_cls=TikTokenTokenizer,
                    **tokenizer_kwargs,
                ),
            )

        self._tokenizer = tokenizer
