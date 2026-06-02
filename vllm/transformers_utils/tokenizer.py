# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import warnings


def __getattr__(name: str):
    # Keep until lm-eval is updated
    if name == "get_tokenizer":
        from vllm.tokenizers import get_tokenizer

        warnings.warn(
            "`vllm.transformers_utils.tokenizer.get_tokenizer` "
            "has been moved to `vllm.tokenizers.get_tokenizer`. "
            "The old name will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

        return get_tokenizer
    
    # cohere start
    if name == "init_tokenizer_from_configs":
        from vllm.tokenizers import cached_tokenizer_from_config

        warnings.warn(
            "`vllm.transformers_utils.tokenizer.init_tokenizer_from_configs` "
            "has been moved to `vllm.tokenizers.cached_tokenizer_from_config`. "
            "The old name will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

        return cached_tokenizer_from_config
    # cohere end

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
