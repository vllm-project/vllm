# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings


def __getattr__(name: str):
    if name == "TokenizerBase":
        from .tokenizers import TokenizerLike

        warnings.warn(
            "`vllm.transformers_utils.tokenizer_base.TokenizerBase` "
            "has been moved to `vllm.transformers_utils.tokenizers.TokenizerLike`. "
            "The old name will be removed in v0.13.",
            DeprecationWarning,
            stacklevel=2,
        )

        return TokenizerLike
    if name == "TokenizerRegistry":
        from .registry import TokenizerRegistry

        warnings.warn(
            "`vllm.transformers_utils.tokenizer_base.TokenizerRegistry` "
            "has been moved to `vllm.transformers_utils.registry.TokenizerRegistry`. "
            "The old name will be removed in v0.13.",
            DeprecationWarning,
            stacklevel=2,
        )

        return TokenizerRegistry

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
