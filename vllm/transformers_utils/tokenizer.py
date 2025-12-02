# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings
from typing import Any

from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)


def __getattr__(name: str):
    if name == "AnyTokenizer":
        warnings.warn(
            "`vllm.transformers_utils.tokenizer.AnyTokenizer` has been moved to "
            "`vllm.tokenizers.TokenizerLike`. "
            "The old name will be removed in v0.13.",
            DeprecationWarning,
            stacklevel=2,
        )

        return TokenizerLike
    if name == "get_tokenizer":
        from vllm.tokenizers import get_tokenizer

        warnings.warn(
            "`vllm.transformers_utils.tokenizer.get_tokenizer` "
            "has been moved to `vllm.tokenizers.get_tokenizer`. "
            "The old name will be removed in v0.13.",
            DeprecationWarning,
            stacklevel=2,
        )

        return get_tokenizer
    if name == "cached_get_tokenizer":
        from vllm.tokenizers import cached_get_tokenizer

        warnings.warn(
            "`vllm.transformers_utils.tokenizer.cached_get_tokenizer` "
            "has been moved to `vllm.tokenizers.cached_get_tokenizer`. "
            "The old name will be removed in v0.13.",
            DeprecationWarning,
            stacklevel=2,
        )

        return cached_get_tokenizer
    if name == "cached_tokenizer_from_config":
        from vllm.tokenizers import cached_tokenizer_from_config

        warnings.warn(
            "`vllm.transformers_utils.tokenizer.cached_tokenizer_from_config` "
            "has been moved to `vllm.tokenizers.cached_tokenizer_from_config`. "
            "The old name will be removed in v0.13.",
            DeprecationWarning,
            stacklevel=2,
        )

        return cached_tokenizer_from_config
    if name == "init_tokenizer_from_configs":
        from vllm.tokenizers import init_tokenizer_from_config

        warnings.warn(
            "`vllm.transformers_utils.tokenizer.init_tokenizer_from_configs` "
            "has been moved to `vllm.tokenizers.init_tokenizer_from_config`. "
            "The old name will be removed in v0.13.",
            DeprecationWarning,
            stacklevel=2,
        )

        return init_tokenizer_from_config

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def decode_tokens(
    tokenizer: TokenizerLike,
    token_ids: list[int],
    *,
    skip_special_tokens: bool | None = None,
) -> str:
    """
    Backend-agnostic equivalent of HF's
    `tokenizer.decode(token_ids, ...)`.

    `skip_special_tokens=None` means to use the backend's default
    settings.
    """
    kw_args: dict[str, Any] = {}

    if skip_special_tokens is not None:
        kw_args["skip_special_tokens"] = skip_special_tokens

    return tokenizer.decode(token_ids, **kw_args)


def encode_tokens(
    tokenizer: TokenizerLike,
    text: str,
    *,
    truncation: bool | None = None,
    max_length: int | None = None,
    add_special_tokens: bool | None = None,
) -> list[int]:
    """
    Backend-agnostic equivalent of HF's
    `tokenizer.encode(text, ...)`.

    `add_special_tokens=None` means to use the backend's default
    settings.
    """

    kw_args: dict[str, Any] = {}
    if max_length is not None:
        kw_args["max_length"] = max_length

    if truncation is not None:
        kw_args["truncation"] = truncation

    if add_special_tokens is not None:
        kw_args["add_special_tokens"] = add_special_tokens

    return tokenizer.encode(text, **kw_args)
