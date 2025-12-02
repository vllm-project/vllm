# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from typing_extensions import assert_never

from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike, get_tokenizer

if TYPE_CHECKING:
    from vllm.config import ModelConfig


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
    if name == "get_cached_tokenizer":
        from vllm.tokenizers.hf import get_cached_tokenizer

        warnings.warn(
            "`vllm.transformers_utils.tokenizer.get_cached_tokenizer` "
            "has been moved to `vllm.tokenizers.hf.get_cached_tokenizer`. "
            "The old name will be removed in v0.13.",
            DeprecationWarning,
            stacklevel=2,
        )

        return get_cached_tokenizer

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


cached_get_tokenizer = lru_cache(get_tokenizer)


def cached_tokenizer_from_config(
    model_config: "ModelConfig",
    **kwargs: Any,
):
    return cached_get_tokenizer(
        model_config.tokenizer,
        tokenizer_mode=model_config.tokenizer_mode,
        revision=model_config.tokenizer_revision,
        trust_remote_code=model_config.trust_remote_code,
        **kwargs,
    )


def init_tokenizer_from_configs(model_config: "ModelConfig"):
    runner_type = model_config.runner_type
    if runner_type == "generate" or runner_type == "draft":
        truncation_side = "left"
    elif runner_type == "pooling":
        truncation_side = "right"
    else:
        assert_never(runner_type)

    return get_tokenizer(
        model_config.tokenizer,
        tokenizer_mode=model_config.tokenizer_mode,
        trust_remote_code=model_config.trust_remote_code,
        revision=model_config.tokenizer_revision,
        truncation_side=truncation_side,
    )
