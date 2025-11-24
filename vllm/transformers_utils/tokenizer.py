# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import copy
import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import huggingface_hub
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing_extensions import assert_never

from vllm import envs
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    get_sentence_transformer_tokenizer_config,
    list_filtered_repo_files,
)
from vllm.transformers_utils.tokenizers import MistralTokenizer
from vllm.transformers_utils.utils import check_gguf_file

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.transformers_utils.tokenizer_base import TokenizerBase
else:
    ModelConfig = Any
    TokenizerBase = Any

logger = init_logger(__name__)

AnyTokenizer: TypeAlias = PreTrainedTokenizer | PreTrainedTokenizerFast | TokenizerBase


def decode_tokens(
    tokenizer: AnyTokenizer,
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
    if skip_special_tokens is not None:
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    return tokenizer.decode(token_ids)


def encode_tokens(
    tokenizer: AnyTokenizer,
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


def get_cached_tokenizer(tokenizer: AnyTokenizer) -> AnyTokenizer:
    """
    By default, transformers will recompute multiple tokenizer properties
    each time they are called, leading to a significant slowdown.
    This proxy caches these properties for faster access.
    """
    cached_tokenizer = copy.copy(tokenizer)

    tokenizer_all_special_ids = tokenizer.all_special_ids
    tokenizer_all_special_tokens = tokenizer.all_special_tokens
    tokenizer_all_special_tokens_extended = tokenizer.all_special_tokens_extended
    tokenizer_vocab = tokenizer.get_vocab()
    tokenizer_len = len(tokenizer)

    max_token_id = max(tokenizer_vocab.values())
    # Some tokenizers (e.g., QwenTokenizer) have special tokens that
    # are added and included in the implementation of the vocab_size
    # property, but not in get_vocab(); if there is an implementation
    # of vocab size, we should take the greater value.
    if hasattr(tokenizer, "vocab_size"):
        with contextlib.suppress(NotImplementedError):
            max_token_id = max(max_token_id, tokenizer.vocab_size)

    class CachedTokenizer(tokenizer.__class__):  # type: ignore
        @property
        def all_special_ids(self) -> list[int]:
            return tokenizer_all_special_ids

        @property
        def all_special_tokens(self) -> list[str]:
            return tokenizer_all_special_tokens

        @property
        def all_special_tokens_extended(self) -> list[str]:
            return tokenizer_all_special_tokens_extended

        @property
        def max_token_id(self) -> int:
            return max_token_id

        def get_vocab(self) -> dict[str, int]:
            return tokenizer_vocab

        def __len__(self) -> int:
            return tokenizer_len

        def __reduce__(self):
            return get_cached_tokenizer, (tokenizer,)

    CachedTokenizer.__name__ = f"Cached{tokenizer.__class__.__name__}"

    cached_tokenizer.__class__ = CachedTokenizer
    return cached_tokenizer


def get_tokenizer(
    tokenizer_name: str | Path,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    revision: str | None = None,
    download_dir: str | None = None,
    **kwargs,
) -> AnyTokenizer:
    """Gets a tokenizer for the given model name via HuggingFace or ModelScope."""
    if envs.VLLM_USE_MODELSCOPE:
        # download model from ModelScope hub,
        # lazy import so that modelscope is not required for normal use.
        # pylint: disable=C.
        from modelscope.hub.snapshot_download import snapshot_download

        # avoid circuit import
        from vllm.model_executor.model_loader.weight_utils import get_lock

        # Only set the tokenizer here, model will be downloaded on the workers.
        if not os.path.exists(tokenizer_name):
            # Use file lock to prevent multiple processes from
            # downloading the same file at the same time.
            with get_lock(tokenizer_name, download_dir):
                tokenizer_path = snapshot_download(
                    model_id=tokenizer_name,
                    cache_dir=download_dir,
                    revision=revision,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    # Ignore weights - we only need the tokenizer.
                    ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"],
                )
                tokenizer_name = tokenizer_path

    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    if "truncation_side" not in kwargs:
        kwargs["truncation_side"] = "left"

    # Separate model folder from file path for GGUF models
    is_gguf = check_gguf_file(tokenizer_name)
    if is_gguf:
        kwargs["gguf_file"] = Path(tokenizer_name).name
        tokenizer_name = Path(tokenizer_name).parent

    # if `tokenizer_mode` == "auto", check if tokenizer can be loaded via Mistral format
    # first to use official Mistral tokenizer if possible.
    mistral_common_installed = importlib.util.find_spec("mistral_common") is not None
    if tokenizer_mode == "auto" and mistral_common_installed:
        allow_patterns = ["tekken.json", "tokenizer.model.v*"]
        files_list = list_filtered_repo_files(
            model_name_or_path=str(tokenizer_name),
            allow_patterns=allow_patterns,
            revision=revision,
        )
        if len(files_list) > 0:
            tokenizer_mode = "mistral"

    tokenizer: AnyTokenizer
    if tokenizer_mode == "mistral":
        logger.debug_once(f"Loading MistralTokenizer from {tokenizer_name}")
        tokenizer = MistralTokenizer.from_pretrained(
            str(tokenizer_name), revision=revision
        )
    elif tokenizer_mode == "custom":
        from vllm.transformers_utils.tokenizer_base import TokenizerRegistry

        logger.debug_once(f"Loading CustomTokenizer from {tokenizer_name}")
        tokenizer = TokenizerRegistry.get_tokenizer(
            str(tokenizer_name),
            *args,
            revision=revision,
            download_dir=download_dir,
            **kwargs,
        )
    else:
        try:
            logger.debug_once(f"Loading AutoTokenizer from {tokenizer_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                **kwargs,
            )
        except ValueError as e:
            # If the error pertains to the tokenizer class not existing or not
            # currently being imported,
            # suggest using the --trust-remote-code flag.
            if not trust_remote_code and (
                "does not exist or is not currently imported." in str(e)
                or "requires you to execute the tokenizer file" in str(e)
            ):
                err_msg = (
                    "Failed to load the tokenizer. If the tokenizer "
                    "is a custom tokenizer not yet available in the "
                    "HuggingFace transformers library, consider "
                    "setting `trust_remote_code=True` in LLM or using "
                    "the `--trust-remote-code` flag in the CLI."
                )
                raise RuntimeError(err_msg) from e
            else:
                raise e

        # The special_tokens in tokenizer should also be
        # controlled by do_lower_case in encoder_config
        encoder_config = get_sentence_transformer_tokenizer_config(
            tokenizer_name, revision
        )
        if isinstance(encoder_config, dict) and encoder_config.get(
            "do_lower_case", False
        ):
            special_tokens_map = {
                k: v.lower() for k, v in tokenizer.special_tokens_map.items()
            }
            tokenizer.add_special_tokens(special_tokens_map)

        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            logger.warning(
                "Using a slow tokenizer. This might cause a significant "
                "slowdown. Consider using a fast tokenizer instead."
            )
        tokenizer = get_cached_tokenizer(tokenizer)

    return tokenizer


cached_get_tokenizer = lru_cache(get_tokenizer)


def cached_tokenizer_from_config(
    model_config: ModelConfig,
    **kwargs: Any,
):
    return cached_get_tokenizer(
        model_config.tokenizer,
        tokenizer_mode=model_config.tokenizer_mode,
        revision=model_config.tokenizer_revision,
        trust_remote_code=model_config.trust_remote_code,
        **kwargs,
    )


def init_tokenizer_from_configs(model_config: ModelConfig):
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
