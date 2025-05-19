# SPDX-License-Identifier: Apache-2.0

import contextlib
import copy
import os
import warnings
from functools import lru_cache
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import huggingface_hub
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from vllm.envs import VLLM_USE_MODELSCOPE
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.tokenizer_base import (TokenizerBase,
                                                    TokenizerRegistry)
from vllm.transformers_utils.tokenizers import MistralTokenizer
from vllm.transformers_utils.utils import check_gguf_file
from vllm.utils import make_async

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

AnyTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast,
                     TokenizerBase]


def decode_tokens(
    tokenizer: AnyTokenizer,
    token_ids: list[int],
    *,
    skip_special_tokens: Optional[bool] = None,
) -> str:
    """
    Backend-agnostic equivalent of HF's
    `tokenizer.decode(token_ids, ...)`.

    `skip_special_tokens=None` means to use the backend's default
    settings.
    """
    if skip_special_tokens is not None:
        return tokenizer.decode(token_ids,
                                skip_special_tokens=skip_special_tokens)

    return tokenizer.decode(token_ids)


def encode_tokens(
    tokenizer: AnyTokenizer,
    text: str,
    *,
    truncation: Optional[bool] = None,
    max_length: Optional[int] = None,
    add_special_tokens: Optional[bool] = None,
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
    tokenizer_all_special_tokens_extended = (
        tokenizer.all_special_tokens_extended)
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
            return get_cached_tokenizer, (tokenizer, )

    CachedTokenizer.__name__ = f"Cached{tokenizer.__class__.__name__}"

    cached_tokenizer.__class__ = CachedTokenizer
    return cached_tokenizer


def patch_padding_side(tokenizer: PreTrainedTokenizer) -> None:
    """Patch _pad method to accept `padding_side` for older tokenizers."""
    orig_pad = tokenizer._pad

    def _pad(
        self: PreTrainedTokenizer,
        *args,
        padding_side: Optional[str] = None,
        **kwargs,
    ):
        if padding_side is not None and padding_side != self.padding_side:
            msg = ("`padding_side` argument is not supported by "
                   f"{type(tokenizer).__name__} and will be ignored.")
            warnings.warn(msg, stacklevel=2)

        return orig_pad(*args, **kwargs)

    tokenizer._pad = MethodType(_pad, tokenizer)


def get_tokenizer(
    tokenizer_name: Union[str, Path],
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    download_dir: Optional[str] = None,
    **kwargs,
) -> AnyTokenizer:
    """Gets a tokenizer for the given model name via HuggingFace or ModelScope.
    """
    if VLLM_USE_MODELSCOPE:
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
                    ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"])
                tokenizer_name = tokenizer_path

    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError(
                "Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    if "truncation_side" not in kwargs:
        kwargs["truncation_side"] = "left"

    # Separate model folder from file path for GGUF models
    is_gguf = check_gguf_file(tokenizer_name)
    if is_gguf:
        kwargs["gguf_file"] = Path(tokenizer_name).name
        tokenizer_name = Path(tokenizer_name).parent

    # if tokenizer is from official mistral org
    is_from_mistral_org = str(tokenizer_name).split("/")[0] == "mistralai"
    if is_from_mistral_org and tokenizer_mode != "mistral":
        warnings.warn(
            'It is strongly recommended to run mistral models with '
            '`--tokenizer-mode "mistral"` to ensure correct '
            'encoding and decoding.',
            FutureWarning,
            stacklevel=2)

    tokenizer: AnyTokenizer
    if tokenizer_mode == "mistral":
        tokenizer = MistralTokenizer.from_pretrained(str(tokenizer_name),
                                                     revision=revision)
    elif tokenizer_mode == "custom":
        tokenizer = TokenizerRegistry.get_tokenizer(str(tokenizer_name),
                                                    *args,
                                                    revision=revision,
                                                    download_dir=download_dir,
                                                    **kwargs)
    else:
        try:
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
                    or "requires you to execute the tokenizer file" in str(e)):
                err_msg = ("Failed to load the tokenizer. If the tokenizer "
                           "is a custom tokenizer not yet available in the "
                           "HuggingFace transformers library, consider "
                           "setting `trust_remote_code=True` in LLM or using "
                           "the `--trust-remote-code` flag in the CLI.")
                raise RuntimeError(err_msg) from e
            else:
                raise e

        # NOTE: We can remove this after https://github.com/THUDM/ChatGLM3/issues/1324
        if type(tokenizer).__name__ in ("ChatGLMTokenizer",
                                        "ChatGLM4Tokenizer"):
            assert isinstance(tokenizer, PreTrainedTokenizer)
            patch_padding_side(tokenizer)

        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            logger.warning(
                "Using a slow tokenizer. This might cause a significant "
                "slowdown. Consider using a fast tokenizer instead.")
        tokenizer = get_cached_tokenizer(tokenizer)

    return tokenizer


cached_get_tokenizer = lru_cache(get_tokenizer)


def cached_tokenizer_from_config(
    model_config: "ModelConfig",
    **kwargs: Any,
):
    return cached_get_tokenizer(
        model_config.tokenizer,
        tokenizer_mode=model_config.tokenizer_mode,
        tokenizer_revision=model_config.tokenizer_revision,
        trust_remote_code=model_config.trust_remote_code,
        **kwargs,
    )


def get_lora_tokenizer(lora_request: LoRARequest, *args,
                       **kwargs) -> Optional[AnyTokenizer]:
    if lora_request is None:
        return None
    try:
        tokenizer = get_tokenizer(lora_request.lora_path, *args, **kwargs)
    except Exception as e:
        # No tokenizer was found in the LoRA folder,
        # use base model tokenizer
        logger.warning(
            "No tokenizer found in %s, using base model tokenizer instead. "
            "(Exception: %s)", lora_request.lora_path, e)
        tokenizer = None
    return tokenizer


get_lora_tokenizer_async = make_async(get_lora_tokenizer)
