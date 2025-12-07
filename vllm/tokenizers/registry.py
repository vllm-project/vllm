# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, overload

import huggingface_hub
from typing_extensions import assert_never

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.transformers_utils.gguf_utils import (
    check_gguf_file,
    get_gguf_file_path_from_hf,
    is_gguf,
    is_remote_gguf,
    split_remote_gguf,
)
from vllm.transformers_utils.repo_utils import list_filtered_repo_files
from vllm.utils.import_utils import resolve_obj_by_qualname

from .protocol import TokenizerLike

if TYPE_CHECKING:
    from vllm.config import RendererConfig

logger = init_logger(__name__)

_T = TypeVar("_T", bound=type[TokenizerLike])


class TokenizerRegistry:
    # Tokenizer name -> tokenizer_cls or (tokenizer module, tokenizer class)
    REGISTRY: dict[str, type[TokenizerLike] | tuple[str, str]] = {}

    # In-tree tokenizers
    @staticmethod
    @overload
    def register(tokenizer_mode: str) -> Callable[[_T], _T]: ...

    # OOT tokenizers
    @staticmethod
    @overload
    def register(tokenizer_mode: str, module: str, class_name: str) -> None: ...

    @staticmethod
    def register(
        tokenizer_mode: str,
        module: str | None = None,
        class_name: str | None = None,
    ) -> Callable[[_T], _T] | None:
        # In-tree tokenizers
        if module is None or class_name is None:

            def wrapper(tokenizer_cls: _T) -> _T:
                assert tokenizer_mode not in TokenizerRegistry.REGISTRY
                TokenizerRegistry.REGISTRY[tokenizer_mode] = tokenizer_cls

                return tokenizer_cls

            return wrapper

        # OOT tokenizers
        if tokenizer_mode in TokenizerRegistry.REGISTRY:
            logger.warning(
                "%s.%s is already registered for tokenizer_mode=%r. "
                "It is overwritten by the new one.",
                module,
                class_name,
                tokenizer_mode,
            )

        TokenizerRegistry.REGISTRY[tokenizer_mode] = (module, class_name)

        return None

    @staticmethod
    def get_tokenizer(tokenizer_mode: str, *args, **kwargs) -> "TokenizerLike":
        if tokenizer_mode not in TokenizerRegistry.REGISTRY:
            raise ValueError(f"No tokenizer registered for {tokenizer_mode=!r}.")

        item = TokenizerRegistry.REGISTRY[tokenizer_mode]
        if isinstance(item, type):
            return item.from_pretrained(*args, **kwargs)

        module, class_name = item
        logger.debug_once(f"Loading {class_name} for {tokenizer_mode=!r}")

        class_ = resolve_obj_by_qualname(f"{module}.{class_name}")
        return class_.from_pretrained(*args, **kwargs)


def get_tokenizer(
    tokenizer_name: str | Path,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    revision: str | None = None,
    download_dir: str | None = None,
    **kwargs,
) -> TokenizerLike:
    """Gets a tokenizer for the given model name via HuggingFace or ModelScope."""
    if envs.VLLM_USE_MODELSCOPE:
        # download model from ModelScope hub,
        # lazy import so that modelscope is not required for normal use.
        from modelscope.hub.snapshot_download import snapshot_download

        # avoid circular import
        from vllm.model_executor.model_loader.weight_utils import get_lock

        # Only set the tokenizer here, model will be downloaded on the workers.
        if not Path(tokenizer_name).exists():
            # Use file lock to prevent multiple processes from
            # downloading the same file at the same time.
            with get_lock(tokenizer_name, download_dir):
                tokenizer_path = snapshot_download(
                    model_id=str(tokenizer_name),
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

        tokenizer_mode = "hf"
        kwargs["use_fast"] = False

    if "truncation_side" not in kwargs:
        kwargs["truncation_side"] = "left"

    # Separate model folder from file path for GGUF models
    if is_gguf(tokenizer_name):
        if check_gguf_file(tokenizer_name):
            kwargs["gguf_file"] = Path(tokenizer_name).name
            tokenizer_name = Path(tokenizer_name).parent
        elif is_remote_gguf(tokenizer_name):
            tokenizer_name, quant_type = split_remote_gguf(tokenizer_name)
            # Get the HuggingFace Hub path for the GGUF file
            gguf_file = get_gguf_file_path_from_hf(
                tokenizer_name,
                quant_type,
                revision=revision,
            )
            kwargs["gguf_file"] = gguf_file

    # Try to use official Mistral tokenizer if possible
    if tokenizer_mode == "auto" and importlib.util.find_spec("mistral_common"):
        allow_patterns = ["tekken.json", "tokenizer.model.v*"]
        files_list = list_filtered_repo_files(
            model_name_or_path=str(tokenizer_name),
            allow_patterns=allow_patterns,
            revision=revision,
        )
        if len(files_list) > 0:
            tokenizer_mode = "mistral"

    # Fallback to HF tokenizer
    if tokenizer_mode == "auto":
        tokenizer_mode = "hf"

    tokenizer_args = (tokenizer_name, *args)
    tokenizer_kwargs = dict(
        trust_remote_code=trust_remote_code,
        revision=revision,
        download_dir=download_dir,
        **kwargs,
    )

    if tokenizer_mode == "custom":
        logger.warning_once(
            "TokenizerRegistry now uses `tokenizer_mode` as the registry key "
            "instead of `tokenizer_name`. "
            "Please update the definition of `.from_pretrained` in "
            "your custom tokenizer to accept `args=%s`, `kwargs=%s`. "
            "Then, you can pass `tokenizer_mode=%r` instead of "
            "`tokenizer_mode='custom'` when initializing vLLM.",
            tokenizer_args,
            str(tokenizer_kwargs),
            tokenizer_name,
        )

        tokenizer_mode = str(tokenizer_name)

    tokenizer = TokenizerRegistry.get_tokenizer(
        tokenizer_mode,
        *tokenizer_args,
        **tokenizer_kwargs,
    )
    if not tokenizer.is_fast:
        logger.warning(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )

    return tokenizer


cached_get_tokenizer = lru_cache(get_tokenizer)


def cached_tokenizer_from_config(renderer_config: "RendererConfig", **kwargs):
    return cached_get_tokenizer(
        renderer_config.tokenizer,
        tokenizer_mode=renderer_config.tokenizer_mode,
        revision=renderer_config.tokenizer_revision,
        trust_remote_code=renderer_config.trust_remote_code,
        **kwargs,
    )


def init_tokenizer_from_config(renderer_config: "RendererConfig"):
    runner_type = renderer_config.model_config.runner_type
    if runner_type == "generate" or runner_type == "draft":
        truncation_side = "left"
    elif runner_type == "pooling":
        truncation_side = "right"
    else:
        assert_never(runner_type)

    return get_tokenizer(
        renderer_config.tokenizer,
        tokenizer_mode=renderer_config.tokenizer_mode,
        trust_remote_code=renderer_config.trust_remote_code,
        revision=renderer_config.tokenizer_revision,
        truncation_side=truncation_side,
    )
