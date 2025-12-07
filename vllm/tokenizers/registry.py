# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

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
from vllm.utils.import_utils import resolve_obj_by_qualname

from .protocol import TokenizerLike

if TYPE_CHECKING:
    from vllm.config import RendererConfig

logger = init_logger(__name__)

_T = TypeVar("_T", bound=TokenizerLike)


class TokenizerRegistry:
    # Tokenizer name ->  (tokenizer module, tokenizer class)
    REGISTRY: dict[str, tuple[str, str]] = {
        "deepseekv32": ("vllm.tokenizers.deepseekv32", "DeepseekV32Tokenizer"),
        "hf": ("vllm.tokenizers.hf", "CachedHfTokenizer"),
        "mistral": ("vllm.tokenizers.mistral", "MistralTokenizer"),
    }

    @staticmethod
    def register(tokenizer_mode: str, module: str, class_name: str) -> None:
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
    def init_tokenizer(tokenizer_mode: str, *args, **kwargs) -> TokenizerLike:
        if tokenizer_mode not in TokenizerRegistry.REGISTRY:
            raise ValueError(f"No tokenizer registered for {tokenizer_mode=!r}.")

        module, class_name = TokenizerRegistry.REGISTRY[tokenizer_mode]
        logger.debug_once(f"Loading {class_name} for {tokenizer_mode=!r}")

        cls_: type[TokenizerLike] = resolve_obj_by_qualname(f"{module}.{class_name}")
        return cls_.from_pretrained(*args, **kwargs)


def get_tokenizer(
    tokenizer_cls: type[_T],
    tokenizer_name: str | Path,
    *args,
    trust_remote_code: bool = False,
    revision: str | None = None,
    download_dir: str | None = None,
    **kwargs,
) -> _T:
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

    tokenizer_args = (tokenizer_name, *args)
    tokenizer_kwargs = dict[str, Any](
        trust_remote_code=trust_remote_code,
        revision=revision,
        download_dir=download_dir,
        **kwargs,
    )

    tokenizer = tokenizer_cls.from_pretrained(*tokenizer_args, **tokenizer_kwargs)
    if not tokenizer.is_fast:
        logger.warning(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )

    return tokenizer  # type: ignore


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
    if renderer_config.skip_tokenizer_init:
        return None

    runner_type = renderer_config.model_config.runner_type
    if runner_type == "generate" or runner_type == "draft":
        truncation_side = "left"
    elif runner_type == "pooling":
        truncation_side = "right"
    else:
        assert_never(runner_type)

    return cached_tokenizer_from_config(
        renderer_config,
        truncation_side=truncation_side,
    )
