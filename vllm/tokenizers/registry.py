# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import huggingface_hub
from typing_extensions import TypeVar, assert_never, deprecated

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
    from vllm.config.model import ModelConfig, RunnerType

logger = init_logger(__name__)


_VLLM_TOKENIZERS = {
    "deepseek_v32": ("deepseek_v32", "DeepseekV32Tokenizer"),
    "hf": ("hf", "CachedHfTokenizer"),
    "mistral": ("mistral", "MistralTokenizer"),
}


@dataclass
class _TokenizerRegistry:
    # Tokenizer mode ->  (tokenizer module, tokenizer class)
    tokenizers: dict[str, tuple[str, str]] = field(default_factory=dict)

    def register(self, tokenizer_mode: str, module: str, class_name: str) -> None:
        if tokenizer_mode in self.tokenizers:
            logger.warning(
                "%s.%s is already registered for tokenizer_mode=%r. "
                "It is overwritten by the new one.",
                module,
                class_name,
                tokenizer_mode,
            )

        self.tokenizers[tokenizer_mode] = (module, class_name)

        return None

    def load_tokenizer_cls(self, tokenizer_mode: str) -> type[TokenizerLike]:
        if tokenizer_mode not in self.tokenizers:
            raise ValueError(f"No tokenizer registered for {tokenizer_mode=!r}.")

        module, class_name = self.tokenizers[tokenizer_mode]
        logger.debug_once(f"Loading {class_name} for {tokenizer_mode=!r}")

        return resolve_obj_by_qualname(f"{module}.{class_name}")

    def load_tokenizer(self, tokenizer_mode: str, *args, **kwargs) -> TokenizerLike:
        tokenizer_cls = self.load_tokenizer_cls(tokenizer_mode)
        return tokenizer_cls.from_pretrained(*args, **kwargs)


TokenizerRegistry = _TokenizerRegistry(
    {
        mode: (f"vllm.tokenizers.{mod_relname}", cls_name)
        for mode, (mod_relname, cls_name) in _VLLM_TOKENIZERS.items()
    }
)


def resolve_tokenizer_args(
    tokenizer_name: str | Path,
    *args,
    runner_type: "RunnerType" = "generate",
    tokenizer_mode: str = "auto",
    **kwargs,
):
    revision: str | None = kwargs.get("revision")
    download_dir: str | None = kwargs.get("download_dir")

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

    if "truncation_side" not in kwargs:
        if runner_type == "generate" or runner_type == "draft":
            kwargs["truncation_side"] = "left"
        elif runner_type == "pooling":
            kwargs["truncation_side"] = "right"
        else:
            assert_never(runner_type)

    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")

        tokenizer_mode = "hf"
        kwargs["use_fast"] = False

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

    return tokenizer_mode, tokenizer_name, args, kwargs


cached_resolve_tokenizer_args = lru_cache(resolve_tokenizer_args)


def tokenizer_args_from_config(config: "ModelConfig", **kwargs):
    return cached_resolve_tokenizer_args(
        config.tokenizer,
        runner_type=config.runner_type,
        tokenizer_mode=config.tokenizer_mode,
        revision=config.tokenizer_revision,
        trust_remote_code=config.trust_remote_code,
        **kwargs,
    )


_T = TypeVar("_T", bound=TokenizerLike, default=TokenizerLike)


def get_tokenizer(
    tokenizer_name: str | Path,
    *args,
    tokenizer_cls: type[_T] = TokenizerLike,  # type: ignore[assignment]
    trust_remote_code: bool = False,
    revision: str | None = None,
    download_dir: str | None = None,
    **kwargs,
) -> _T:
    """Gets a tokenizer for the given model name via HuggingFace or ModelScope."""
    tokenizer_mode, tokenizer_name, args, kwargs = cached_resolve_tokenizer_args(
        tokenizer_name,
        *args,
        trust_remote_code=trust_remote_code,
        revision=revision,
        download_dir=download_dir,
        **kwargs,
    )

    if tokenizer_cls == TokenizerLike:
        tokenizer_cls_ = TokenizerRegistry.load_tokenizer_cls(tokenizer_mode)
    else:
        tokenizer_cls_ = tokenizer_cls

    tokenizer = tokenizer_cls_.from_pretrained(tokenizer_name, *args, **kwargs)
    if not tokenizer.is_fast:
        logger.warning(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )

    return tokenizer  # type: ignore


cached_get_tokenizer = lru_cache(get_tokenizer)


def cached_tokenizer_from_config(model_config: "ModelConfig", **kwargs):
    if model_config.skip_tokenizer_init:
        return None

    return cached_get_tokenizer(
        model_config.tokenizer,
        runner_type=model_config.runner_type,
        tokenizer_mode=model_config.tokenizer_mode,
        revision=model_config.tokenizer_revision,
        trust_remote_code=model_config.trust_remote_code,
        **kwargs,
    )


@deprecated(
    "Renamed to `cached_tokenizer_from_config`. The old name will be removed in v0.14."
)
def init_tokenizer_from_config(model_config: "ModelConfig"):
    return cached_tokenizer_from_config(model_config)
