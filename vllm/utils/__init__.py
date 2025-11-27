# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import uuid
import warnings
from typing import Any

import torch

from vllm.logger import init_logger

_DEPRECATED_MAPPINGS = {
    "cprofile": "profiling",
    "cprofile_context": "profiling",
    # Used by lm-eval
    "get_open_port": "network_utils",
}


def __getattr__(name: str) -> Any:  # noqa: D401 - short deprecation docstring
    """Module-level getattr to handle deprecated utilities."""
    if name in _DEPRECATED_MAPPINGS:
        submodule_name = _DEPRECATED_MAPPINGS[name]
        warnings.warn(
            f"vllm.utils.{name} is deprecated and will be removed in a future version. "
            f"Use vllm.utils.{submodule_name}.{name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        module = __import__(f"vllm.utils.{submodule_name}", fromlist=[submodule_name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # expose deprecated names in dir() for better UX/tab-completion
    return sorted(list(globals().keys()) + list(_DEPRECATED_MAPPINGS.keys()))


logger = init_logger(__name__)

# Constants related to forcing the attention backend selection

# String name of register which may be set in order to
# force auto-selection of attention backend by Attention
# wrapper
STR_BACKEND_ENV_VAR: str = "VLLM_ATTENTION_BACKEND"

# Possible string values of STR_BACKEND_ENV_VAR
# register, corresponding to possible backends
STR_FLASHINFER_ATTN_VAL: str = "FLASHINFER"
STR_FLASH_ATTN_VAL: str = "FLASH_ATTN"
STR_INVALID_VAL: str = "INVALID"

MASK_64_BITS = (1 << 64) - 1


def random_uuid() -> str:
    return f"{uuid.uuid4().int & MASK_64_BITS:016x}"  # 16 hex chars


def length_from_prompt_token_ids_or_embeds(
    prompt_token_ids: list[int] | None,
    prompt_embeds: torch.Tensor | None,
) -> int:
    """Calculate the request length (in number of tokens) give either
    prompt_token_ids or prompt_embeds.
    """
    prompt_token_len = None if prompt_token_ids is None else len(prompt_token_ids)
    prompt_embeds_len = None if prompt_embeds is None else len(prompt_embeds)

    if prompt_token_len is None:
        if prompt_embeds_len is None:
            raise ValueError("Neither prompt_token_ids nor prompt_embeds were defined.")
        return prompt_embeds_len
    else:
        if prompt_embeds_len is not None and prompt_embeds_len != prompt_token_len:
            raise ValueError(
                "Prompt token ids and prompt embeds had different lengths"
                f" prompt_token_ids={prompt_token_len}"
                f" prompt_embeds={prompt_embeds_len}"
            )
        return prompt_token_len
