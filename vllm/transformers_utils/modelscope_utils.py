# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache
import importlib.util
import warnings

import vllm.envs as envs

_MODELSCOPE_FALLBACK_WARNED = False


@cache
def modelscope_is_available() -> bool:
    return importlib.util.find_spec("modelscope") is not None


def should_use_modelscope() -> bool:
    return envs.VLLM_USE_MODELSCOPE and modelscope_is_available()


def warn_modelscope_fallback(context: str) -> None:
    global _MODELSCOPE_FALLBACK_WARNED
    if (
        envs.VLLM_USE_MODELSCOPE
        and not modelscope_is_available()
        and not _MODELSCOPE_FALLBACK_WARNED
    ):
        warnings.warn(
            f"{context}: VLLM_USE_MODELSCOPE=True but modelscope is not "
            "installed; falling back to Hugging Face Hub."
        )
        _MODELSCOPE_FALLBACK_WARNED = True
