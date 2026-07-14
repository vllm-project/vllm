# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache
import importlib.util
import warnings

from packaging import version

import vllm.envs as envs

_MODELSCOPE_FALLBACK_WARNED = False


@cache
def modelscope_is_available() -> bool:
    if importlib.util.find_spec("modelscope") is None:
        return False

    try:
        import modelscope
    except Exception:
        return False

    return version.parse(getattr(modelscope, "__version__", "0")) > version.parse(
        "1.18.0"
    )


def should_use_modelscope() -> bool:
    return envs.VLLM_USE_MODELSCOPE and modelscope_is_available()


def configure_modelscope_runtime() -> None:
    # Keep ModelScope runtime configuration opt-in at the call site.
    return None


def warn_modelscope_fallback(context: str) -> None:
    global _MODELSCOPE_FALLBACK_WARNED
    if (
        envs.VLLM_USE_MODELSCOPE
        and not modelscope_is_available()
        and not _MODELSCOPE_FALLBACK_WARNED
    ):
        warnings.warn(
            f"{context}: VLLM_USE_MODELSCOPE=True but modelscope is unavailable "
            "or unsupported; falling back to Hugging Face Hub."
        )
        _MODELSCOPE_FALLBACK_WARNED = True
