# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache
from pathlib import Path
from threading import Lock

from vllm import envs

_modelscope_init_lock = Lock()


def maybe_init_modelscope(model: str | Path) -> bool:
    """Initialize ModelScope only for a remote ModelScope resource."""
    if not envs.VLLM_USE_MODELSCOPE or Path(model).exists():
        return False
    with _modelscope_init_lock:
        _patch_modelscope_hub()
    return True


@cache
def _patch_modelscope_hub() -> None:
    try:
        import modelscope
        from packaging import version

        # patch_hub begins from modelscope>=1.18.1
        if version.parse(modelscope.__version__) <= version.parse("1.18.0"):
            raise ImportError(
                "Using vLLM with ModelScope needs modelscope>=1.18.1, please "
                "install by `pip install modelscope -U`"
            )
        from modelscope.utils.hf_util import patch_hub

        patch_hub()
    except ImportError as err:
        raise ImportError(
            "Please install modelscope>=1.18.1 via "
            "`pip install modelscope>=1.18.1` to use ModelScope."
        ) from err
