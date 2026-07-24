# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for the torchembed package.

Users of vLLM should always import **only** these wrappers.
"""

import functools
from importlib.util import find_spec

from vllm.logger import init_logger

logger = init_logger(__name__)


@functools.cache
def is_torchembed_available() -> bool:
    """Return True if torchembed+triton are installed AND VLLM_USE_TORCHEMBED=1.

    The torchembed Triton kernel is opt-in because in vLLM's packed-token
    inference layout the built-in CUDA kernel is typically faster (in-place,
    no format-conversion copies needed).  Set the env variable explicitly to
    benchmark or to use torchembed when the native C++ ops are unavailable.
    """
    from vllm import envs
    if not envs.VLLM_USE_TORCHEMBED:
        return False
    return (find_spec("torchembed") is not None
            and find_spec("triton") is not None)


__all__ = [
    "is_torchembed_available",
]
