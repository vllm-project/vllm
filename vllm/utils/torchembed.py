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
    """Return `True` if the torchembed package is available."""
    return find_spec("torchembed") is not None


__all__ = [
    "is_torchembed_available",
]
