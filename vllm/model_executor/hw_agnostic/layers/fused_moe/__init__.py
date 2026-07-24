# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FusedMoE Triton path."""

from contextlib import contextmanager
from typing import Any

_config: dict[str, Any] | None = None


@contextmanager
def override_config(config):
    """Override the autotune config used by the Triton GEMM kernel."""
    global _config
    old_config = _config
    _config = config
    try:
        yield
    finally:
        _config = old_config


def get_config() -> dict[str, Any] | None:
    return _config
