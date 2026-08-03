# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling kernels (NVIDIA).

``rmsnorm`` / ``sconv`` import eagerly. The SwiGLU kernels and the FA4
relative-attention wrapper are exposed lazily to keep this package's
import path lightweight.
"""

from typing import TYPE_CHECKING

from .norm import add_rmsnorm, rmsnorm
from .sconv import fused_sconv

_LAZY_EXPORTS = {
    "silu_and_mul_triton": "silu_and_mul",
    "sink_silu_mul_epilogue": "silu_and_mul",
    "inkling_fa4_rel_attention": "fa4_rel_attention",
}

if TYPE_CHECKING:
    from .fa4_rel_attention import inkling_fa4_rel_attention  # noqa: F401
    from .silu_and_mul import (  # noqa: F401
        silu_and_mul_triton,
        sink_silu_mul_epilogue,
    )

__all__ = [
    "add_rmsnorm",
    "rmsnorm",
    "fused_sconv",
    *sorted(_LAZY_EXPORTS),
]


def __getattr__(name: str):
    module = _LAZY_EXPORTS.get(name)
    if module is not None:
        import importlib

        mod = importlib.import_module(f".{module}", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
