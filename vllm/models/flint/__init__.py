# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flint model entry point."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import FlintForCausalLM

__all__ = ["FlintForCausalLM"]


def __getattr__(name: str):
    # Lazy so that resolving FlintConfig does not import CUDA kernels.
    if name in __all__:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
