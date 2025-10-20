# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utilities for detecting optional dependencies."""

from __future__ import annotations

import importlib.util
from functools import cache


@cache
def _has_module(module_name: str) -> bool:
    """Return True if *module_name* can be found in the current environment.

    The result is cached so that subsequent queries for the same module incur
    no additional overhead.
    """
    return importlib.util.find_spec(module_name) is not None


def has_pplx() -> bool:
    """Whether the optional `pplx_kernels` package is available."""
    return _has_module("pplx_kernels")


def has_deep_ep() -> bool:
    """Whether the optional `deep_ep` package is available."""
    return _has_module("deep_ep")


def has_deep_gemm() -> bool:
    """Whether the optional `deep_gemm` package is available."""
    return _has_module("deep_gemm")


def has_triton_kernels() -> bool:
    """Whether the optional `triton_kernels` package is available."""
    return _has_module("triton_kernels")


def has_tilelang() -> bool:
    """Whether the optional `tilelang` package is available."""
    return _has_module("tilelang")
