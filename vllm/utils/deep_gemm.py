# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for DeepGEMM API changes.

Users of vLLM should always import **only** these wrappers.
"""
from __future__ import annotations

import functools
import importlib
from typing import Any, Callable, NoReturn

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.utils import has_deep_gemm


@functools.cache
def is_blackwell_deep_gemm_used() -> bool:
    """Return ``True`` if vLLM is configured to use DeepGEMM on a
    Blackwell-class GPU.
    """
    if not (envs.VLLM_USE_DEEP_GEMM and has_deep_gemm()):
        return False

    _lazy_init()
    if _per_block_cast_impl is None:
        return False

    return (current_platform.is_cuda()
            and current_platform.is_device_capability(100))


def _missing(*_: Any, **__: Any) -> NoReturn:
    """Placeholder for unavailable DeepGEMM backend."""
    raise RuntimeError(
        "DeepGEMM backend is not available. Please install the `deep_gemm` "
        "package to enable FP8 kernels.")


def _resolve_symbol(module, new: str, old: str) -> Callable[..., Any] | None:
    """Return the *new* symbol if it exists, otherwise the *old* one."""
    if hasattr(module, new):
        return getattr(module, new)
    if hasattr(module, old):
        return getattr(module, old)
    return None


_fp8_gemm_nt_impl: Callable[..., Any] | None = None
_grouped_impl: Callable[..., Any] | None = None
_grouped_masked_impl: Callable[..., Any] | None = None
_per_block_cast_impl: Callable[..., Any] | None = None


def _lazy_init() -> None:
    """Import deep_gemm and resolve symbols on first use."""
    global _fp8_gemm_nt_impl, _grouped_impl, _grouped_masked_impl, \
        _per_block_cast_impl

    # fast path
    if (_fp8_gemm_nt_impl is not None or _grouped_impl is not None
            or _grouped_masked_impl is not None
            or _per_block_cast_impl is not None):
        return

    if not has_deep_gemm():
        return

    _dg = importlib.import_module("deep_gemm")

    _fp8_gemm_nt_impl = _resolve_symbol(_dg, "fp8_gemm_nt",
                                        "gemm_fp8_fp8_bf16_nt")
    _grouped_impl = _resolve_symbol(
        _dg, "m_grouped_fp8_gemm_nt_contiguous",
        "m_grouped_gemm_fp8_fp8_bf16_nt_contiguous")
    _grouped_masked_impl = _resolve_symbol(
        _dg, "fp8_m_grouped_gemm_nt_masked",
        "m_grouped_gemm_fp8_fp8_bf16_nt_masked")
    # Try to get per_token_cast_to_fp8 from DeepGEMM math utils.
    try:
        _math_mod = importlib.import_module(
            "deep_gemm.utils.math")  # type: ignore
        _per_block_cast_impl = getattr(_math_mod, "per_block_cast_to_fp8",
                                       None)
    except ModuleNotFoundError:
        _per_block_cast_impl = None


def fp8_gemm_nt(*args, **kwargs):
    _lazy_init()
    if _fp8_gemm_nt_impl is None:
        return _missing(*args, **kwargs)
    return _fp8_gemm_nt_impl(*args, **kwargs)


def m_grouped_fp8_gemm_nt_contiguous(*args, **kwargs):
    _lazy_init()
    if _grouped_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_impl(*args, **kwargs)


def fp8_m_grouped_gemm_nt_masked(*args, **kwargs):
    _lazy_init()
    if _grouped_masked_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_masked_impl(*args, **kwargs)


def per_block_cast_to_fp8(x, *args, **kwargs):
    _lazy_init()
    if _per_block_cast_impl is not None and is_blackwell_deep_gemm_used():
        return _per_block_cast_impl(x, use_ue8m0=True)
    # TODO: refactor the `per_block_cast_to_fp8` from tests to vllm utils
    from tests.kernels.quant_utils import per_block_cast_to_fp8 as _pbcf
    return _pbcf(x, *args, **kwargs)


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    """Return a global difference metric for unit tests.

    DeepGEMM kernels on Blackwell/B200 currently exhibit noticeable per-element
    error, causing ``torch.testing.assert_close`` to fail.  Instead of checking
    every element, we compute a cosine-style similarity over the whole tensor
    and report ``1 - sim``.  Once kernel accuracy improves this helper can be
    removed.
    """

    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


__all__ = [
    "calc_diff",
    "fp8_gemm_nt",
    "m_grouped_fp8_gemm_nt_contiguous",
    "fp8_m_grouped_gemm_nt_masked",
    "per_block_cast_to_fp8",
    "is_blackwell_deep_gemm_used",
]
