# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for FlashInfer API changes.

Users of vLLM should always import **only** these wrappers.
"""
from __future__ import annotations

import contextlib
import functools
import importlib
from typing import Any, Callable, NoReturn

from vllm.logger import init_logger

logger = init_logger(__name__)


@functools.cache
def has_flashinfer() -> bool:
    """Return ``True`` if FlashInfer is available."""
    try:
        import flashinfer  # noqa: F401
        return True
    except ImportError:
        return False


def _missing(*_: Any, **__: Any) -> NoReturn:
    """Placeholder for unavailable FlashInfer backend."""
    raise RuntimeError(
        "FlashInfer backend is not available. Please install the package "
        "to enable FlashInfer kernels: "
        "https://github.com/flashinfer-ai/flashinfer")


def _get_submodule(module_name: str) -> Any | None:
    """Safely import a submodule and return it, or None if not available."""
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None


# Initialize FlashInfer components
if not has_flashinfer():
    _cutlass_fused_moe_impl: Callable[..., Any] | None = None
    _fp4_quantize_impl: Callable[..., Any] | None = None
    _fp4_swizzle_blockscale_impl: Callable[..., Any] | None = None
    _autotune_impl: Callable[..., Any] | None = None
else:
    # Import main flashinfer module
    _fi = importlib.import_module("flashinfer")  # type: ignore

    # Import fused_moe submodule
    _fused_moe_mod = _get_submodule("flashinfer.fused_moe")
    _cutlass_fused_moe_impl = getattr(_fused_moe_mod, "cutlass_fused_moe",
                                      None) if _fused_moe_mod else None

    # Import fp4_quant functions
    _fp4_quantize_impl = getattr(_fi, "fp4_quantize", None) if _fi else None
    _fp4_swizzle_blockscale_impl = getattr(_fi, "fp4_swizzle_blockscale",
                                           None) if _fi else None

    # Import autotuner submodule
    _autotuner_mod = _get_submodule("flashinfer.autotuner")
    _autotune_impl = getattr(_autotuner_mod, "autotune",
                             None) if _autotuner_mod else None


@functools.cache
def has_flashinfer_cutlass_fused_moe() -> bool:
    """Return ``True`` if FlashInfer CUTLASS fused MoE is available."""
    return all([
        _cutlass_fused_moe_impl,
        _fp4_quantize_impl,
        _fp4_swizzle_blockscale_impl,
    ])


def flashinfer_cutlass_fused_moe(*args, **kwargs):
    """FlashInfer CUTLASS fused MoE kernel."""
    if _cutlass_fused_moe_impl is None:
        return _missing(*args, **kwargs)
    return _cutlass_fused_moe_impl(*args, **kwargs)


def fp4_quantize(*args, **kwargs):
    """FlashInfer FP4 quantization."""
    if _fp4_quantize_impl is None:
        return _missing(*args, **kwargs)
    return _fp4_quantize_impl(*args, **kwargs)


def fp4_swizzle_blockscale(*args, **kwargs):
    """FlashInfer FP4 swizzle blockscale."""
    if _fp4_swizzle_blockscale_impl is None:
        return _missing(*args, **kwargs)
    return _fp4_swizzle_blockscale_impl(*args, **kwargs)


def autotune(*args, **kwargs):
    """FlashInfer autotuner."""
    if _autotune_impl is None:
        # return a null context since autotune is a context manager
        return contextlib.nullcontext()
    return _autotune_impl(*args, **kwargs)


__all__ = [
    "has_flashinfer",
    "has_flashinfer_cutlass_fused_moe",
    "flashinfer_cutlass_fused_moe",
    "fp4_quantize",
    "fp4_swizzle_blockscale",
    "autotune",
]
