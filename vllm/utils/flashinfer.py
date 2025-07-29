# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for FlashInfer API changes.

Users of vLLM should always import **only** these wrappers.
"""
from __future__ import annotations

import contextlib
import functools
import importlib
import importlib.util
from typing import Any, Callable, NoReturn

from vllm.logger import init_logger

logger = init_logger(__name__)


@functools.cache
def has_flashinfer() -> bool:
    """Return ``True`` if FlashInfer is available."""
    # Use find_spec to check if the module exists without importing it
    # This avoids potential CUDA initialization side effects
    return importlib.util.find_spec("flashinfer") is not None


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


# General lazy import wrapper
def _lazy_import_wrapper(module_name: str,
                         attr_name: str,
                         fallback_fn: Callable[..., Any] = _missing):
    """Create a lazy import wrapper for a specific function."""

    @functools.cache
    def _get_impl():
        if not has_flashinfer():
            return None
        mod = _get_submodule(module_name)
        return getattr(mod, attr_name, None) if mod else None

    def wrapper(*args, **kwargs):
        impl = _get_impl()
        if impl is None:
            return fallback_fn(*args, **kwargs)
        return impl(*args, **kwargs)

    return wrapper


# Create lazy wrappers for each function
flashinfer_trtllm_fp8_block_scale_moe = _lazy_import_wrapper(
    "flashinfer.fused_moe", "trtllm_fp8_block_scale_moe")
flashinfer_cutlass_fused_moe = _lazy_import_wrapper("flashinfer.fused_moe",
                                                    "cutlass_fused_moe")
fp4_quantize = _lazy_import_wrapper("flashinfer", "fp4_quantize")
nvfp4_block_scale_interleave = _lazy_import_wrapper(
    "flashinfer", "nvfp4_block_scale_interleave")

# Special case for autotune since it returns a context manager
autotune = _lazy_import_wrapper(
    "flashinfer.autotuner",
    "autotune",
    fallback_fn=lambda *args, **kwargs: contextlib.nullcontext())


@functools.cache
def has_flashinfer_moe() -> bool:
    """Return ``True`` if FlashInfer MoE module is available."""
    return has_flashinfer() and importlib.util.find_spec(
        "flashinfer.fused_moe") is not None


@functools.cache
def has_flashinfer_cutlass_fused_moe() -> bool:
    """Return ``True`` if FlashInfer CUTLASS fused MoE is available."""
    if not has_flashinfer_moe():
        return False

    # Check if all required functions are available
    required_functions = [
        ("flashinfer.fused_moe", "cutlass_fused_moe"),
        ("flashinfer", "fp4_quantize"),
        ("flashinfer", "nvfp4_block_scale_interleave"),
    ]

    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


__all__ = [
    "has_flashinfer",
    "flashinfer_trtllm_fp8_block_scale_moe",
    "flashinfer_cutlass_fused_moe",
    "fp4_quantize",
    "nvfp4_block_scale_interleave",
    "autotune",
    "has_flashinfer_moe",
    "has_flashinfer_cutlass_fused_moe",
]
