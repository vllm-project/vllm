"""
Dynamic dispatcher for CPU extension operations.

This module provides a unified interface to torch.ops regardless of which
CPU extension is loaded (_C for AVX2 or _C_avx512 for AVX512).

The dispatcher detects which extension is available and routes calls
to the correct namespace.

Background:
-----------
The cpu-build-dispatcher branch builds two separate .so files:
- _C.so (AVX2 build) -> registers to torch.ops._C
- _C_avx512.so (AVX512 build) -> registers to torch.ops._C_avx512

At runtime, only ONE extension is loaded based on CPU detection.
This dispatcher routes calls to whichever extension is available.
"""

import torch
from functools import lru_cache
from typing import Any, Optional


@lru_cache(maxsize=1)
def _detect_cpu_extension() -> str:
    """
    Detect which CPU extension is loaded.

    Returns:
        "_C_avx512" if AVX512 extension loaded
        "_C" if AVX2 extension loaded (or CUDA build with _C)
        "" if no extension detected
    """
    # Check for AVX512 extension first (more specific)
    if hasattr(torch.ops, '_C_avx512'):
        # Verify it has actual CPU ops registered (not empty)
        if hasattr(torch.ops._C_avx512, 'silu_and_mul'):
            return "_C_avx512"

    # Check for AVX2/default extension
    if hasattr(torch.ops, '_C'):
        # For CPU builds, check for a CPU-specific op
        # For CUDA builds, _C will have CUDA ops
        if hasattr(torch.ops._C, 'silu_and_mul'):
            return "_C"

    # No CPU extension loaded
    return ""


@lru_cache(maxsize=1)
def get_ops():
    """
    Get the correct torch.ops module for CPU operations.

    Returns:
        torch.ops._C_avx512 or torch.ops._C depending on loaded extension

    Note:
        Falls back to torch.ops._C for CUDA builds where ops are registered there.
    """
    ext = _detect_cpu_extension()
    if ext == "_C_avx512":
        return torch.ops._C_avx512
    elif ext == "_C":
        return torch.ops._C
    else:
        # Fall back to _C (for CUDA builds or if detection fails)
        return torch.ops._C


@lru_cache(maxsize=1)
def get_utils():
    """
    Get the correct torch.ops utils module.

    Returns:
        torch.ops._C_avx512_utils or torch.ops._C_utils
    """
    ext = _detect_cpu_extension()
    if ext == "_C_avx512":
        if hasattr(torch.ops, '_C_avx512_utils'):
            return torch.ops._C_avx512_utils
    if hasattr(torch.ops, '_C_utils'):
        return torch.ops._C_utils
    # Return None if no utils module found (caller should handle)
    return None


@lru_cache(maxsize=1)
def get_cpu_ops():
    """
    Get the correct torch.ops CPU-specific module (_C_cpu or _C_avx512_cpu).

    Returns:
        The CPU ops module, or None if not available
    """
    ext = _detect_cpu_extension()
    if ext == "_C_avx512":
        if hasattr(torch.ops, '_C_avx512_cpu'):
            return torch.ops._C_avx512_cpu
    if hasattr(torch.ops, '_C_cpu'):
        return torch.ops._C_cpu
    return None


def has_op(op_name: str) -> bool:
    """
    Check if an operation is available in the loaded extension.

    Args:
        op_name: Name of the operation (e.g., "onednn_mm")

    Returns:
        True if operation exists, False otherwise
    """
    try:
        ops = get_ops()
        return hasattr(ops, op_name)
    except Exception:
        return False


def get_op(op_name: str) -> Optional[Any]:
    """
    Get an operation by name, or None if not available.

    Args:
        op_name: Name of the operation

    Returns:
        The operation callable, or None if not found
    """
    try:
        ops = get_ops()
        return getattr(ops, op_name, None)
    except Exception:
        return None


class _OpsProxy:
    """
    Proxy object that forwards attribute access to the correct ops module.

    Usage:
        from vllm._ops_dispatch import ops
        ops.silu_and_mul(out, x)  # Automatically routes to correct extension
    """

    def __getattr__(self, name: str) -> Any:
        ops = get_ops()
        return getattr(ops, name)


class _UtilsProxy:
    """
    Proxy object that forwards attribute access to the correct utils module.

    Usage:
        from vllm._ops_dispatch import utils
        utils.init_cpu_threads_env(cpu_ids)
    """

    def __getattr__(self, name: str) -> Any:
        utils_module = get_utils()
        if utils_module is None:
            raise RuntimeError("No CPU utils extension loaded")
        return getattr(utils_module, name)


class _CpuOpsProxy:
    """
    Proxy object that forwards attribute access to the correct CPU ops module.

    Usage:
        from vllm._ops_dispatch import cpu_ops
        cpu_ops.mla_decode_kvcache(...)
    """

    def __getattr__(self, name: str) -> Any:
        cpu_ops_module = get_cpu_ops()
        if cpu_ops_module is None:
            raise RuntimeError("No CPU ops extension loaded")
        return getattr(cpu_ops_module, name)


# Global proxy instances for convenient access
ops = _OpsProxy()
utils = _UtilsProxy()
cpu_ops = _CpuOpsProxy()


def clear_cache():
    """Clear cached extension detection (useful for testing)."""
    _detect_cpu_extension.cache_clear()
    get_ops.cache_clear()
    get_utils.cache_clear()
    get_cpu_ops.cache_clear()
