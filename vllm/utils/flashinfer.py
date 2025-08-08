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
import os
from typing import Any, Callable, NoReturn, Optional

import requests

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# This is the storage path for the cubins, it can be replaced
# with a local path for testing.
# Referenced from https://github.com/flashinfer-ai/flashinfer/blob/0c9a92c3d9a7e043ab6f3f7b2273269caf6ab044/flashinfer/jit/cubin_loader.py#L35  # noqa: E501
FLASHINFER_CUBINS_REPOSITORY = os.environ.get(
    "FLASHINFER_CUBINS_REPOSITORY",
    "https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/",  # noqa: E501
)


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
flashinfer_trtllm_fp8_per_tensor_scale_moe = _lazy_import_wrapper(
    "flashinfer.fused_moe", "trtllm_fp8_per_tensor_scale_moe")
flashinfer_cutlass_fused_moe = _lazy_import_wrapper("flashinfer.fused_moe",
                                                    "cutlass_fused_moe")
fp4_quantize = _lazy_import_wrapper("flashinfer", "fp4_quantize")
nvfp4_block_scale_interleave = _lazy_import_wrapper(
    "flashinfer", "nvfp4_block_scale_interleave")
trtllm_fp4_block_scale_moe = _lazy_import_wrapper(
    "flashinfer", "trtllm_fp4_block_scale_moe")

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
        ("flashinfer.fused_moe", "trtllm_fp4_block_scale_moe"),
    ]

    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_nvidia_artifactory() -> bool:
    """Return ``True`` if NVIDIA's artifactory is accessible.

    This checks connectivity to the kernel inference library artifactory
    which is required for downloading certain cubin kernels like TRTLLM FHMA.
    """
    try:
        # Use a short timeout to avoid blocking for too long
        response = requests.get(FLASHINFER_CUBINS_REPOSITORY, timeout=5)
        accessible = response.status_code == 200
        if accessible:
            logger.debug_once("NVIDIA artifactory is accessible")
        else:
            logger.warning_once(
                "NVIDIA artifactory returned failed status code: %d",
                response.status_code)
        return accessible
    except Exception as e:
        logger.warning_once("Failed to connect to NVIDIA artifactory: %s", e)
        return False


def use_trtllm_attention(
    num_tokens: int,
    max_seq_len: int,
    kv_cache_dtype: str,
    num_qo_heads: Optional[int],
    num_kv_heads: Optional[int],
    attn_head_size: Optional[int],
) -> bool:
    # Requires SM100 and NVIDIA artifactory to be accessible to download cubins
    if not (current_platform.is_device_capability(100)
            and has_nvidia_artifactory()):
        return False

    # Check if the dimensions are supported by TRTLLM decode attention
    if (attn_head_size is None or num_qo_heads is None or num_kv_heads is None
            or num_qo_heads % num_kv_heads != 0):
        return False

    env_value = envs.VLLM_USE_TRTLLM_ATTENTION
    if env_value is not None:
        logger.info_once("VLLM_USE_TRTLLM_ATTENTION is set to %s", env_value)
        # Environment variable is set - respect it
        # Making the conditional check for zero because
        # the path is automatically enabled if the batch size condition
        # is satisfied.
        use_trtllm = (env_value == "1")
        if use_trtllm:
            logger.info_once("Using TRTLLM attention.")
        return use_trtllm
    else:
        # Environment variable not set - use auto-detection
        use_trtllm = (num_tokens <= 256 and max_seq_len < 131072
                      and kv_cache_dtype == "auto")
        if use_trtllm:
            logger.warning_once("Using TRTLLM attention (auto-detected).")
        return use_trtllm


__all__ = [
    "has_flashinfer",
    "flashinfer_trtllm_fp8_block_scale_moe",
    "flashinfer_cutlass_fused_moe",
    "fp4_quantize",
    "nvfp4_block_scale_interleave",
    "trtllm_fp4_block_scale_moe",
    "autotune",
    "has_flashinfer_moe",
    "has_flashinfer_cutlass_fused_moe",
    "has_nvidia_artifactory",
    "use_trtllm_attention",
]
