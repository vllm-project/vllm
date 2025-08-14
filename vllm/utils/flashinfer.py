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
from typing import Any, Callable, NoReturn

import requests
import torch

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


def support_trtllm_attention(
    num_qo_heads: int,
    num_kv_heads: int,
) -> bool:
    """
    Return ``True`` if the attention satisfies the minimum requirement
    of TRT-LLM kernel
    """
    # Requires SM100 and NVIDIA artifactory to be accessible to download cubins
    # and check if the dimensions are supported
    return (current_platform.is_device_capability(100)
            and has_nvidia_artifactory() and num_qo_heads % num_kv_heads == 0)


def use_trtllm_attention(
    num_qo_heads: int,
    num_kv_heads: int,
    num_tokens: int,
    max_seq_len: int,
    kv_cache_dtype: str,
    has_sinks: bool = False,
    enable_fusion: bool = False,
) -> bool:
    """
    Return ``True`` if TRTLLM attention is usable
    """
    if not support_trtllm_attention(num_qo_heads, num_kv_heads):
        return False

    if kv_cache_dtype.startswith("fp8"):
        if enable_fusion:
            logger.info_once("Using TRTLLM attention (FP8 kv cache required).")
            return True
        else:
            # Remove this when TRTLLM attn kernel supports FP8 kv cache
            # without attn+quant fusion.
            raise ValueError("Flashinfer TRTLLM attention does not support "
                             "FP8 kv cache without attn+quant fusion. "
                             "Suggested the following configs: "
                             "(1) set kv_cache_dtype to 'auto', or "
                             "(2) turn on enable_attn_fusion, or "
                             "(3) disable Flashinfer backend")
    elif kv_cache_dtype == "auto" and enable_fusion:
        raise ValueError("Flashinfer TRTLLM attention does not support "
                         "auto kv cache dtype with attn+quant fusion. "
                         "Suggested the following configs: "
                         "(1) set kv_cache_dtype to 'fp8', or "
                         "(2) turn off enable_attn_fusion, or "
                         "(3) disable Flashinfer backend")

    # If sinks are being used, we must use TRTLLM attention as it's
    # the only backend that supports them
    if has_sinks:
        logger.info_once(
            "Using TRTLLM attention (required for attention sinks).")
        return True

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


if has_flashinfer():

    @torch.library.custom_op(
        "vllm::flashinfer_mm_fp4",
        mutates_args=[],
        device_types="cuda",
    )
    def flashinfer_mm_fp4(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        g_scale: torch.Tensor,
        dtype: torch.dtype,
        backend: str,
    ) -> torch.Tensor:
        from flashinfer import mm_fp4 as flashinfer_mm_fp4_
        return flashinfer_mm_fp4_(A,
                                  B,
                                  A_scale,
                                  B_scale,
                                  g_scale,
                                  dtype,
                                  block_size=16,
                                  backend=backend)

    @torch.library.register_fake("vllm::flashinfer_mm_fp4", )
    def flashinfer_mm_fp4_fake(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        g_scale: torch.Tensor,
        dtype: torch.dtype,
        backend: str,
    ) -> torch.Tensor:
        return torch.empty(A.shape[0],
                           B.shape[1],
                           dtype=dtype,
                           device=A.device)


def flashinfer_scaled_fp4_mm(a: torch.Tensor, b: torch.Tensor,
                             block_scale_a: torch.Tensor,
                             block_scale_b: torch.Tensor, alpha: torch.Tensor,
                             out_dtype: torch.dtype,
                             backend: str) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    assert block_scale_a.ndim == 2 and block_scale_b.ndim == 2
    assert a.stride(-1) == 1 and b.stride(-1) == 1
    assert a.shape[1] == b.shape[1]
    assert block_scale_a.shape[1] == a.shape[1] // 8
    assert block_scale_b.shape[1] == b.shape[1] // 8

    if backend == "cutlass":
        block_scale_a = block_scale_a.view(torch.uint8)
        block_scale_b = block_scale_b.view(torch.uint8)

    return flashinfer_mm_fp4(
        a,
        b.t(),
        block_scale_a,
        block_scale_b.t(),
        alpha,
        out_dtype,
        backend=backend,
    )


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
    "support_trtllm_attention",
    "use_trtllm_attention",
    "flashinfer_scaled_fp4_mm",
]
