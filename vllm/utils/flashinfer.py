# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for FlashInfer API changes.

Users of vLLM should always import **only** these wrappers.
"""

import contextlib
import functools
import importlib
import importlib.util
import os
import shutil
from collections.abc import Callable
from typing import Any, NoReturn

import requests
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
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
def has_flashinfer_cubin() -> bool:
    """Return `True` if flashinfer-cubin package is available."""
    if envs.VLLM_HAS_FLASHINFER_CUBIN:
        return True
    if importlib.util.find_spec("flashinfer_cubin") is not None:
        return True
    logger.debug_once("flashinfer-cubin package was not found")
    return False


@functools.cache
def has_flashinfer() -> bool:
    """Return `True` if flashinfer-python package is available."""
    # Use find_spec to check if the module exists without importing it
    # This avoids potential CUDA initialization side effects
    if importlib.util.find_spec("flashinfer") is None:
        logger.debug_once("FlashInfer unavailable since package was not found")
        return False
    # When not using flashinfer cubin,
    # Also check if nvcc is available since it's required to JIT compile flashinfer
    if not has_flashinfer_cubin() and shutil.which("nvcc") is None:
        logger.debug_once(
            "FlashInfer unavailable since nvcc was not found "
            "and not using pre-downloaded cubins"
        )
        return False
    return True


def _missing(*_: Any, **__: Any) -> NoReturn:
    """Placeholder for unavailable FlashInfer backend."""
    raise RuntimeError(
        "FlashInfer backend is not available. Please install the package "
        "to enable FlashInfer kernels: "
        "https://github.com/flashinfer-ai/flashinfer"
    )


def _get_submodule(module_name: str) -> Any | None:
    """Safely import a submodule and return it, or None if not available."""
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None


# General lazy import wrapper
def _lazy_import_wrapper(
    module_name: str, attr_name: str, fallback_fn: Callable[..., Any] = _missing
):
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
    "flashinfer.fused_moe", "trtllm_fp8_block_scale_moe"
)
flashinfer_trtllm_fp8_per_tensor_scale_moe = _lazy_import_wrapper(
    "flashinfer.fused_moe", "trtllm_fp8_per_tensor_scale_moe"
)
flashinfer_cutlass_fused_moe = _lazy_import_wrapper(
    "flashinfer.fused_moe", "cutlass_fused_moe"
)
flashinfer_cutedsl_grouped_gemm_nt_masked = _lazy_import_wrapper(
    "flashinfer.cute_dsl.blockscaled_gemm", "grouped_gemm_nt_masked"
)
flashinfer_fp4_quantize = _lazy_import_wrapper("flashinfer", "fp4_quantize")
nvfp4_batched_quantize = _lazy_import_wrapper("flashinfer", "nvfp4_batched_quantize")
silu_and_mul_scaled_nvfp4_experts_quantize = _lazy_import_wrapper(
    "flashinfer", "silu_and_mul_scaled_nvfp4_experts_quantize"
)
scaled_fp4_grouped_quantize = _lazy_import_wrapper(
    "flashinfer", "scaled_fp4_grouped_quantize"
)
nvfp4_block_scale_interleave = _lazy_import_wrapper(
    "flashinfer", "nvfp4_block_scale_interleave"
)
trtllm_fp4_block_scale_moe = _lazy_import_wrapper(
    "flashinfer", "trtllm_fp4_block_scale_moe"
)

# Special case for autotune since it returns a context manager
autotune = _lazy_import_wrapper(
    "flashinfer.autotuner",
    "autotune",
    fallback_fn=lambda *args, **kwargs: contextlib.nullcontext(),
)


@functools.cache
def has_flashinfer_comm() -> bool:
    """Return `True` if FlashInfer comm module is available."""
    return has_flashinfer() and importlib.util.find_spec("flashinfer.comm") is not None


@functools.cache
def has_flashinfer_all2all() -> bool:
    """Return `True` if FlashInfer mnnvl all2all is available."""
    if not has_flashinfer_comm():
        return False

    # Check if all required functions are available
    required_functions = [
        ("flashinfer.comm", "Mapping"),
        ("flashinfer.comm.mnnvl", "MnnvlMemory"),
        ("flashinfer.comm.trtllm_alltoall", "MnnvlMoe"),
        ("flashinfer.comm.trtllm_alltoall", "MoEAlltoallInfo"),
    ]

    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_flashinfer_moe() -> bool:
    """Return `True` if FlashInfer MoE module is available."""
    return (
        has_flashinfer()
        and importlib.util.find_spec("flashinfer.fused_moe") is not None
    )


@functools.cache
def has_flashinfer_cutedsl() -> bool:
    """Return ``True`` if FlashInfer cutedsl module is available."""
    return (
        has_flashinfer() and importlib.util.find_spec("flashinfer.cute_dsl") is not None
    )


@functools.cache
def has_flashinfer_trtllm_fused_moe() -> bool:
    """Return `True` if FlashInfer TRTLLM fused MoE is available."""
    if not has_flashinfer_moe():
        return False
    required_functions = [
        ("flashinfer.fused_moe", "trtllm_fp8_block_scale_moe"),
        ("flashinfer.fused_moe", "trtllm_fp8_per_tensor_scale_moe"),
        ("flashinfer.fused_moe", "trtllm_fp4_block_scale_moe"),
    ]
    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_flashinfer_cutlass_fused_moe() -> bool:
    """Return `True` if FlashInfer CUTLASS fused MoE is available."""
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
def has_flashinfer_cutedsl_grouped_gemm_nt_masked() -> bool:
    """Return ``True`` if FlashInfer CUTLASS fused MoE is available."""
    if not has_flashinfer_cutedsl():
        return False

    # Check if all required functions are available
    required_functions = [
        ("flashinfer.cute_dsl.blockscaled_gemm", "grouped_gemm_nt_masked"),
        ("flashinfer", "scaled_fp4_grouped_quantize"),
        ("flashinfer", "silu_and_scaled_nvfp4_experts_quantize"),
    ]

    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_nvidia_artifactory() -> bool:
    """Return `True` if NVIDIA's artifactory is accessible.

    This checks connectivity to the kernel inference library artifactory
    which is required for downloading certain cubin kernels like TRTLLM FHMA.
    """
    # If we have pre-downloaded cubins, we can assume the cubins are available.
    if has_flashinfer_cubin():
        return True

    try:
        # Use a short timeout to avoid blocking for too long
        response = requests.get(FLASHINFER_CUBINS_REPOSITORY, timeout=5)
        accessible = response.status_code == 200
        if accessible:
            logger.debug_once("NVIDIA artifactory is accessible")
        else:
            logger.warning_once(
                "NVIDIA artifactory returned failed status code: %d",
                response.status_code,
            )
        return accessible
    except Exception as e:
        logger.warning_once("Failed to connect to NVIDIA artifactory: %s", e)
        return False


@functools.cache
def check_trtllm_attention_support(
    is_prefill: bool,
    num_qo_heads: int | None = None,
    num_kv_heads: int | None = None,
    dcp_world_size: int | None = None,
    kv_cache_dtype: str | None = None,
    q_data_type: torch.dtype | None = None,
    has_sinks: bool | None = None,
    has_spec: bool | None = None,
) -> tuple[bool | None, str]:
    """
    Check if the provided config + current platform is supported by TRTLLM attention.

    Args:
        is_prefill: Whether it is prefill.
        num_qo_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.
        dcp_world_size: World size of decode context parallel.
        kv_cache_dtype: Data type of the key/value cache. Could be "auto".
        q_data_type: Data type of the query.
        has_sinks: Whether sinks are being used.
        has_spec: Whether speculative decoding is being used.

    If any args (except for is_prefill) are set to None, the check for that arg is
    skipped.

    Returns:
        A tuple of (bool | None, str). If the bool is:
        - True: TRTLLM attention must be used.
        - False: TRTLLM attention must not be used.
        - None: TRTLLM attention can be used.
        The str is the reason why it must or must not be used. Empty string if can be
        used.
    """

    if vllm_is_batch_invariant():
        return False, "Batch-invariant mode is enabled."

    if not has_nvidia_artifactory():
        return False, "NVIDIA artifactory is not accessible."

    if current_platform.is_device_capability(90):
        if is_prefill:
            return False, "SM90 is not supported for prefill."
        if q_data_type in [torch.float8_e4m3fn, torch.float8_e5m2]:
            return False, "xqa does not support FP8-Q."
    elif current_platform.is_device_capability_family(100):
        if (
            is_prefill
            and kv_cache_dtype is not None
            and not kv_cache_dtype.startswith("fp8")
            and q_data_type in [torch.float8_e4m3fn, torch.float8_e5m2]
        ):
            return False, "trtllm-gen prefill does not support FP8-Q with BF16/FP16-Q."
    else:
        return False, "SMs other than 90/100/103 are not supported."

    if dcp_world_size is not None and dcp_world_size > 1:
        return False, "DCP is not supported due to lack of LSE return support."

    if (
        num_qo_heads is not None
        and num_kv_heads is not None
        and num_qo_heads % num_kv_heads != 0
    ):
        return False, "num_qo_heads must be a multiple of num_kv_heads."

    if has_spec and not is_prefill:
        return True, "Has speculative decoding in decode phase."

    if has_sinks:
        return True, "Has attention sinks."

    return None, ""


def force_use_trtllm_attention() -> bool | None:
    """
    This function should only be called during initialization stage when vllm config
    is set.
    Return `None` if --attention-config.use_trtllm_attention is not set,
    return `True` if TRTLLM attention is forced to be used,
    return `False` if TRTLLM attention is forced to be not used.
    """
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    return vllm_config.attention_config.use_trtllm_attention


def use_trtllm_attention(
    is_prefill: bool,
    num_qo_heads: int | None = None,
    num_kv_heads: int | None = None,
    dcp_world_size: int | None = None,
    kv_cache_dtype: str | None = None,
    q_data_type: torch.dtype | None = None,
    has_sinks: bool | None = None,
    has_spec: bool | None = None,
    silent: bool = False,
) -> bool:
    """
    Decides whether to use TRTLLM attention based on these two functions:
    - check_trtllm_attention_support(): whether TRTLLM attention must or must not be
      used.
    - force_use_trtllm_attention(): whether the user wants to force/disable TRTLLM
      attention.
    If the decision does not match the user's preference, print the warning messages.

    Args:
        is_prefill: Whether it is prefill.
        num_qo_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.
        dcp_world_size: World size of decode context parallel.
        kv_cache_dtype: Data type of the key/value cache. Could be "auto".
        q_data_type: Data type of the query.
        has_sinks: Whether sinks are being used.
        has_spec: Whether speculative decoding is being used.
        silent: Whether to print the warning/info messages.

    If any args (except for is_prefill) are set to None, the check for that arg is
    skipped.

    Returns: whether to use TRTLLM attention.
    """
    supports_trtllm, reason = check_trtllm_attention_support(
        is_prefill,
        num_qo_heads,
        num_kv_heads,
        dcp_world_size,
        kv_cache_dtype,
        q_data_type,
        has_sinks,
        has_spec,
    )
    force_use_trtllm = force_use_trtllm_attention()
    phase_str = "prefill" if is_prefill else "decode"
    prefix = "[FlashInfer Attention]"

    # Helper functions to print warning/info if not silent.
    def print_warning(msg: str):
        if not silent:
            logger.warning_once(msg)

    def print_info(msg: str):
        if not silent:
            logger.info_once(msg)

    if force_use_trtllm is True:
        if supports_trtllm is False:
            print_warning(
                f"{prefix} Using non-TRTLLM for {phase_str} even though --attention-"
                f"config.use_trtllm_attention is set to 1. (Reason: {reason})"
            )
            return False
        else:
            print_info(
                f"{prefix} Using TRTLLM for {phase_str}. (Reason: --attention-config."
                f"use_trtllm_attention is set to 1.)"
            )
            return True
    elif force_use_trtllm is False:
        if supports_trtllm is True:
            print_warning(
                f"{prefix} Using TRTLLM for {phase_str} even though --attention-config."
                f"use_trtllm_attention is set to 0. (Reason: {reason})"
            )
            return True
        else:
            print_info(
                f"{prefix} Using non-TRTLLM for {phase_str}. (Reason: --attention-"
                f"config.use_trtllm_attention is set to 0.)"
            )
            return False
    else:
        if supports_trtllm is True:
            print_info(f"{prefix} Using TRTLLM for {phase_str}. (Reason: {reason})")
            return True
        elif supports_trtllm is False:
            print_info(f"{prefix} Using non-TRTLLM for {phase_str}. (Reason: {reason})")
            return False
        else:
            print_info(
                f"{prefix} Using TRTLLM for {phase_str}. (Reason: auto-detected.)"
            )
            return True


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

        return flashinfer_mm_fp4_(
            A, B, A_scale, B_scale, g_scale, dtype, block_size=16, backend=backend
        )

    @torch.library.register_fake(
        "vllm::flashinfer_mm_fp4",
    )
    def flashinfer_mm_fp4_fake(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        g_scale: torch.Tensor,
        dtype: torch.dtype,
        backend: str,
    ) -> torch.Tensor:
        return torch.empty(A.shape[0], B.shape[1], dtype=dtype, device=A.device)

    @torch.library.custom_op(
        "vllm::bmm_fp8",
        mutates_args=[],
        device_types="cuda",
    )
    def bmm_fp8(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        dtype: torch.dtype,
        backend: str,
    ) -> torch.Tensor:
        from flashinfer import bmm_fp8 as bmm_fp8_

        return bmm_fp8_(A, B, A_scale, B_scale, dtype, None, backend)

    @torch.library.register_fake(
        "vllm::bmm_fp8",
    )
    def bmm_fp8_fake(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        dtype: torch.dtype,
        backend: str,
    ) -> torch.Tensor:
        return torch.empty(
            A.shape[0], A.shape[1], B.shape[2], dtype=dtype, device=A.device
        )


def flashinfer_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    backend: str,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    assert block_scale_a.ndim == 2 and block_scale_b.ndim == 2
    assert a.stride(-1) == 1 and b.stride(-1) == 1
    assert a.shape[1] == b.shape[1]

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


def flashinfer_scaled_fp8_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert scale_a.numel() == 1 and scale_b.numel() == 1
    assert a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn
    assert a.device.type == "cuda" and b.device.type == "cuda"
    assert scale_a.dtype == torch.float32 and scale_b.dtype == torch.float32
    assert scale_a.device.type == "cuda" and scale_b.device.type == "cuda"

    output = bmm_fp8(
        a.unsqueeze(0),
        b.unsqueeze(0),
        scale_a,
        scale_b,
        out_dtype,
        "auto",
    ).view(a.shape[0], b.shape[1])

    if bias is not None:
        output = output + bias
    return output


__all__ = [
    "has_flashinfer",
    "flashinfer_trtllm_fp8_block_scale_moe",
    "flashinfer_cutlass_fused_moe",
    "flashinfer_cutedsl_grouped_gemm_nt_masked",
    "flashinfer_fp4_quantize",
    "silu_and_mul_scaled_nvfp4_experts_quantize",
    "scaled_fp4_grouped_quantize",
    "nvfp4_block_scale_interleave",
    "trtllm_fp4_block_scale_moe",
    "autotune",
    "has_flashinfer_moe",
    "has_flashinfer_comm",
    "has_flashinfer_all2all",
    "has_flashinfer_cutlass_fused_moe",
    "has_flashinfer_cutedsl_grouped_gemm_nt_masked",
    "has_nvidia_artifactory",
    "check_trtllm_attention_support",
    "force_use_trtllm_attention",
    "use_trtllm_attention",
    "flashinfer_scaled_fp4_mm",
    "flashinfer_scaled_fp8_mm",
]
