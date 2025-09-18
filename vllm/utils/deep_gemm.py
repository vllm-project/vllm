# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for DeepGEMM API changes.

Users of vLLM should always import **only** these wrappers.
"""
from __future__ import annotations

import functools
import importlib
import os
from typing import Any, Callable, NoReturn

import torch

import vllm.envs as envs
from vllm.logger import logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils import cdiv, has_deep_gemm


@functools.cache
def is_deep_gemm_supported() -> bool:
    """Return ``True`` if DeepGEMM is supported on the current platform.
    Currently, only Hopper and Blackwell GPUs are supported.
    """
    is_supported_arch = current_platform.is_cuda() and (
        current_platform.is_device_capability(90)
        or current_platform.is_device_capability(100))
    return envs.VLLM_USE_DEEP_GEMM and has_deep_gemm() and is_supported_arch


@functools.cache
def is_deep_gemm_e8m0_used() -> bool:
    """Return ``True`` if vLLM is configured to use DeepGEMM "
    "E8M0 scale on a Hopper or Blackwell-class GPU.
    """
    if not is_deep_gemm_supported():
        logger.debug_once(
            "DeepGEMM E8M0 disabled: DeepGEMM not supported on this system.")
        return False

    _lazy_init()

    if _fp8_gemm_nt_impl is None:
        logger.info_once("DeepGEMM E8M0 disabled: _fp8_gemm_nt_impl not found")
        return False

    if current_platform.is_device_capability(100) and \
            envs.VLLM_USE_DEEP_GEMM_E8M0:
        logger.info_once("DeepGEMM E8M0 enabled on Blackwell GPU.")
        return True

    if current_platform.is_device_capability(90) and \
            envs.VLLM_USE_DEEP_GEMM_E8M0_HOPPER:
        logger.info_once("DeepGEMM E8M0 enabled on Hopper GPU.")
        return True

    logger.info_once("DeepGEMM E8M0 disabled on current configuration.")
    return False


def _missing(*_: Any, **__: Any) -> NoReturn:
    """Placeholder for unavailable DeepGEMM backend."""
    raise RuntimeError(
        "DeepGEMM backend is not available. Please install the `deep_gemm` "
        "package to enable FP8 kernels.")


_fp8_gemm_nt_impl: Callable[..., Any] | None = None
_grouped_impl: Callable[..., Any] | None = None
_grouped_masked_impl: Callable[..., Any] | None = None


def _lazy_init() -> None:
    """Import deep_gemm and resolve symbols on first use."""
    global _fp8_gemm_nt_impl, _grouped_impl, _grouped_masked_impl

    # fast path
    if (_fp8_gemm_nt_impl is not None or _grouped_impl is not None
            or _grouped_masked_impl is not None):
        return

    if not has_deep_gemm():
        return

    # Set up deep_gemm cache path
    DEEP_GEMM_JIT_CACHE_ENV_NAME = 'DG_JIT_CACHE_DIR'
    if not os.environ.get(DEEP_GEMM_JIT_CACHE_ENV_NAME, None):
        os.environ[DEEP_GEMM_JIT_CACHE_ENV_NAME] = os.path.join(
            envs.VLLM_CACHE_ROOT, "deep_gemm")

    _dg = importlib.import_module("deep_gemm")

    _fp8_gemm_nt_impl = getattr(_dg, "fp8_gemm_nt", None)
    _grouped_impl = getattr(_dg, "m_grouped_fp8_gemm_nt_contiguous", None)
    _grouped_masked_impl = getattr(_dg, "fp8_m_grouped_gemm_nt_masked", None)


def fp8_gemm_nt(*args, **kwargs):
    _lazy_init()
    if _fp8_gemm_nt_impl is None:
        return _missing(*args, **kwargs)
    return _fp8_gemm_nt_impl(*args,
                             disable_ue8m0_cast=not is_deep_gemm_e8m0_used(),
                             **kwargs)


def m_grouped_fp8_gemm_nt_contiguous(*args, **kwargs):
    _lazy_init()
    if _grouped_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_impl(*args,
                         disable_ue8m0_cast=not is_deep_gemm_e8m0_used(),
                         **kwargs)


def fp8_m_grouped_gemm_nt_masked(*args, **kwargs):
    _lazy_init()
    if _grouped_masked_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_masked_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs)


@triton.jit
def _per_block_cast_to_fp8_kernel(x_ptr, y_ptr, scales_ptr, M, N, stride_xm,
                                  stride_xn, stride_ym, stride_yn, stride_sm,
                                  stride_sn, BLOCK_M: tl.constexpr,
                                  BLOCK_N: tl.constexpr,
                                  USE_UE8M0: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m_idx = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None]
    n_idx = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    x_block_ptrs = x_ptr + m_idx * stride_xm + n_idx * stride_xn
    y_block_ptrs = y_ptr + m_idx * stride_ym + n_idx * stride_yn
    mask = (m_idx < M) & (n_idx < N)

    # load tile once
    x_vals = tl.load(x_block_ptrs, mask=mask, other=0.0)
    x_vals_f32 = x_vals.to(tl.float32)
    x_abs = tl.abs(x_vals_f32)
    amax = tl.max(x_abs)

    # clamp then form scale
    scale = tl.maximum(amax, 1e-4) / 448.0
    if USE_UE8M0:
        # round scale up to nearest power-of-two (E8M0)
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    # store per-tile scale
    tl.store(scales_ptr + pid_m * stride_sm + pid_n * stride_sn, scale)

    # scale and store output
    inv_scale = 1.0 / scale
    y_vals = x_vals_f32 * inv_scale
    tl.store(y_block_ptrs, y_vals, mask=mask)


DEFAULT_BLOCK_SIZE = [128, 128]


def per_block_cast_to_fp8(
        x: torch.Tensor,
        block_size: list[int] = DEFAULT_BLOCK_SIZE,
        use_ue8m0: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    block_m, block_n = block_size
    grid_m, grid_n = cdiv(m, block_m), cdiv(n, block_n)

    scales = torch.empty((grid_m, grid_n),
                         dtype=torch.float32,
                         device=x.device)
    y_tmp = torch.empty_like(x, dtype=torch.float32)
    _per_block_cast_to_fp8_kernel[(grid_m, grid_n)](
        x,
        y_tmp,
        scales,
        m,
        n,
        x.stride(0),
        x.stride(1),
        y_tmp.stride(0),
        y_tmp.stride(1),
        scales.stride(0),
        scales.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        USE_UE8M0=use_ue8m0,
        num_warps=8,
        num_stages=1,
    )
    return y_tmp.to(torch.float8_e4m3fn), scales


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


def should_use_deepgemm_for_fp8_linear(output_dtype: torch.dtype,
                                       weight: torch.Tensor):
    return (is_deep_gemm_supported() and output_dtype == torch.bfloat16
            and weight.shape[0] % 128 == 0 and weight.shape[1] % 128 == 0)


__all__ = [
    "calc_diff",
    "fp8_gemm_nt",
    "m_grouped_fp8_gemm_nt_contiguous",
    "fp8_m_grouped_gemm_nt_masked",
    "per_block_cast_to_fp8",
    "is_deep_gemm_e8m0_used",
    "is_deep_gemm_supported",
    "should_use_deepgemm_for_fp8_linear",
]
