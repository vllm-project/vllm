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


def _ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def _align(x: int, y: int) -> int:
    return cdiv(x, y) * y


DEFAULT_BLOCK_SIZE = [128, 128]


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/dd6ed14acbc7445dcef224248a77ab4d22b5f240/deep_gemm/utils/math.py#L38
# TODO(wentao): optimize this function, using triton or cuda kernel
def per_block_cast_to_fp8(
        x: torch.Tensor,
        block_size: list[int] = DEFAULT_BLOCK_SIZE,
        use_ue8m0: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    block_m, block_n = block_size
    x_padded = torch.zeros((_align(m, block_m), _align(n, block_n)),
                           dtype=x.dtype,
                           device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, block_m, x_padded.size(1) // block_n, block_n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2))


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
