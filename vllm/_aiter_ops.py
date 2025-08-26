# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from typing import Callable, Optional

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op


def is_aiter_supported(func: Callable) -> Callable:
    """Decorator that only executes the function if 
    ROCm AITER package is supported on gfx9 archs.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # checks the platform, device arch and aiter library existance.
        from importlib.util import find_spec

        from vllm.platforms.rocm import on_gfx9

        if (current_platform.is_rocm() and on_gfx9()
                and find_spec("aiter") is not None):
            return func(*args, **kwargs)
        else:
            # Return None or do nothing if not supported
            return None

    return wrapper


def _rocm_aiter_tuned_gemm_impl(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
        scale_a: Optional[torch.Tensor] = None,
        scale_b: Optional[torch.Tensor] = None) -> torch.Tensor:

    # This AITER function can be used for
    # - BF16 and FP16 matmul
    #   e.g. vllm/model_executor/layers/linear.py
    # - per-tensor activations + per-tensor weights
    #   e.g. vllm/model_executor/layers/quantization/utils/w8a8_utils.py
    from aiter.tuned_gemm import tgemm as aiter_tgemm

    return aiter_tgemm.mm(input,
                          weight,
                          otype=out_dtype,
                          scale_a=scale_a,
                          scale_b=scale_b,
                          bias=bias)


def _rocm_aiter_tuned_gemm_fake(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
        scale_a: Optional[torch.Tensor] = None,
        scale_b: Optional[torch.Tensor] = None) -> torch.Tensor:

    m = input.shape[0]
    n = weight.shape[0]
    if out_dtype is None:
        out_dtype = input.dtype
    return torch.empty((m, n), dtype=out_dtype, device=input.device)


# Global flag to ensure ops are registered only once
_OPS_REGISTERED = False


class aiter_ops:
    _IS_AITER_ENABLED = envs.VLLM_ROCM_USE_AITER

    @classmethod
    @is_aiter_supported
    def is_linear_enabled(cls) -> bool:
        """"Verifies device specs and availability of env variable."""
        return envs.VLLM_ROCM_USE_AITER_LINEAR and cls._IS_AITER_ENABLED

    @staticmethod
    @is_aiter_supported
    def register_ops_once() -> None:
        global _OPS_REGISTERED
        if not _OPS_REGISTERED:
            # register all the custom ops here
            direct_register_custom_op(
                op_name="rocm_aiter_tuned_gemm",
                op_func=_rocm_aiter_tuned_gemm_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_tuned_gemm_fake,
                dispatch_key=current_platform.dispatch_key,
            )
            _OPS_REGISTERED = True

    @staticmethod
    def rocm_aiter_tuned_gemm(
            input: torch.Tensor,  # [M, K]
            weight: torch.Tensor,  # [N, K]
            bias: Optional[torch.Tensor] = None,
            out_dtype: Optional[torch.dtype] = None,
            scale_a: Optional[torch.Tensor] = None,
            scale_b: Optional[torch.Tensor] = None) -> torch.Tensor:

        return torch.ops.vllm.rocm_aiter_tuned_gemm(
            input,
            weight,
            bias=bias,
            out_dtype=out_dtype,
            scale_a=scale_a,
            scale_b=scale_b,
        )


aiter_ops.register_ops_once()
