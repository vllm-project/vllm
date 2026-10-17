# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

_cutedsl_available: bool | None = None
_cute_ctx: tuple[Any, Any] | None = None


def is_cutedsl_available(kernel_name: str) -> bool:
    global _cutedsl_available
    if _cutedsl_available is not None:
        return _cutedsl_available
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401

        _cutedsl_available = True
    except ImportError:
        _cutedsl_available = False
        logger.info(
            "cuteDSL (CUTLASS Python) not available, %s disabled",
            kernel_name,
        )
    return _cutedsl_available


def cute_context():
    global _cute_ctx
    if _cute_ctx is not None:
        return _cute_ctx
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream

    _cute_ctx = (cute, CUstream)
    return _cute_ctx


def current_cuda_stream():
    _, CUstream = cute_context()
    from vllm.utils.torch_utils import current_stream

    return CUstream(current_stream().cuda_stream)


def use_pdl() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_arch_support_pdl()


def cutlass_dtype(dtype: torch.dtype):
    from cutlass import BFloat16, Float16, Float32

    if dtype == torch.bfloat16:
        return BFloat16
    if dtype == torch.float16:
        return Float16
    if dtype == torch.float32:
        return Float32
    raise ValueError(f"unsupported router GEMM dtype: {dtype}")


def make_fake_gemm_tensors(
    *,
    M,
    K,
    N,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    divisibility: int,
):
    from cutlass import Float32
    from quack.compile_utils import make_fake_tensor

    hidden_states = make_fake_tensor(
        cutlass_dtype(a_dtype), (M, K), divisibility=divisibility
    )
    router_weight = make_fake_tensor(
        cutlass_dtype(b_dtype), (N, K), divisibility=divisibility
    )
    output = make_fake_tensor(Float32, (M, N), divisibility=1)
    return hidden_states, router_weight, output


def validate_common_gemm_inputs(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    output_dtype: torch.dtype,
    *,
    op_name: str,
    k_multiple: int | None = None,
) -> None:
    if hidden_states.dim() != 2 or router_weight.dim() != 2:
        raise ValueError("hidden_states and router_weight must be 2D tensors")
    if hidden_states.device.type != "cuda" or router_weight.device.type != "cuda":
        raise ValueError("hidden_states and router_weight must have device_type=cuda")
    if hidden_states.device != router_weight.device:
        raise ValueError(
            "hidden_states and router_weight must be on the same CUDA device"
        )
    if output_dtype != torch.float32:
        raise ValueError(f"{op_name} only supports output_dtype=torch.float32")
    if hidden_states.shape[1] != router_weight.shape[1]:
        raise ValueError(
            "hidden_states and router_weight must have matching K dimensions"
        )
    if k_multiple is not None and hidden_states.shape[1] % k_multiple != 0:
        raise ValueError(f"{op_name} requires K to be divisible by {k_multiple}")
    if not hidden_states.is_contiguous() or not router_weight.is_contiguous():
        raise ValueError(
            "hidden_states and router_weight must be contiguous row-major inputs"
        )
