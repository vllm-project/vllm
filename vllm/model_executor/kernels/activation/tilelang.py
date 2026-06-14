# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang activation op wrappers."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any

import torch

from vllm.utils.torch_utils import direct_register_custom_op


@cache
def _get_tilelang_activation_kernel(
    kernel_name: str,
    block_size: int = 1024,
    threads: int = 128,
):
    from vllm.model_executor.kernels.activation import tilelang_kernels

    return getattr(tilelang_kernels, kernel_name)(
        BLOCK_SIZE=block_size,
        threads=threads,
    )


def _validate_act_and_mul_tilelang_args(
    out: torch.Tensor,
    x: torch.Tensor,
) -> int:
    assert x.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    assert x.ndim > 0
    assert x.shape[-1] % 2 == 0

    hidden = x.shape[-1] // 2
    assert out.shape == x.shape[:-1] + (hidden,)
    assert x.is_contiguous()
    assert out.is_contiguous()
    return hidden


def _run_act_and_mul_tilelang(
    out: torch.Tensor,
    x: torch.Tensor,
    kernel_name: str,
    *args: float,
) -> None:
    hidden = _validate_act_and_mul_tilelang_args(out, x)
    if x.numel() == 0:
        return

    x_2d = x.view(-1, x.shape[-1])
    out_2d = out.view(-1, hidden)
    kernel = _get_tilelang_activation_kernel(kernel_name)
    kernel(x_2d, out_2d, *args)


def silu_and_mul_tilelang(out: torch.Tensor, x: torch.Tensor) -> None:
    """Compute bf16 SiLU-and-mul into a pre-allocated output tensor."""
    _run_act_and_mul_tilelang(out, x, "silu_and_mul_tilelang_kernel")


def mul_and_silu_tilelang(out: torch.Tensor, x: torch.Tensor) -> None:
    """Compute bf16 mul-and-SiLU into a pre-allocated output tensor."""
    _run_act_and_mul_tilelang(out, x, "mul_and_silu_tilelang_kernel")


def fatrelu_and_mul_tilelang(
    out: torch.Tensor,
    x: torch.Tensor,
    threshold: float,
) -> None:
    """Compute bf16 FATReLU-and-mul into a pre-allocated output tensor."""
    _run_act_and_mul_tilelang(out, x, "fatrelu_and_mul_tilelang_kernel", threshold)


def silu_and_mul_with_clamp_tilelang(
    out: torch.Tensor,
    x: torch.Tensor,
    swiglu_limit: float,
) -> None:
    """Compute bf16 clamped SiLU-and-mul into a pre-allocated output tensor."""
    _run_act_and_mul_tilelang(
        out,
        x,
        "silu_and_mul_with_clamp_tilelang_kernel",
        swiglu_limit,
    )


def gelu_and_mul_tilelang(out: torch.Tensor, x: torch.Tensor) -> None:
    """Compute bf16 GELU-and-mul into a pre-allocated output tensor."""
    _run_act_and_mul_tilelang(out, x, "gelu_and_mul_tilelang_kernel")


def gelu_tanh_and_mul_tilelang(out: torch.Tensor, x: torch.Tensor) -> None:
    """Compute bf16 tanh-approx GELU-and-mul into a pre-allocated output tensor."""
    _run_act_and_mul_tilelang(out, x, "gelu_tanh_and_mul_tilelang_kernel")


def swigluoai_and_mul_tilelang(
    out: torch.Tensor,
    x: torch.Tensor,
    alpha: float,
    limit: float,
) -> None:
    """Compute bf16 GPT-OSS SwiGLU OAI into a pre-allocated tensor."""
    _run_act_and_mul_tilelang(
        out,
        x,
        "swigluoai_and_mul_tilelang_kernel",
        alpha,
        limit,
    )


def swiglustep_and_mul_tilelang(
    out: torch.Tensor,
    x: torch.Tensor,
    limit: float,
) -> None:
    """Compute bf16 SwiGLU-step-and-mul into a pre-allocated output tensor."""
    _run_act_and_mul_tilelang(out, x, "swiglustep_and_mul_tilelang_kernel", limit)


def _activation_tilelang_fake(*args: object, **kwargs: object) -> None:
    return None


_TILELANG_ACTIVATION_OPS: tuple[tuple[str, Callable[..., Any]], ...] = (
    ("silu_and_mul_tilelang", silu_and_mul_tilelang),
    ("mul_and_silu_tilelang", mul_and_silu_tilelang),
    ("fatrelu_and_mul_tilelang", fatrelu_and_mul_tilelang),
    ("silu_and_mul_with_clamp_tilelang", silu_and_mul_with_clamp_tilelang),
    ("gelu_and_mul_tilelang", gelu_and_mul_tilelang),
    ("gelu_tanh_and_mul_tilelang", gelu_tanh_and_mul_tilelang),
    ("swigluoai_and_mul_tilelang", swigluoai_and_mul_tilelang),
    ("swiglustep_and_mul_tilelang", swiglustep_and_mul_tilelang),
)

for _op_name, _op_func in _TILELANG_ACTIVATION_OPS:
    direct_register_custom_op(
        op_name=_op_name,
        op_func=_op_func,
        mutates_args=["out"],
        fake_impl=_activation_tilelang_fake,
    )
