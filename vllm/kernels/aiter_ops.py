# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools

import torch
from torch import Tensor
from torch.library import Library

from vllm import ir
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

current_platform.import_kernels()

_FP8_DTYPE = current_platform.fp8_dtype()


def is_aiter_found() -> bool:
    from importlib.util import find_spec

    return find_spec("aiter") is not None


aiter_lib = Library("vllm_aiter", "FRAGMENT")
"""
This library holds torch custom ops for wrapped AITER ops.
Many AITER ops want to remain invisible to torch.compile even after lowering.
They are thus wrapped into torch custom ops inside the IR op implementations.
"""

direct_register_aiter_op = functools.partial(
    direct_register_custom_op, target_lib=aiter_lib
)
"""Syntactic sugar for registering AITER custom ops."""

AITER_SUPPORTED = is_aiter_found()
"""Most kernels in this file are supported if AITER is installed."""

rms_no_var_16bit_only = (
    lambda x, weight, epsilon, variance_size=None: variance_size is None
    and x.dtype in (torch.float16, torch.bfloat16)
    and (weight is None or weight.dtype == x.dtype)
)
"""AITER rms_norm only supports float16 and bfloat16 acts, no var_size override,
and requires weight dtype to match x dtype."""


@ir.ops.rms_norm.register_impl(
    "aiter", supports_args=rms_no_var_16bit_only, supported=AITER_SUPPORTED
)
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    assert variance_size is None
    assert x.dtype in (torch.float16, torch.bfloat16)
    if weight is None:
        weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    return torch.ops.vllm_aiter.rms_norm(x, weight, epsilon)


def _rms_norm_impl(x: Tensor, weight: Tensor, variance_epsilon: float) -> Tensor:
    from aiter import rms_norm

    if x.dim() > 2:
        x_original_shape = x.shape
        x = x.reshape(-1, x_original_shape[-1])
        x = rms_norm(x, weight, variance_epsilon)
        return x.reshape(x_original_shape)

    return rms_norm(x, weight, variance_epsilon)


def _rms_norm_fake(x: Tensor, weight: Tensor, variance_epsilon: float) -> Tensor:
    return torch.empty_like(x)


direct_register_aiter_op(
    op_name="rms_norm", op_func=_rms_norm_impl, fake_impl=_rms_norm_fake
)


def _static_per_tensor_quant_fp8_impl(x: Tensor, scale: Tensor) -> Tensor:
    from aiter.ops.quant import per_tensor_quant_hip

    return per_tensor_quant_hip(x, scale, _FP8_DTYPE)[0]  # Drop the scale


def _static_per_tensor_quant_fp8_fake(x: Tensor, scale: Tensor) -> Tensor:
    return torch.empty_like(x, dtype=_FP8_DTYPE)


direct_register_aiter_op(
    op_name="static_per_tensor_quant_fp8",
    op_func=_static_per_tensor_quant_fp8_impl,
    fake_impl=_static_per_tensor_quant_fp8_fake,
)

_static_quant_fp8_16bit_per_tensor = (
    lambda x, scale, fp8_dtype, num_token_padding=None: (
        scale.numel() == 1
        and x.dtype in (torch.float16, torch.bfloat16)
        and num_token_padding is None
    )
)
"""AITER static_quant_fp8 requires a scalar (per-tensor) scale, 16-bit activations,
and no token padding. Per-token scales are not supported"""


@ir.ops.static_quant_fp8.register_impl(
    "aiter",
    supports_args=_static_quant_fp8_16bit_per_tensor,
    supported=AITER_SUPPORTED,
)
def static_quant_fp8(
    x: Tensor,
    scale: Tensor,
    fp8_dtype: torch.dtype,
    num_token_padding: int | None = None,
) -> Tensor:
    assert scale.numel() == 1  # Only per tensor
    assert x.dtype in (torch.float16, torch.bfloat16)
    assert num_token_padding is None
    return torch.ops.vllm_aiter.static_per_tensor_quant_fp8(x, scale)


def _dynamic_per_tensor_quant_fp8_impl(x: Tensor) -> tuple[Tensor, Tensor]:
    from aiter.ops.quant import per_tensor_quant_hip

    return per_tensor_quant_hip(x, None, _FP8_DTYPE)


def _dynamic_per_tensor_quant_fp8_fake(x: Tensor) -> tuple[Tensor, Tensor]:
    return (
        torch.empty_like(x, dtype=_FP8_DTYPE),
        torch.empty(1, dtype=torch.float32, device=x.device),
    )


direct_register_aiter_op(
    op_name="dynamic_per_tensor_quant_fp8",
    op_func=_dynamic_per_tensor_quant_fp8_impl,
    fake_impl=_dynamic_per_tensor_quant_fp8_fake,
)


def _dynamic_per_token_quant_fp8_impl(x: Tensor) -> tuple[Tensor, Tensor]:
    from aiter.ops.quant import dynamic_per_token_scaled_quant

    out_shape = x.shape
    out = torch.empty(out_shape, dtype=_FP8_DTYPE, device=x.device)
    scale = torch.empty((*out_shape[:-1], 1), dtype=torch.float32, device=x.device)
    dynamic_per_token_scaled_quant(
        out,
        x,
        scale,
        scale_ub=None,
        shuffle_scale=False,
        num_rows=None,
        num_rows_factor=1,
    )
    return out, scale


def _dynamic_per_token_quant_fp8_fake(x: Tensor) -> tuple[Tensor, Tensor]:
    out_shape = x.shape
    return (
        torch.empty(out_shape, dtype=_FP8_DTYPE, device=x.device),
        torch.empty((*out_shape[:-1], 1), dtype=torch.float32, device=x.device),
    )


direct_register_aiter_op(
    op_name="dynamic_per_token_quant_fp8",
    op_func=_dynamic_per_token_quant_fp8_impl,
    fake_impl=_dynamic_per_token_quant_fp8_fake,
)


_dynamic_quant_fp8_16bit_no_ub = (
    lambda x, per_token, fp8_dtype, scale_ub=None, num_token_padding=None: (
        scale_ub is None
        and num_token_padding is None
        and x.dtype in (torch.float16, torch.bfloat16)
    )
)
"""AITER dynamic_quant_fp8 requires float16/bfloat16, no scale upper bound,
and no token padding."""


@ir.ops.dynamic_quant_fp8.register_impl(
    "aiter",
    supports_args=_dynamic_quant_fp8_16bit_no_ub,
    supported=AITER_SUPPORTED,
)
def dynamic_quant_fp8(
    x: Tensor,
    per_token: bool,
    fp8_dtype: torch.dtype,
    scale_ub: Tensor | None = None,
    num_token_padding: int | None = None,
) -> tuple[Tensor, Tensor]:
    assert scale_ub is None
    assert num_token_padding is None
    assert x.dtype in (torch.float16, torch.bfloat16)
    if per_token:
        return torch.ops.vllm_aiter.dynamic_per_token_quant_fp8(x)
    return torch.ops.vllm_aiter.dynamic_per_tensor_quant_fp8(x)


def _dynamic_group_quant_fp8_impl(x: Tensor, group_size: int) -> tuple[Tensor, Tensor]:
    from aiter import QuantType, get_hip_quant

    aiter_per1x128_quant = get_hip_quant(QuantType.per_1x128)
    return aiter_per1x128_quant(x, quant_dtype=_FP8_DTYPE)


def _dynamic_group_quant_fp8_fake(x: Tensor, group_size: int) -> tuple[Tensor, Tensor]:
    orig_shape = x.shape
    N = orig_shape[-1]
    return (
        torch.empty(orig_shape, dtype=_FP8_DTYPE, device=x.device),
        torch.empty(
            orig_shape[:-1] + ((N + group_size - 1) // group_size,),
            dtype=torch.float32,
            device=x.device,
        ),
    )


direct_register_aiter_op(
    op_name="dynamic_group_quant_fp8",
    op_func=_dynamic_group_quant_fp8_impl,
    fake_impl=_dynamic_group_quant_fp8_fake,
)


_dynamic_group_quant_fp8_128_rowmajor = (
    lambda x, group_shape, column_major, use_ue8m0, fp8_dtype, scale_alignment=1: (
        group_shape[-1] == 128
        and not column_major
        and not use_ue8m0
        and x.is_contiguous()
        and x.shape[-1] % 128 == 0
        and x.dtype in (torch.float16, torch.bfloat16)
    )
)
"""AITER dynamic_group_quant_fp8 requires group_size=128, row-major scales,
no ue8m0 exponent format, contiguous input, hidden dim divisible by 128,
and 16-bit activations."""


@ir.ops.dynamic_group_quant_fp8.register_impl(
    "aiter",
    supports_args=_dynamic_group_quant_fp8_128_rowmajor,
    supported=AITER_SUPPORTED,
)
def dynamic_group_quant_fp8(
    x: Tensor,
    group_shape: list[int],
    column_major: bool,
    use_ue8m0: bool,
    fp8_dtype: torch.dtype,
    scale_alignment: int = 1,
) -> tuple[Tensor, Tensor]:
    assert group_shape[-1] == 128
    assert not column_major
    assert not use_ue8m0
    assert x.shape[-1] % 128 == 0
    assert x.dtype in (torch.float16, torch.bfloat16)
    return torch.ops.vllm_aiter.dynamic_group_quant_fp8(x, group_shape[-1])
