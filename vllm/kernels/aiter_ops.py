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

rms_no_var_16bit_only = lambda x, w, e, v: v is None and x.dtype in (
    torch.float16,
    torch.bfloat16,
)
"""AITER rms_norm only supports float16 and bfloat16 acts and no var_size override."""


@ir.ops.rms_norm.register_impl(
    "aiter", supports_args=rms_no_var_16bit_only, supported=AITER_SUPPORTED
)
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    assert variance_size is None
    assert x.dtype in (torch.float16, torch.bfloat16)
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
