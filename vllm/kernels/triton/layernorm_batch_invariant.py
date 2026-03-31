# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import Tensor

from vllm import ir
from vllm.triton_utils import HAS_TRITON

rms_norm_no_var = lambda x, weight, epsilon, variance_size=None: variance_size is None
"""Variance size override not supported"""


@ir.ops.rms_norm.register_impl(
    "triton_batch_invariant",
    supported=HAS_TRITON,
    supports_args=rms_norm_no_var,
    batch_invariant=True,
)
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    assert variance_size is None
    if weight is None:
        weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)

    # TODO move kernel here
    from vllm.model_executor.layers.batch_invariant import rms_norm_batch_invariant

    return rms_norm_batch_invariant(x, weight, epsilon)


rms_add_no_var_size = (
    lambda x, x_residual, weight, epsilon, variance_size=None: variance_size is None
)
"""vLLM Kernel does not support variance_size parameter."""


@ir.ops.fused_add_rms_norm.register_impl(
    "triton_batch_invariant",
    supported=HAS_TRITON,
    supports_args=rms_add_no_var_size,
    batch_invariant=True,
)
def fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    if weight is None:
        # Kernel requires weight tensor, pass ones
        weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)

    assert variance_size is None
    # TODO move kernel here
    from vllm.model_executor.layers.batch_invariant import rms_norm_batch_invariant

    return rms_norm_batch_invariant(x + x_residual, weight, epsilon), x + x_residual
