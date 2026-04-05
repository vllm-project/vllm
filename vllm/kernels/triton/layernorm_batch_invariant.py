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
