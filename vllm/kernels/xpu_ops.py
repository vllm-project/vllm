# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import Tensor

from vllm import ir
from vllm.platforms import current_platform

current_platform.import_kernels()


def is_xpu_kernels_found() -> bool:
    from importlib.util import find_spec

    return find_spec("vllm_xpu_kernels") is not None


XPU_KERNELS_SUPPORTED = is_xpu_kernels_found()
"""Kernels in this file are supported if vLLM XPU kernels are installed."""

rms_no_var = lambda x, weight, epsilon, variance_size=None: variance_size is None


@ir.ops.rms_norm.register_impl(
    "xpu_kernels", supports_args=rms_no_var, supported=XPU_KERNELS_SUPPORTED
)
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    if weight is None:
        # Kernel requires weight tensor, pass ones
        weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    assert variance_size is None
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    torch.ops._C.rms_norm(output, x, weight, epsilon)
    return output
