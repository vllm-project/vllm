# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from torch import Tensor

from vllm import ir
from vllm.platforms import current_platform

current_platform.import_kernels()

CUDA_ALIKE = current_platform.is_cuda_alike()
"""Most kernels in this file are supported on all CUDA-alike platforms."""


mixer2_rms_norm_gated_has_weight = (
    lambda x, gate, weight, epsilon, group_size=None: weight is not None
)
"""Triton gated RMSNorm kernel requires a weight tensor."""


@ir.ops.mixer2_rms_norm_gated.register_impl(
    "triton", supports_args=mixer2_rms_norm_gated_has_weight, supported=CUDA_ALIKE
)
def mixer2_rms_norm_gated(
    x: Tensor,
    gate: Tensor,
    weight: Tensor | None,
    epsilon: float,
    group_size: int | None = None,
) -> Tensor:
    assert weight is not None
    from vllm.model_executor.layers.mamba.ops.layernorm_gated import rms_norm_gated

    return rms_norm_gated(
        x,
        weight,
        bias=None,
        z=gate,
        eps=epsilon,
        group_size=group_size,
        norm_before_gate=False,
    )
