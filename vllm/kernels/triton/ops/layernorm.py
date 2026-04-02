# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import Tensor

from vllm import ir
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

CUDA_ALIKE = current_platform.is_cuda_alike()


def _rms_norm_gated_triton_impl(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    z: Tensor | None,
    epsilon: float,
    group_size: int | None,
    norm_before_gate: bool,
    activation: str,
) -> Tensor:
    from vllm.model_executor.layers.fla.ops.layernorm_guard import layer_norm_fwd

    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    if z is not None:
        assert z.shape == x_shape_og
        z = z.reshape(-1, z.shape[-1])
        if z.stride(-1) != 1:
            z = z.contiguous()
    weight = weight.contiguous()

    if bias is not None:
        bias = bias.contiguous()

    y, _, _ = layer_norm_fwd(
        x,
        weight,
        bias,
        epsilon,
        z=z,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=True,
        activation=activation,
    )
    return y.reshape(x_shape_og)


def _rms_norm_gated_triton_fake(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    z: Tensor | None,
    epsilon: float,
    group_size: int | None,
    norm_before_gate: bool,
    activation: str,
) -> Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="rms_norm_gated_triton",
    op_func=_rms_norm_gated_triton_impl,
    mutates_args=[],
    fake_impl=_rms_norm_gated_triton_fake,
)


@ir.ops.rms_norm_gated.register_impl("triton", supported=CUDA_ALIKE)
def rms_norm_gated(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    z: Tensor | None,
    epsilon: float,
    group_size: int | None = None,
    norm_before_gate: bool = False,
    activation: str = "swish",
) -> Tensor:
    return torch.ops.vllm.rms_norm_gated_triton(
        x, weight, bias, z, epsilon, group_size, norm_before_gate, activation
    )
