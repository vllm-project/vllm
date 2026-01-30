# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    CUTLASS_BLOCK_FP8_SUPPORTED,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .BlockScaledMMKernel import Fp8BlockScaledMMKernel
from .triton import TritonBlockScaledMMKernel


# We need to pass in the is_hopper flag as argument because the function
# current_platform.is_device_capability() is not supported by Torch compiler.
def cutlass_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    return ops.cutlass_scaled_mm(
        A,
        B.T,
        out_dtype=output_dtype,
        scale_a=As,
        scale_b=Bs.T,
    )


def _padded_cutlass(
    qx: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    pad_multiple = 4
    dim = qx.shape[0]
    padded = (
        dim if dim % pad_multiple == 0 else dim + pad_multiple - (dim % pad_multiple)
    )

    has_pad = padded > dim

    if has_pad:
        padded_shape = [padded, *qx.shape[1:]]
        padded_qx = torch.zeros(padded_shape, device=qx.device, dtype=qx.dtype)
        padded_qx[0 : qx.shape[0], ...].copy_(qx)

        padded_x_scale_shape = [*x_scale.shape[1:], padded]
        padded_x_scale = torch.ones(
            padded_x_scale_shape, device=x_scale.device, dtype=x_scale.dtype
        ).permute(-1, -2)
        padded_x_scale[0 : x_scale.shape[0], ...].copy_(x_scale)

        output = cutlass_scaled_mm(
            padded_qx, weight, padded_x_scale, weight_scale, block_size, output_dtype
        )
        return output[0 : qx.shape[0], ...]
    else:
        return cutlass_scaled_mm(
            qx, weight, x_scale, weight_scale, block_size, output_dtype
        )


def _padded_cutlass_fake(
    qx: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(
        (qx.size(0), weight.size(0)), dtype=output_dtype, device=qx.device
    )


direct_register_custom_op(
    "padded_cutlass",
    _padded_cutlass,
    fake_impl=_padded_cutlass_fake,
)


class CutlassBlockScaledMMKernel(Fp8BlockScaledMMKernel):
    is_hopper: bool = current_platform.is_device_capability(90)

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not CUTLASS_BLOCK_FP8_SUPPORTED:
            return (
                False,
                f"The device compute capability of \
                {compute_capability} is not supported.",
            )

    @classmethod
    def ordered_fallback_kernels(cls) -> list[type["Fp8BlockScaledMMKernel"]]:
        return [TritonBlockScaledMMKernel]

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if self.is_hopper:
            return torch.ops.vllm.padded_cutlass(
                A,
                B,
                As,
                Bs,
                list(self.weight_group_shape),
                out_dtype,
            )
        else:
            return cutlass_scaled_mm(
                A,
                B,
                As,
                Bs,
                list(self.weight_group_shape),
                out_dtype,
            )
