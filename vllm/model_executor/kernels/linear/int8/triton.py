# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
    triton_scaled_mm,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    prepare_int8_scaled_mm_weights,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .base import (
    Int8LinearKernel,
    Int8LinearLayerConfig,
)


class TritonInt8LinearKernel(Int8LinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if current_platform.is_cuda_alike() or current_platform.is_xpu():
            return True, None
        return False, "requires ROCm, CUDA or XPU."

    @classmethod
    def can_implement(cls, c: Int8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        prepare_int8_scaled_mm_weights(layer, self.config, self.layer_param_names)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)

        symmetric = azp_adj is None
        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(), i_s, i_zp, symmetric=symmetric
        )

        # A static per-tensor azp is already folded into azp_adj.
        azp = x_zp if i_zp is None else None
        return torch.ops.vllm.int8_scaled_mm_func(
            x_q, w_q, x_s, w_s, x.dtype, bias, azp_adj, azp
        )


def _int8_scaled_mm_func(
    qx: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
    azp_adj: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
) -> torch.Tensor:
    return triton_scaled_mm(
        qx,
        weight,
        scale_a=x_scale,
        scale_b=weight_scale,
        out_dtype=output_dtype,
        bias=bias,
        azp_adj=azp_adj,
        azp=azp,
    )


def _int8_scaled_mm_fake(
    qx: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
    azp_adj: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty(
        (qx.size(0), weight.size(1)), dtype=output_dtype, device=qx.device
    )


direct_register_custom_op(
    "int8_scaled_mm_func",
    _int8_scaled_mm_func,
    fake_impl=_int8_scaled_mm_fake,
)
