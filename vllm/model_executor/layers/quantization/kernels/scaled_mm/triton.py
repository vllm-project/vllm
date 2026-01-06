# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
    triton_scaled_mm,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import ScaledMMLinearKernel, ScaledMMLinearLayerConfig


class TritonScaledMMLinearKernel(ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if current_platform.is_cuda_alike():
            return True, None
        return False, "Requires ROCm or CUDA."

    @classmethod
    def can_implement(cls, c: ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if not c.input_symmetric:
            return False, "Only symmetric input is supported."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = getattr(layer, self.w_q_name)
        replace_parameter(
            layer,
            self.w_q_name,
            torch.nn.Parameter(weight.t().data, requires_grad=False),
        )

        # INPUT SCALE
        if self.config.is_static_input_scheme:
            input_scale = getattr(layer, self.i_s_name)
            replace_parameter(
                layer,
                self.i_s_name,
                torch.nn.Parameter(input_scale.max(), requires_grad=False),
            )
            setattr(layer, self.i_zp_name, None)
        else:
            setattr(layer, self.i_s_name, None)
            setattr(layer, self.i_zp_name, None)

        setattr(layer, self.azp_adj_name, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, i_s, i_zp, azp_adj = self._get_weight_params(layer)

        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(), i_s, i_zp, symmetric=True
        )

        assert x_zp is None, "Triton kernel only supports symmetric quantization"

        return triton_scaled_mm(
            x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=x.dtype, bias=bias
        )
