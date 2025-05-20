# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise)
from vllm.platforms import current_platform
from vllm import envs

from .ScaledMMLinearKernel import (ScaledMMLinearKernel,
                                   ScaledMMLinearLayerConfig)

class CPUScaledMMLinearKernel(ScaledMMLinearKernel):

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def can_implement(
            cls, c: ScaledMMLinearLayerConfig) -> tuple[bool, Optional[str]]:

        if not (current_platform.is_cpu() and envs.VLLM_CPU_SGL_KERNEL and c.input_symmetric):
            return False, "CPUScaledMM requires symmetric input quantization running on CPU with VLLM_CPU_SGL_KERNEL."

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # WEIGHT
        weight = getattr(layer, self.w_q_name)
        packed_weight = torch.ops._C.convert_weight_packed(weight)
        replace_parameter(
            layer, self.w_q_name,
            torch.nn.Parameter(packed_weight, requires_grad=False))

        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(layer.bias.to(torch.float32),
                        requires_grad=False) 

        # WEIGHT SCALE
        # CPU kernels only support per-channel.
        # If we have a fused module (QKV, MLP) with per tensor scales (thus N
        # scales being passed to the kernel), convert to the per-channel case.
        is_fused_module = len(layer.logical_widths) > 1
        weight_scale = getattr(layer, self.w_s_name)
        if is_fused_module and not self.config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale,
                                                  layer.logical_widths)
        replace_parameter(
            layer, self.w_s_name,
            torch.nn.Parameter(weight_scale.data, requires_grad=False))

        setattr(layer, self.i_s_name, None)
        setattr(layer, self.i_zp_name, None)
        setattr(layer, self.azp_adj_name, None)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        w_q, w_s, _, _, _ = self._get_weight_params(layer)
        return torch.ops._C.int8_scaled_mm_with_quant(
            x,
            w_q,
            w_s,
            bias,
            x.dtype,
            True,
        )
