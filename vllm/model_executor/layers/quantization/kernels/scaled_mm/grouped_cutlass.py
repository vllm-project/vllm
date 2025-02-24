# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.platforms import current_platform

from .GroupedMMLinearKernel import (GroupedMMLinearKernel,
                                    GroupedMMLinearLayerConfig)


class CutlassGroupMMLinearKernel(GroupedMMLinearKernel):

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def can_implement(
            cls, c: GroupedMMLinearLayerConfig) -> Tuple[bool, Optional[str]]:

        if (not current_platform.is_cuda() and not current_platform.is_cpu()):
            return False, "CutlassScaledMM requires running on CUDA or CPU."

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # WEIGHT
        # Cutlass kernels need transposed weight.
        weight = getattr(layer, self.w_q_name)
        replace_parameter(
            layer, self.w_q_name,
            torch.nn.Parameter(weight.t().data, requires_grad=False))

        # WEIGHT SCALE
        # Cutlass kernels support only per-tensor and per-channel.
        # If we have a fused module (QKV, MLP) with per tensor scales (thus N
        # scales being passed to the kernel), convert to the per-channel case.
        # is_fused_module = len(layer.logical_widths) > 1
        # weight_scale = getattr(layer, self.w_s_name)
        # if is_fused_module and not self.config.is_per_out_ch:
        #     weight_scale = convert_to_channelwise(weight_scale,
        #                                           layer.logical_widths)
        # if is_fused_module and not self.config.is_per_act_token:
        #     input_scale = convert_to_channelwise(weight_scale,
        #                                           layer.logical_widths)
        # replace_parameter(
        #     layer, self.w_s_name,
        #     torch.nn.Parameter(weight_scale.data, requires_grad=False))

        # INPUT SCALE
        if self.config.is_static_input_scheme:
            input_scale = getattr(layer, self.i_s_name)

            replace_parameter(
                layer, self.i_s_name,
                torch.nn.Parameter(input_scale.max(), requires_grad=False))

        else:
            setattr(layer, self.i_s_name, None)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        w_q, w_s, i_s = self._get_weight_params(layer)

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        x_q, x_s, x_zp = ops.scaled_int8_quant(x, i_s, None, symmetric=True)

        return ops.cutlass_grouped_mm(x_q,
                                      w_q,
                                      scale_a=x_s,
                                      scale_b=w_s,
                                      out_dtype=x.dtype,
                                      bias=bias)
