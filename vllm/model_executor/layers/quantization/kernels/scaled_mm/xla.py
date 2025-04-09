# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Optional, Tuple

import torch
from functorch.experimental.control_flow import cond  # noqa: F401

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise)
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (ScaledMMLinearKernel,
                                   ScaledMMLinearLayerConfig)


class XLAScaledMMLinearKernel(ScaledMMLinearKernel):

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            "TPU platform does have a concept of compute capability, "
            "this method should not be called.")

    @classmethod
    def can_implement(
            cls, c: ScaledMMLinearLayerConfig) -> Tuple[bool, Optional[str]]:

        if not current_platform.is_tpu():
            return False, "ScaledMMXLA requires running on TPU."

        if c.is_static_input_scheme:
            return False, "ScaledMMXLA requires dynamic activation scales."

        if not c.input_symmetric:
            return False, "ScaledMMXLA requires symmetric activation scales."

        if not c.is_channelwise:
            return False, "ScaledMMXLA requires channelwise weight scales"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # WEIGHT
        # [out, in] (different than cutlass_scaled_mm)
        weight = getattr(layer, self.w_q_name)
        replace_parameter(layer, self.w_q_name,
                          torch.nn.Parameter(weight.data, requires_grad=False))

        # WEIGHT SCALE
        # XLA kernels support only per-tensor and per-channel.
        # If we have a fused module (QKV, MLP) with per tensor scales (thus N
        # scales being passed to the kernel), convert to the per-channel case.
        is_fused_module = len(layer.logical_widths) > 1
        weight_scale = getattr(layer, self.w_s_name)
        if is_fused_module and not self.config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale,
                                                  layer.logical_widths)

        # [out_channel,] (different than cutlass_scaled_mm)
        weight_scale = weight_scale.squeeze(-1)
        replace_parameter(
            layer, self.w_s_name,
            torch.nn.Parameter(weight_scale.data, requires_grad=False))

        # Only support symmetric dynamic activation quantization.
        setattr(layer, self.i_s_name, None)
        setattr(layer, self.i_zp_name, None)
        setattr(layer, self.azp_adj_name, None)

        # Filter warning for cond usage in apply_weights. It is okay
        # to specialize the graph since bias is not dynamic.
        warnings.filterwarnings(
            "ignore",
            message=
            "Pred is a Python constant. When used with torch.cond, it specializes on one of the branches."  # noqa: E501
        )

    def no_add_bias(self, x: torch.Tensor, bias: Optional[torch.Tensor]):
        return x

    def add_bias(self, x: torch.Tensor, bias: Optional[torch.Tensor]):
        return x + bias

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        w_q, w_s, _, _, _ = self._get_weight_params(layer)

        import torch_xla.experimental.xla_quantized_matmul  # noqa: F401
        out = torch.ops.xla.quantized_matmul(x,
                                             w_q,
                                             w_s,
                                             zero_point=None,
                                             block_size=-1,
                                             int4_weight=False,
                                             quantize_activation=True)
        # `quantized_matmul` output is fp32, cast it down to bf16 for perf
        out = out.to(x.dtype)
        # Explicitly capture control flow to make dynamo happy.
        # https://pytorch.org/docs/main/generated/exportdb/index.html#cond-branch-class-method # noqa: E501
        return cond(bias is None, self.no_add_bias, self.add_bias, [out, bias])
