# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (ScaledMMLinearKernel,
                                   ScaledMMLinearLayerConfig)


class CutlassScaledMMLinearKernel(ScaledMMLinearKernel):

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def is_supported(
        cls,
        compute_capability: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        # Cutlass is also supported on CPU
        if current_platform.is_cpu():
            return True, ""

        if not current_platform.is_cuda():
            return False, "CutlassScaledMM requires running on CUDA or CPU."

        # Defer to compute-capability-based support determination
        return super().is_supported(compute_capability)

    @classmethod
    def can_implement(
            cls, c: ScaledMMLinearLayerConfig) -> Tuple[bool, Optional[str]]:
        # All schemes supported
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # WEIGHT
        # Cutlass kernels need transposed weight.
        weight_param = getattr(layer, self.w_q_name)
        self.replace_parameter(layer, self.w_q_name, weight_param.t())

        # WEIGHT SCALE
        # Cutlass kernels support only per-tensor and per-channel.
        w_scale_param = getattr(layer, self.w_s_name)
        w_scale_param = self.maybe_unfuse_weight_scale(layer, w_scale_param)
        self.replace_parameter(layer, self.w_s_name, w_scale_param)

        # INPUT SCALE
        if self.config.is_static_input_scheme:
            input_scale_param = getattr(layer, self.i_s_name)

            if self.config.input_symmetric:
                self.replace_parameter(layer, self.i_s_name,
                                       input_scale_param.max())
                setattr(layer, self.i_zp_name, None)
            else:
                input_zero_point = getattr(layer, self.i_zp_name)

                i_scale, i_zp = self.fuse_asymmetric_params(
                    input_scale_param, input_zero_point)
                self.replace_parameter(layer, self.i_s_name, i_scale)
                self.replace_parameter(layer, self.i_zp_name, i_zp)
        else:
            setattr(layer, self.i_s_name, None)
            setattr(layer, self.i_zp_name, None)

        # azp_adj is the AZP adjustment term, used to account for weights.
        # It does not depend on scales or azp, so it is the same for
        # static and dynamic quantization.
        # For more details, see csrc/quantization/cutlass_w8a8/Epilogues.md
        # https://github.com/vllm-project/vllm/blob/8d59dbb00044a588cab96bcdc028006ed922eb06/csrc/quantization/cutlass_w8a8/Epilogues.md
        if not self.config.input_symmetric:
            weight = getattr(layer, self.w_q_name)
            azp_adj = weight.sum(dim=0, keepdim=True, dtype=torch.int32)
            if self.config.is_static_input_scheme:
                # cutlass_w8a8 requires azp to be folded into azp_adj
                # in the per-tensor case
                azp_adj = getattr(layer, self.i_zp_name) * azp_adj
            setattr(layer, self.azp_adj_name,
                    torch.nn.Parameter(azp_adj, requires_grad=False))
        else:
            setattr(layer, self.azp_adj_name, None)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        w_q, w_s, i_s, i_zp, azp_adj = self._get_weight_params(layer)

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        sym = self.config.input_symmetric
        x_q, x_s, x_zp = ops.scaled_int8_quant(x, i_s, i_zp, symmetric=sym)

        if x_zp is not None:
            # Currently, static is always per-tensor and dynamic is per-token
            static = i_zp is not None
            azp = None if static else x_zp
            return ops.cutlass_scaled_mm_azp(x_q,
                                             w_q,
                                             scale_a=x_s,
                                             scale_b=w_s,
                                             out_dtype=x.dtype,
                                             azp_adj=azp_adj,
                                             azp=azp,
                                             bias=bias)
        return ops.cutlass_scaled_mm(x_q,
                                     w_q,
                                     scale_a=x_s,
                                     scale_b=w_s,
                                     out_dtype=x.dtype,
                                     bias=bias)
