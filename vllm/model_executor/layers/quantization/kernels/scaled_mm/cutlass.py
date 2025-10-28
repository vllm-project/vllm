# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from collections.abc import Callable

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import ScaledMMLinearKernel, ScaledMMLinearLayerConfig


def cutlass_w8a8_scaled_mm(
    *,
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: torch.dtype,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor,
    output_shape: list,
) -> torch.Tensor:
    # Fused GEMM_DQ
    output = ops.cutlass_scaled_mm(
        A, B, out_dtype=out_dtype, scale_a=As, scale_b=Bs, bias=bias
    )
    return output.view(*output_shape)


def process_weights_after_loading(
    config: ScaledMMLinearLayerConfig,
    layer: torch.nn.Module,
    w_q_name: str,
    w_s_name: str,
    i_s_name: str,
    i_zp_name: str,
    azp_adj_name: str,
):
    # WEIGHT
    # Cutlass kernels need transposed weight.
    weight = getattr(layer, w_q_name)
    replace_parameter(
        layer,
        w_q_name,
        torch.nn.Parameter(weight.t().data, requires_grad=False),
    )

    # WEIGHT SCALE
    # Cutlass kernels support only per-tensor and per-channel.
    # If we have a fused module (QKV, MLP) with per tensor scales (thus N
    # scales being passed to the kernel), convert to the per-channel case.
    is_fused_module = len(layer.logical_widths) > 1
    weight_scale = getattr(layer, w_s_name)
    if is_fused_module and not config.is_channelwise:
        weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
    replace_parameter(
        layer,
        w_s_name,
        torch.nn.Parameter(weight_scale.data, requires_grad=False),
    )

    # INPUT SCALE
    if config.is_static_input_scheme:
        input_scale = getattr(layer, i_s_name)

        if config.input_symmetric:
            replace_parameter(
                layer,
                i_s_name,
                torch.nn.Parameter(input_scale.max(), requires_grad=False),
            )
            setattr(layer, i_zp_name, None)
        else:
            input_zero_point = getattr(layer, i_zp_name)

            # reconstruct the ranges
            int8_traits = torch.iinfo(torch.int8)
            azps = input_zero_point.to(dtype=torch.int32)
            range_max = (input_scale * (int8_traits.max - azps)).max()
            range_min = (input_scale * (int8_traits.min - azps)).min()

            scale = (range_max - range_min) / (int8_traits.max - int8_traits.min)
            replace_parameter(
                layer, i_s_name, torch.nn.Parameter(scale, requires_grad=False)
            )

            # AZP loaded as int8 but used as int32
            azp = (int8_traits.min - range_min / scale).to(dtype=torch.int32)
            replace_parameter(
                layer, i_zp_name, torch.nn.Parameter(azp, requires_grad=False)
            )

    else:
        setattr(layer, i_s_name, None)
        setattr(layer, i_zp_name, None)

    # azp_adj is the AZP adjustment term, used to account for weights.
    # It does not depend on scales or azp, so it is the same for
    # static and dynamic quantization.
    # For more details, see csrc/quantization/w8a8/cutlass/Epilogues.md
    # https://github.com/vllm-project/vllm/blob/main/csrc/quantization/w8a8/cutlass/Epilogues.md
    if not config.input_symmetric:
        weight = getattr(layer, w_q_name)
        azp_adj = weight.sum(dim=0, keepdim=True, dtype=torch.int32)
        if config.is_static_input_scheme:
            # cutlass_w8a8 requires azp to be folded into azp_adj
            # in the per-tensor case
            azp_adj = getattr(layer, i_zp_name) * azp_adj
        setattr(
            layer,
            azp_adj_name,
            torch.nn.Parameter(azp_adj, requires_grad=False),
        )
    else:
        setattr(layer, azp_adj_name, None)


class CutlassScaledMMLinearKernel(ScaledMMLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def can_implement(cls, c: ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "CutlassScaledMM requires running on CUDA."

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _, param_names = self.layer_mapping_function(layer)

        process_weights_after_loading(self.config, layer, *param_names)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        (w_q, w_s, i_s, i_zp, azp_adj), _ = self.layer_mapping_function(layer)

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        symmetric = azp_adj is None
        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(), i_s, i_zp, symmetric=symmetric
        )

        if x_zp is not None:
            # Currently, static is always per-tensor and dynamic is per-token
            static = i_zp is not None
            azp = None if static else x_zp
            return ops.cutlass_scaled_mm_azp(
                x_q,
                w_q,
                scale_a=x_s,
                scale_b=w_s,
                out_dtype=x.dtype,
                azp_adj=azp_adj,
                azp=azp,
                bias=bias,
            )
        return ops.cutlass_scaled_mm(
            x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=x.dtype, bias=bias
        )


class CutlassFP8ScaledMMLinearKernel(ScaledMMLinearKernel):
    def __init__(
        self, c: ScaledMMLinearLayerConfig, layer_mapping_function: Callable
    ) -> None:
        self.quant_fp8 = QuantFP8(
            static=c.is_static_input_scheme,
            group_shape=GroupShape.PER_TENSOR,
            num_token_padding=None,
        )
        super().__init__(c, layer_mapping_function)

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    @classmethod
    def can_implement(cls, c: ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return (
                False,
                "CutlassFP8ScaledMMLinearKernel is supported "
                + "on CUDA platforms Only.",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        # ops.scaled_fp8_quant supports both dynamic and static quant.
        #   If dynamic, layer.input_scale is None and x_scale computed from x.
        #   If static, layer.input_scale is scalar and x_scale is input_scale.
        (w, w_s, x_s), _ = self.layer_mapping_function(layer)
        # View input as 2D matrix for fp8 methods
        x_2d = x.view(-1, x.shape[-1])

        out_dtype = self.config.out_dtype
        out_dtype = x.dtype if out_dtype is None else out_dtype
        # If input not quantized
        # TODO(luka) remove this path if not used anymore
        x_2d_q = x_2d
        if x.dtype != current_platform.fp8_dtype():
            x_2d_q, x_s = self.quant_fp8(
                x_2d,
                x_s,
            )

        output_shape = [*x_2d_q.shape[:-1], w.shape[1]]

        return cutlass_w8a8_scaled_mm(
            A=x_2d_q,
            B=w,
            out_dtype=out_dtype,
            As=x_s,
            Bs=w_s,
            bias=bias,
            output_shape=output_shape,
        )
