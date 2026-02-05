# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.model_executor.layers.utils import check_cpu_sgl_kernel
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum

from ..base import Int8Params
from .ScaledMMLinearKernel import (
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)


class CPUInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cpu():
            return False, "requires CPU."
        return True, None

    @classmethod
    def can_implement(
        cls, config: Int8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        params = self._get_layer_params(layer)
        weight = params.weight
        dtype = weight.dtype
        N, K = weight.size()
        if (
            current_platform.get_cpu_architecture() == CpuArchEnum.X86
            and envs.VLLM_CPU_SGL_KERNEL
            and self.config.input_symmetric
            and check_cpu_sgl_kernel(N, K, dtype)
        ):
            self.linear_method = self._apply_weights_sgl
            self.process_weights_for_sgl(layer, params)
        else:
            self.linear_method = self._apply_weights_onednn
            self.process_weights_for_onednn(layer, params)

    def process_weights_for_onednn(
        self, layer: torch.nn.Module, params: Int8Params
    ) -> None:
        # WEIGHT
        # Transpose to [K, N] for convenience
        weight = params.weight
        replace_parameter(
            layer,
            params.WEIGHT,
            torch.nn.Parameter(weight.t().data, requires_grad=False),
        )

        # WEIGHT SCALE
        # oneDNN kernels support only per-tensor and per-channel.
        # If we have a fused module (QKV, MLP) with per tensor scales (thus N
        # scales being passed to the kernel), convert to the per-channel case.
        is_fused_module = len(layer.logical_widths) > 1
        weight_scale = params.weight_scale
        if is_fused_module and not self.config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        replace_parameter(
            layer,
            params.WEIGHT_SCALE,
            torch.nn.Parameter(weight_scale.data, requires_grad=False),
        )

        # INPUT SCALE
        if self.config.is_static_input_scheme:
            input_scale = params.input_scale

            assert input_scale is not None

            if self.config.input_symmetric:
                replace_parameter(
                    layer,
                    params.INPUT_SCALE,
                    torch.nn.Parameter(input_scale.max(), requires_grad=False),
                )
            else:
                input_zero_point = params.input_zero_point

                assert input_zero_point is not None
                # reconstruct the ranges
                int8_traits = torch.iinfo(torch.int8)
                azps = input_zero_point.to(dtype=torch.int32)
                range_max = (input_scale * (int8_traits.max - azps)).max()
                range_min = (input_scale * (int8_traits.min - azps)).min()

                scale = (range_max - range_min) / (int8_traits.max - int8_traits.min)
                replace_parameter(
                    layer,
                    params.INPUT_SCALE,
                    torch.nn.Parameter(scale, requires_grad=False),
                )

                azp = (
                    (int8_traits.min - range_min / scale).round().to(dtype=torch.int32)
                )
                replace_parameter(
                    layer,
                    params.INPUT_ZERO_POINT,
                    torch.nn.Parameter(azp, requires_grad=False),
                )

        # Different from cutlass, oneDNN kernels only need the AZP adjustment
        # term for dynamic quantization. And s_b should be folded into the
        # term. Such as:
        # s_a * s_b * [(A - zp_a)B] + bias =
        # s_a * (s_b * AB) - s_a * s_b * zp_a * B + bias =
        # s_a * GEMM_output - s_a * zp_a * adj + bias
        if not (self.config.input_symmetric and self.config.is_static_input_scheme):
            weight = getattr(layer, params.WEIGHT)
            weight_scale = getattr(layer, params.WEIGHT_SCALE)
            azp_adj = weight.sum(dim=0, keepdim=True, dtype=torch.float32)
            azp_adj = azp_adj * weight_scale.squeeze()
            setattr(
                layer,
                params.AZP_ADJ,
                torch.nn.Parameter(azp_adj, requires_grad=False),
            )

        weight = getattr(layer, params.WEIGHT)
        self.dnnl_handler = ops.create_onednn_scaled_mm(
            weight,
            getattr(layer, params.WEIGHT_SCALE),
            torch.get_default_dtype(),
            getattr(layer, params.INPUT_SCALE) is None,
            not self.config.input_symmetric,
            32,
        )
        # weight is prepacked and maintained by the dnnl_handler,
        # release the original weight
        setattr(layer, params.WEIGHT, None)
        del params

    def process_weights_for_sgl(
        self, layer: torch.nn.Module, params: Int8Params
    ) -> None:
        # WEIGHT
        packed_weight = torch.ops._C.convert_weight_packed(params.weight)
        replace_parameter(
            layer, params.WEIGHT, torch.nn.Parameter(packed_weight, requires_grad=False)
        )

        if layer.bias is not None:
            bias = layer.bias
            layer.register_parameter(
                "bias_fp32", torch.nn.Parameter(bias.float().data, requires_grad=False)
            )

        # WEIGHT SCALE
        # CPU SGL kernels only support per-channel.
        # For per-tensor quant, convert to the per-channel case.
        weight_scale = params.weight_scale
        if not self.config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        replace_parameter(
            layer,
            params.WEIGHT_SCALE,
            torch.nn.Parameter(weight_scale.data, requires_grad=False),
        )

        del params

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.linear_method(
            layer,
            x,
            bias,
        )

    def _apply_weights_onednn(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        params = self._get_layer_params(layer)

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        x_q, x_s, x_zp = ops.onednn_scaled_int8_quant(
            x, params.input_scale, params.input_zero_point, self.config.input_symmetric
        )

        m = x.size(0)
        n = self.dnnl_handler.n
        out = torch.empty((m, n), dtype=x.dtype)
        ops.onednn_scaled_mm(
            self.dnnl_handler, x_q, out, x_s, x_zp, params.azp_adj, bias
        )

        return out

    def _apply_weights_sgl(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        params = self._get_layer_params(layer)
        return torch.ops._C.int8_scaled_mm_with_quant(
            x,
            params.weight,
            params.weight_scale,
            layer.bias_fp32 if bias is not None else None,
            x.dtype,
            True,
        )
