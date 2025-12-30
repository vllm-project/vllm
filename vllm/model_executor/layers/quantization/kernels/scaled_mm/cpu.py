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

from .ScaledMMLinearKernel import ScaledMMLinearKernel, ScaledMMLinearLayerConfig


class CPUScaledMMLinearKernel(ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cpu():
            return False, "Requires CPU."
        return True, None

    @classmethod
    def can_implement(cls, c: ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = getattr(layer, self.w_q_name)
        dtype = weight.dtype
        N, K = weight.size()
        if (
            current_platform.get_cpu_architecture() == CpuArchEnum.X86
            and envs.VLLM_CPU_SGL_KERNEL
            and self.config.input_symmetric
            and check_cpu_sgl_kernel(N, K, dtype)
        ):
            self.linear_method = self._apply_weights_sgl
            self.process_weights_for_sgl(layer)
        else:
            self.linear_method = self._apply_weights_onednn
            self.process_weights_for_onednn(layer)

    def process_weights_for_onednn(self, layer: torch.nn.Module) -> None:
        # WEIGHT
        # Transpose to [K, N] for convenience
        weight = getattr(layer, self.w_q_name)
        replace_parameter(
            layer,
            self.w_q_name,
            torch.nn.Parameter(weight.t().data, requires_grad=False),
        )

        # WEIGHT SCALE
        # oneDNN kernels support only per-tensor and per-channel.
        # If we have a fused module (QKV, MLP) with per tensor scales (thus N
        # scales being passed to the kernel), convert to the per-channel case.
        is_fused_module = len(layer.logical_widths) > 1
        weight_scale = getattr(layer, self.w_s_name)
        if is_fused_module and not self.config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        replace_parameter(
            layer,
            self.w_s_name,
            torch.nn.Parameter(weight_scale.data, requires_grad=False),
        )

        # INPUT SCALE
        if self.config.is_static_input_scheme:
            input_scale = getattr(layer, self.i_s_name)

            if self.config.input_symmetric:
                replace_parameter(
                    layer,
                    self.i_s_name,
                    torch.nn.Parameter(input_scale.max(), requires_grad=False),
                )
                setattr(layer, self.i_zp_name, None)
            else:
                input_zero_point = getattr(layer, self.i_zp_name)

                # reconstruct the ranges
                int8_traits = torch.iinfo(torch.int8)
                azps = input_zero_point.to(dtype=torch.int32)
                range_max = (input_scale * (int8_traits.max - azps)).max()
                range_min = (input_scale * (int8_traits.min - azps)).min()

                scale = (range_max - range_min) / (int8_traits.max - int8_traits.min)
                replace_parameter(
                    layer, self.i_s_name, torch.nn.Parameter(scale, requires_grad=False)
                )

                azp = (
                    (int8_traits.min - range_min / scale).round().to(dtype=torch.int32)
                )
                replace_parameter(
                    layer, self.i_zp_name, torch.nn.Parameter(azp, requires_grad=False)
                )

        else:
            setattr(layer, self.i_s_name, None)
            setattr(layer, self.i_zp_name, None)

        # Different from cutlass, oneDNN kernels only need the AZP adjustment
        # term for dynamic quantization. And s_b should be folded into the
        # term. Such as:
        # s_a * s_b * [(A - zp_a)B] + bias =
        # s_a * (s_b * AB) - s_a * s_b * zp_a * B + bias =
        # s_a * GEMM_output - s_a * zp_a * adj + bias
        if not (self.config.input_symmetric and self.config.is_static_input_scheme):
            weight = getattr(layer, self.w_q_name)
            weight_scale = getattr(layer, self.w_s_name)
            azp_adj = weight.sum(dim=0, keepdim=True, dtype=torch.float32)
            azp_adj = azp_adj * weight_scale.squeeze()
            setattr(
                layer,
                self.azp_adj_name,
                torch.nn.Parameter(azp_adj, requires_grad=False),
            )
        else:
            setattr(layer, self.azp_adj_name, None)

        weight = getattr(layer, self.w_q_name)
        self.dnnl_handler = ops.create_onednn_scaled_mm(
            weight,
            getattr(layer, self.w_s_name),
            torch.get_default_dtype(),
            getattr(layer, self.i_s_name) is None,
            not self.config.input_symmetric,
            32,
        )
        # weight is prepacked and maintained by the dnnl_handler,
        # release the original weight
        setattr(layer, self.w_q_name, None)
        del weight

    def process_weights_for_sgl(self, layer: torch.nn.Module) -> None:
        # WEIGHT
        weight = getattr(layer, self.w_q_name)
        packed_weight = torch.ops._C.convert_weight_packed(weight)
        replace_parameter(
            layer, self.w_q_name, torch.nn.Parameter(packed_weight, requires_grad=False)
        )

        if layer.bias is not None:
            bias = layer.bias
            layer.register_parameter(
                "bias_fp32", torch.nn.Parameter(bias.float().data, requires_grad=False)
            )

        # WEIGHT SCALE
        # CPU SGL kernels only support per-channel.
        # For per-tensor quant, convert to the per-channel case.
        weight_scale = getattr(layer, self.w_s_name)
        if not self.config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        replace_parameter(
            layer,
            self.w_s_name,
            torch.nn.Parameter(weight_scale.data, requires_grad=False),
        )

        setattr(layer, self.i_s_name, None)
        setattr(layer, self.i_zp_name, None)
        setattr(layer, self.azp_adj_name, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
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
        w_q, w_s, i_s, i_zp, azp_adj = self._get_weight_params(layer)

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        x_q, x_s, x_zp = ops.onednn_scaled_int8_quant(
            x, i_s, i_zp, self.config.input_symmetric
        )

        m = x.size(0)
        n = self.dnnl_handler.n
        out = torch.empty((m, n), dtype=x.dtype)
        ops.onednn_scaled_mm(self.dnnl_handler, x_q, out, x_s, x_zp, azp_adj, bias)

        return out

    def _apply_weights_sgl(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, _, _, _ = self._get_weight_params(layer)
        return torch.ops._C.int8_scaled_mm_with_quant(
            x,
            w_q,
            w_s,
            layer.bias_fp32 if bias is not None else None,
            x.dtype,
            True,
        )
