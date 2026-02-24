# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
    unpack_quantized_values_into_int32,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

_CPUWNA16_SUPPORTED_QUANT_TYPES = (scalar_types.uint4, scalar_types.uint4b8)


class CPUWNA16LinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cpu():
            return False, "CPUWNA16 only supported on CPU"

        if c.weight_type not in _CPUWNA16_SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                "CPUWNA16, supported types are: "
                f"{_CPUWNA16_SUPPORTED_QUANT_TYPES}",
            )

        if c.group_size != -1 and c.group_size % 2 != 0:
            return (
                False,
                f"Group size ({c.group_size}) not supported by "
                "CPUWNA16, supported group sizes are multiples of 2",
            )

        if c.partition_weight_shape[0] % 32 != 0:
            return (
                False,
                f"Input size ({c.partition_weight_shape[0]}) not supported by "
                "CPUWNA16, supported sizes are multiples of 32",
            )

        if c.partition_weight_shape[1] % 32 != 0:
            return (
                False,
                f"Output size ({c.partition_weight_shape[1]}) not supported by "
                "CPUWNA16, supported sizes are multiples of 32",
            )

        return True, None

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale`  is: {input_dim = 0, output_dim = 1}
    #  `weight_zp`     is: {input_dim = 0, output_dim = 1, packed_dim = 1}
    def _process_gptq_weights(self, layer: torch.nn.Module):
        packed_weight = layer.qweight.data
        bits = self.config.weight_type.mantissa
        pack_factor = 32 // bits
        p_w_k, p_w_n = packed_weight.size()
        input_size = p_w_k * pack_factor
        output_size = p_w_n
        isa_hint = _get_isa_hint(layer.scales.dtype)
        layer.isa_hint = isa_hint

        layer.qzeros = None
        if not self.config.has_g_idx:
            layer.g_idx = None

        # convert input dim packed to output dim packed
        weight = unpack_quantized_values_into_int32(
            packed_weight, self.config.weight_type, 1
        ).view(p_w_k, p_w_n, pack_factor)
        weight = weight.permute(0, 2, 1).reshape(input_size, output_size).contiguous()
        weight = pack_quantized_values_into_int32(weight, self.config.weight_type, 1)
        # make 16 output channel as a block and transpose to the make
        # the block contigous
        weight = (
            weight.view(input_size, -1, 16 // pack_factor)
            .permute(1, 0, 2)
            .reshape(-1, input_size * 16 // pack_factor)
            .contiguous()
        )
        layer.qweight.data = weight

    def process_weights_after_loading(self, layer: torch.nn.Module):
        if not self.config.zero_points:
            # GPTQ
            self._process_gptq_weights(layer)
        else:
            # AWQ
            raise NotImplementedError("AWQ is not supported in CPUWNA16LinearKernel")

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = ops.cpu_gemm_wna16(
            input=x,
            q_weight=layer.qweight,
            scales=layer.scales,
            zeros=layer.qzeros,
            g_idx=layer.g_idx,
            bias=bias,
            pack_factor=8,  # 32 // 4
            isa_hint=layer.isa_hint,
        )
        return x


def _get_isa_hint(dtype: torch.dtype) -> str:
    supports_amx = torch._C._cpu._is_amx_tile_supported()
    if supports_amx and dtype in (torch.bfloat16,):
        return "amx"
    else:
        return "vec"
