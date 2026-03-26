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
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0} (marlin)
    #                  or: {input_dim = 1, output_dim = 0, packed_dim = 1} (CT)
    #  `weight_scale`  is: {input_dim = 0, output_dim = 1} (marlin)
    #                  or: {input_dim = 1, output_dim = 0} (CT)
    #  `weight_zp`     is: {input_dim = 0, output_dim = 1, packed_dim = 1} (marlin)
    #                  or: {input_dim = 1, output_dim = 0, packed_dim = 0} (CT)
    def _process_gptq_weights(self, layer: torch.nn.Module):
        # packed_weight = layer.qweight.data
        packed_weight = getattr(layer, self.w_q_name)
        assert packed_weight.input_dim == packed_weight.packed_dim
        is_ct_format = packed_weight.input_dim == 1
        if is_ct_format:
            packed_weight = packed_weight.t()
        bits = self.config.weight_type.mantissa
        pack_factor = 32 // bits
        p_w_k, _ = packed_weight.size()
        input_size = p_w_k * pack_factor
        isa_hint = _get_isa_hint(getattr(layer, self.w_s_name).dtype)
        layer.isa_hint = isa_hint

        # convert input dim packed to output dim packed
        weight = unpack_quantized_values_into_int32(
            packed_weight, self.config.weight_type, 0
        )
        weight = pack_quantized_values_into_int32(weight, self.config.weight_type, 1)
        # make 16 output channel as a block and transpose to the make
        # the block contiguous
        weight = (
            weight.view(input_size, -1, 16 // pack_factor)
            .permute(1, 0, 2)
            .reshape(-1, input_size * 16 // pack_factor)
            .contiguous()
        )
        getattr(layer, self.w_q_name).data = weight

        # transpose scale, zp for CT format
        if is_ct_format:
            scales = getattr(layer, self.w_s_name)
            scales.data = scales.t().contiguous()
            if self.config.zero_points:
                zp = getattr(layer, self.w_zp_name)
                zp.data = zp.t().contiguous()

    def process_weights_after_loading(self, layer: torch.nn.Module):
        if (not self.config.zero_points) and (self.w_zp_name is not None):
            setattr(layer, self.w_zp_name, None)

        if (not self.config.has_g_idx) and (self.w_gidx_name is not None):
            setattr(layer, self.w_gidx_name, None)

        w_input_dim = getattr(layer, self.w_q_name).input_dim
        w_pack_dim = getattr(layer, self.w_q_name).packed_dim
        quant_method = "gptq" if w_pack_dim == w_input_dim else "awq"

        if quant_method == "gptq":
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
        w_q, w_s, w_zp, w_gidx = self._get_weight_params(layer)
        x = ops.cpu_gemm_wna16(
            input=x,
            q_weight=w_q,
            scales=w_s,
            zeros=w_zp,
            g_idx=w_gidx,
            bias=bias,
            pack_factor=8,  # 32 // 4
            isa_hint=layer.isa_hint,
        )
        return x


def _get_isa_hint(dtype: torch.dtype) -> str:
    supports_amx = torch.cpu._is_amx_tile_supported()
    if supports_amx and dtype in (torch.bfloat16,):
        return "amx"
    else:
        return "vec"
