# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
    unpack_quantized_values_into_int32,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

_CPUWNA16_SUPPORTED_QUANT_TYPES = (scalar_types.uint4, scalar_types.uint4b8)


def _requantize_to_int8(
    float_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    K, N = float_weight.shape
    channel_max = float_weight.abs().amax(dim=0)
    channel_scale = (channel_max / 127.0).clamp(min=1e-10)
    weight_int8 = (float_weight / channel_scale.unsqueeze(0)).round().clamp(
        -128, 127
    ).to(torch.int8)
    return weight_int8, channel_scale


def _dequant_gptq_to_float(
    weight_int4: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    g_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize GPTQ int4 weights to float32.
    GPTQ uses signed int4 (uint4b8: raw [0,15] with bias 8 → [-8, 7]) si NO zero point. 
    Dequant formula: float_w = signed_int4 * scale
    Args:
        weight_int4: [K, N] int32, raw unpacked 4-bit values (0-15)
        scales:      [num_groups, N] bf16/fp16, per-group per-channel scale
        group_size:  number of rows per group
        g_idx:       [K] int32, optional group index for desc_act

    Returns:
        float_weight: [K, N] float32
    """
    K, N = weight_int4.shape
    num_groups = scales.shape[0]


    signed_int4 = weight_int4.float() - 8.0 # uint4b8: raw [0,15] → signed [-8, 7] by subtracting bias 8


    if g_idx is not None:
        # g_idx: [K] int32, g_idx[k] = group index for row k
        scales_expanded = scales[g_idx.long(), :]  # [K, N]
    else:
        scales_expanded = scales.repeat_interleave(group_size, dim=0)  # [K, N]

    float_weight = signed_int4 * scales_expanded.float()

    return float_weight


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
                assert self.w_zp_name
                zp = getattr(layer, self.w_zp_name)
                zp.data = zp.t().contiguous()

    def _process_gptq_weights_int8(self, layer: torch.nn.Module):
        """Convert GPTQ int4 weights to int8 and create oneDNN handler.

        Flow: unpack int4 -> dequant to float -> re-quantize to int8
              -> create oneDNN handler
        """
        w_q, w_s, w_zp, w_gidx = self._get_weight_params(layer)
        packed_weight = w_q.data
        scales = w_s.data
        g_idx = w_gidx.data if w_gidx is not None else None

        bits = self.config.weight_type.size_bits
        pack_factor = 32 // bits
        p_w_k, p_w_n = packed_weight.size()
        input_size = p_w_k * pack_factor
        output_size = p_w_n
        group_size = self.config.group_size if self.config.group_size > 0 \
            else input_size


        weight_int4 = unpack_quantized_values_into_int32(
            packed_weight, self.config.weight_type, 0
        )

        float_weight = _dequant_gptq_to_float(
            weight_int4, scales, group_size, g_idx
        )

        weight_int8, channel_scale = _requantize_to_int8(float_weight)
        channel_scale_2d = channel_scale.unsqueeze(0)

        weight_int8 = weight_int8.t().contiguous().t()

        self.dnnl_handler = ops.create_onednn_scaled_mm(
            weight_int8,
            channel_scale_2d,
            torch.get_default_dtype(),
            True,
            False,
            32,
        )
        del weight_int8, float_weight
        if self.w_q_name and hasattr(layer, self.w_q_name):
            setattr(layer, self.w_q_name, None)
        if self.w_s_name and hasattr(layer, self.w_s_name):
            setattr(layer, self.w_s_name, None)
        if self.w_zp_name and hasattr(layer, self.w_zp_name):
            setattr(layer, self.w_zp_name, None)
    def process_weights_after_loading(self, layer: torch.nn.Module):
        if (not self.config.zero_points) and (self.w_zp_name is not None):
            setattr(layer, self.w_zp_name, None)

        if (not self.config.has_g_idx) and (self.w_gidx_name is not None):
            setattr(layer, self.w_gidx_name, None)

        w_input_dim = getattr(layer, self.w_q_name).input_dim
        w_pack_dim = getattr(layer, self.w_q_name).packed_dim
        quant_method = "gptq" if w_pack_dim == w_input_dim else "awq"

        if quant_method == "gptq":
            if envs.VLLM_CPU_WOQ_INT8_MODE:
                self._process_gptq_weights_int8(layer)
            else:
                self._process_gptq_weights(layer)
        else:
            raise NotImplementedError("AWQ is not supported in CPUWNA16LinearKernel")

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if envs.VLLM_CPU_WOQ_INT8_MODE and hasattr(self, 'dnnl_handler'):
            return self._apply_weights_int8(layer, x, bias)
        else:
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

    def _apply_weights_int8(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Int8 oneDNN inference path."""
        x_shape = x.shape
        x_2d = x.reshape(-1, x_shape[-1]) if len(x_shape) > 2 else x

        x_q, x_s, _ = ops.onednn_scaled_int8_quant(x_2d, None, None, True)

        m = x_2d.size(0)
        n = self.dnnl_handler.n
        out = torch.empty((m, n), dtype=x.dtype)
        ops.onednn_scaled_mm(
            self.dnnl_handler,
            x_q,
            out,
            x_s,
            None,
            None,
            bias,
        )

        out = out.reshape(x_shape[:-1] + (n,)) if len(x_shape) > 2 else out
        return out


def _get_isa_hint(dtype: torch.dtype) -> str:
    supports_amx = torch.cpu._is_amx_tile_supported()
    if supports_amx and dtype in (torch.bfloat16,):
        return "amx"
    else:
        return "vec"
