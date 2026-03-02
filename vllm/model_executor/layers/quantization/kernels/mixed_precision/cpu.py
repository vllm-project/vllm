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



# lyt_debug_G2 helper: dequantize GPTQ int4 weights into float32

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
        g_idx:       [K] int32, optional group index for desc_act (G6)

    Returns:
        float_weight: [K, N] float32
    """
    K, N = weight_int4.shape
    num_groups = scales.shape[0]

    print(f'lyt_debug_G2 _dequant_gptq_to_float called: '
        f'K={K}, N={N}, num_groups={num_groups}, group_size={group_size}, '
        f'has_g_idx={g_idx is not None}')
    print(f'lyt_debug_G2 weight_int4 (raw) range: min={weight_int4.min().item()}, max={weight_int4.max().item()}')
    print(f'lyt_debug_G2 scales range: min={scales.min().item():.6f}, max={scales.max().item():.6f}')

    signed_int4 = weight_int4.float() - 8.0 # uint4b8: raw [0,15] → signed [-8, 7] by subtracting bias 8

    print(f'lyt_debug_G2 signed_int4 range: '
        f'min={signed_int4.min().item():.0f}, max={signed_int4.max().item():.0f}')

    # lyt_debug_G6 desc_act: when g_idx is present, each row maps to its group via g_idx[k] instead of uniform group_size blocks
    if g_idx is not None:
        # g_idx: [K] int32, g_idx[k] = group index for row k
        scales_expanded = scales[g_idx.long(), :]  # [K, N]
        print(f'lyt_debug_G6 desc_act: using g_idx to map rows to groups, '
            f'g_idx range=[{g_idx.min().item()}, {g_idx.max().item()}]')
    else:
        # lyt_debug_G6 Uniform group layout: each group_size rows share the same scale
        scales_expanded = scales.repeat_interleave(group_size, dim=0)  # [K, N]

    float_weight = signed_int4 * scales_expanded.float()

    print(f'lyt_debug_G2 float_weight range: min={float_weight.min().item():.6f}, max={float_weight.max().item():.6f}')
    print(f'lyt_debug_G2 float_weight shape: {float_weight.shape}, dtype: {float_weight.dtype}')

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

    # lyt_debug_G1 new method: GPTQ int4 → float → int8 → oneDNN handler
    def _process_gptq_weights_int8(self, layer: torch.nn.Module):
        """G1: Convert GPTQ int4 weights to int8 and create onednn handler.

        Flow: unpack int4 → dequant to float (G2) → re-quantize to int8 (G3)
              → create oneDNN handler (G4)
        """
        from vllm.model_executor.layers.quantization.cpu_wna16 import _requantize_to_int8

        # Access params through base class names (robust for both
        # GPTQConfig and CompressedTensorsWNA16 paths)
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

        print(f'lyt_debug_G1 _process_gptq_weights_int8 ENTER: '
            f'input_size(K)={input_size}, output_size(N)={output_size}, '
            f'group_size={group_size}, bits={bits}, pack_factor={pack_factor}')
        print(f'lyt_debug_G1 packed_weight shape: {packed_weight.shape}, '
            f'scales shape: {scales.shape}, has_g_idx={g_idx is not None}')

        # lyt_debug_G1: Unpack int4 values
        # GPTQ packed format: input_dim packed (packed_dim=0 in ct, packed_dim=1 in unpack)
        # For compressed_tensors: weight_packed shoudl be {input_dim=0, output_dim=1, packed_dim=0}
        weight_int4 = unpack_quantized_values_into_int32(
            packed_weight, self.config.weight_type, 0
        )
        # weight_int4 shape: [K, N] int32, raw values [0, 15]

        print(f'lyt_debug_G1 after unpack: weight_int4 shape={weight_int4.shape}')

        # lyt_debug Dequant GPTQ int4 into float32 (G2 + G6) 
        float_weight = _dequant_gptq_to_float(
            weight_int4, scales, group_size, g_idx
        )

        # kyt_debug requantize float32 into int8 per-channel (G3)                  
        weight_int8, channel_scale = _requantize_to_int8(float_weight)

        print(f'lyt_debug_G3 final weight_int8 shape: {weight_int8.shape}, '
            f'dtype: {weight_int8.dtype}')
        print(f'lyt_debug_G3 final channel_scale shape: {channel_scale.shape}, '
                f'dtype: {channel_scale.dtype}')

        # lyt_debug_G4 create oneDNN handler for int8 GEMM
        channel_scale_2d = channel_scale.unsqueeze(0)  # [1, N]

        # AZP adjustment: compensation term for dynamic quantization
        azp_adj = weight_int8.sum(dim=0, keepdim=True, dtype=torch.float32)
        azp_adj = azp_adj * channel_scale_2d

        # lyt_debug_G4 oneDNN requires column-major weight: stride(0)==1
        weight_int8 = weight_int8.t().contiguous().t()

        print(f'lyt_debug_G4 creating oneDNN handler: weight_int8 shape={weight_int8.shape}, '
              f'stride={weight_int8.stride()}, channel_scale_2d shape={channel_scale_2d.shape}')
        print(f'lyt_debug_G4 azp_adj shape={azp_adj.shape}, '
            f'range=[{azp_adj.min().item():.4f}, {azp_adj.max().item():.4f}]')

        self.dnnl_handler = ops.create_onednn_scaled_mm(
            weight_int8,                # [K, N] int8, column-major
            channel_scale_2d,            # [1, N] float32
            torch.get_default_dtype(),  # output type (typically bf16)
            True,                        # dynamic_act_quant
            False,                      # use_azp (symmetric input)
            32,                         # primitive_cache_size
        )
        self.azp_adj = torch.nn.Parameter(azp_adj, requires_grad=False)

        print(f'lyt_debug_G4 oneDNN handler created: handler.k={self.dnnl_handler.k}, handler.n={self.dnnl_handler.n}')

        # lyt_debug Clean-up old int4 params to save memory
        del weight_int8, float_weight
        if self.w_q_name and hasattr(layer, self.w_q_name):
            setattr(layer, self.w_q_name, None)
        if self.w_s_name and hasattr(layer, self.w_s_name):
            setattr(layer, self.w_s_name, None)
        if self.w_zp_name and hasattr(layer, self.w_zp_name):
            setattr(layer, self.w_zp_name, None)

        print(f'lyt_debug_G4 _process_gptq_weights_int8 DONE. '
            f'int8 oneDNN path ready, old int4 params cleaned up.')

    # lyt_debug_g4 flag: use int8 path (True) or fallback to original int4 path (False)
    _use_int8_path = True

    def process_weights_after_loading(self, layer: torch.nn.Module):
        if not self.config.zero_points:
            # GPTQ
            if self._use_int8_path:
                # lyt_debug_G4 new int8 path: int4 → dequant → int8 → oneDNN
                self._process_gptq_weights_int8(layer)
            else:
                self._process_gptq_weights(layer)
        else:
            # AWQ
            raise NotImplementedError("AWQ is not supported in CPUWNA16LinearKernel")

    _apply_debug_logged = False

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._use_int8_path and hasattr(self, 'dnnl_handler'):
            # lyt_debug_G5 use oneDNN int8 GEMM instead of cpu_gemm_wna16 int4
            return self._apply_weights_int8(layer, x, bias)
        else:
            x = ops.cpu_gemm_wna16(
                input=x,
                q_weight=layer.qweight,
                scales=layer.scales,
                zeros=layer.qzeros,
                g_idx=layer.g_idx,
                bias=bias,
                pack_factor=8,     # 32 // 4
                isa_hint=layer.isa_hint,
            )
            return x

    # lyt_debug_G5 int8 gemm apply method
    def _apply_weights_int8(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_shape = x.shape
        x_2d = x.reshape(-1, x_shape[-1]) if len(x_shape) > 2 else x

        if not CPUWNA16LinearKernel._apply_debug_logged:
            print(f'lyt_debug_G5 apply ENTER (first call): x shape={x.shape}, dtype={x.dtype}')

        # ly_debug dynamic per-token symmetric quantization: bf16 → int8
        x_q, x_s, _ = ops.onednn_scaled_int8_quant(x_2d, None, None, True)

        m = x_2d.size(0)
        n = self.dnnl_handler.n
        out = torch.empty((m, n), dtype=x.dtype)
        ops.onednn_scaled_mm(
            self.dnnl_handler,
            x_q,
            out,
            x_s,
            None,           # input_zp (symmetric → no zero point)
            self.azp_adj,   #  AZP adjustment
            bias,
        )

        out = out.reshape(x_shape[:-1] + (n,)) if len(x_shape) > 2 else out

        if not CPUWNA16LinearKernel._apply_debug_logged:
            print(f'lyt_debug_G5 apply DONE (first call): out shape={out.shape}, dtype={out.dtype}')
            CPUWNA16LinearKernel._apply_debug_logged = True

        return out


def _get_isa_hint(dtype: torch.dtype) -> str:
    supports_amx = torch._C._cpu._is_amx_tile_supported()
    if supports_amx and dtype in (torch.bfloat16,):
        return "amx"
    else:
        return "vec"
