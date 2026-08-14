# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

_XPUWNA16_SUPPORTED_QUANT_TYPES = (scalar_types.uint4, scalar_types.uint4b8)

logger = init_logger(__name__)


class XPUwNa16LinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUwNa16 only supported on XPU"

        if c.act_type != torch.bfloat16 and c.act_type != torch.float16:
            return False, "XPUwNa16 only supports BF16/FP16 activations"

        if c.weight_type not in _XPUWNA16_SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                "XPUwNa16, supported types are: "
                f"{_XPUWNA16_SUPPORTED_QUANT_TYPES}",
            )
        if c.group_size != -1 and c.group_size % 32 != 0:
            return (
                False,
                f"Group size ({c.group_size}) not supported by "
                "XPUwNa16, supported group sizes are multiples of 32",
            )

        if c.partition_weight_shape[0] % 32 != 0:
            return (
                False,
                f"Input size ({c.partition_weight_shape[0]}) not supported by "
                "XPUwNa16, supported sizes are multiples of 32",
            )

        if c.partition_weight_shape[1] % 32 != 0:
            return (
                False,
                f"Output size ({c.partition_weight_shape[1]}) not supported by "
                "XPUWNA16, supported sizes are multiples of 32",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module):
        layer.weight_scale.data = layer.weight_scale.t().contiguous()

        if self.config.zero_points:
            layer.weight_zero_point.data = layer.weight_zero_point.t().contiguous()
        else:
            weight_zero_point = torch.Tensor([8]).to(torch.int8).to("xpu")
            layer.weight_zero_point = Parameter(weight_zero_point, requires_grad=False)
        if self.config.has_g_idx:
            layer.g_idx.data = layer.g_idx.t().contiguous()
        else:
            layer.g_idx = None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])
        group_size = self.config.group_size
        if group_size == -1:
            # Channelwise: group_size = K (input dimension)
            group_size = reshaped_x.shape[-1]
        out = torch.ops._xpu_C.int4_gemm_w4a16(
            reshaped_x,
            layer.weight_packed.t(),
            bias,
            layer.weight_scale,
            layer.weight_zero_point,
            group_size,
            layer.g_idx,
        )
        return out


class XPUW4A8IntLinearKernel(MPLinearKernel):
    """XPU kernel for W4A8 integer quantization using oneDNN int4_gemm_w4a8.

    Weights are symmetric group-quantized int4 packed as uint4.
    Activations are dynamically quantized per-token to symmetric int8.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUW4A8Int only supported on XPU"
        if c.act_type not in (torch.bfloat16, torch.float16):
            return False, "XPUW4A8Int requires BF16/FP16 activations"
        if c.weight_type != scalar_types.int4:
            return (
                False,
                f"XPUW4A8Int requires int4 weights, got {c.weight_type}",
            )
        if c.zero_points:
            return False, "XPUW4A8Int only supports symmetric weight quantization"
        if c.group_size != -1 and c.group_size % 32 != 0:
            return (
                False,
                f"Group size ({c.group_size}) not supported by XPUW4A8Int, "
                "must be a multiple of 32",
            )
        in_size, out_size = c.partition_weight_shape
        if in_size % 8 != 0 or out_size % 8 != 0:
            return (
                False,
                f"in/out sizes ({in_size}, {out_size}) must be multiples of 8",
            )

        if c.act_type != torch.float16:
            logger.warning_once(
                "XPUW4A8IntLinearKernel is running with model dtype %s, "
                "but int4_gemm_w4a8 produces float16 output. Recommend "
                "setting --dtype float16 for best performance.",
                c.act_type,
            )

        return True, None

    def _pack_int4_weight(self, w: torch.Tensor) -> torch.Tensor:
        # w is [N, K] int8 with values in [-8, 7]
        w_u4 = w.to(torch.int32) + 8  # shift to [0, 15]
        w_u4 = w_u4.reshape(w.shape[0], w.shape[1] // 8, 8)  # [N, K/8, 8]
        shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=w.device)
        packed = ((w_u4 & 0xF) << shifts[None, None, :]).sum(dim=2).to(torch.int32)
        return packed

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight_scale.data = layer.weight_scale.data.t().contiguous()

        device = layer.weight_packed.device
        # TODO: support asymmetric quantization
        weight_zero_point = torch.tensor([8], dtype=torch.int8, device=device)
        layer.weight_zero_point = Parameter(weight_zero_point, requires_grad=False)

        # weight_packed is [out, in] int8, signed int4 values in [-8, 7]
        w = layer.weight_packed.data  # [out, in]

        # TODO: implement asym case
        packed = self._pack_int4_weight(w)  # [out, in/8] packed uint4

        replace_parameter(
            layer,
            self.w_q_name,
            torch.nn.Parameter(packed, requires_grad=False),
        )

        # Free the original unpacked int8 weight (still registered as "weight")
        # to avoid double-storing both int8 [N, K] and int32 [N, K/8] in memory.
        layer.register_parameter("weight", None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])  # [M, K]
        from vllm._xpu_ops import xpu_ops as ops

        # TODO: static and asymmetric quantization case
        # Common code for CompressedTensorsW4A8Int does not read act symmetry data
        quant_x, x_scale, x_zero = ops.dynamic_per_token_int8_quant_ref(
            reshaped_x, True, 8
        )

        out = torch.ops._xpu_C.int4_gemm_w4a8(
            quant_x,
            x_scale,
            x_zero,
            layer.weight_packed.t(),
            layer.weight_scale,
            layer.weight_zero_point,
            self.config.group_size,
            None,  # g_idx not currently supported
            bias,
        )

        return out.to(x.dtype)


class XPUDequantLinearKernel(MPLinearKernel):
    """Generic dequantization-based fallback kernel for XPU.

    For 2-bit weights: re-packs into 4-bit format and uses int4_gemm_w4a16.
    For 8-bit weights: dequantizes to bf16 at load time (small layers only).
    """

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUDequantLinear only supported on XPU"
        if c.act_type not in (torch.bfloat16, torch.float16):
            return False, "XPUDequantLinear only supports BF16/FP16 activations"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        import sys
        num_bits = self.config.weight_type.size_bits
        w_packed = layer.weight_packed.data  # [out, in_packed] int32
        print(f'[DEQUANT] bits={num_bits}, w_packed={list(w_packed.shape)}, dtype={w_packed.dtype}', file=sys.stderr, flush=True)

        if num_bits <= 4:
            # Re-pack into 4-bit format for int4_gemm_w4a16
            # First unpack from N-bit to individual values
            pack_factor = 32 // num_bits
            mask = (1 << num_bits) - 1
            out_features = w_packed.shape[0]
            in_packed = w_packed.shape[1]
            in_features = in_packed * pack_factor

            w_int32 = w_packed.to(torch.int32)
            shifts = torch.arange(
                0, 32, num_bits, dtype=torch.int32, device=w_packed.device
            )
            w_unpacked = ((w_int32.unsqueeze(-1) >> shifts) & mask).reshape(
                out_features, in_features
            )

            if num_bits == 2:
                # Re-center 2-bit values for 4-bit zero point.
                # 2-bit symmetric: zp=2, values [0,1,2,3] → dequant [-2,-1,0,1]*s
                # 4-bit oneDNN:    zp=8, values [0..15]   → dequant [v-8]*s
                # To preserve: (v2-2)*s = (v4-8)*s  →  v4 = v2 + 6
                w_unpacked = w_unpacked + 6

            # Re-pack into 4-bit (int4) packed format [out, in/8] int32
            new_pack_factor = 8  # 32 / 4
            if in_features % new_pack_factor != 0:
                # Pad to multiple of 8
                pad_size = new_pack_factor - (in_features % new_pack_factor)
                w_unpacked = torch.nn.functional.pad(
                    w_unpacked, (0, pad_size), value=0
                )
                in_features = w_unpacked.shape[1]

            w_4bit = w_unpacked.reshape(out_features, in_features // new_pack_factor, new_pack_factor)
            shifts_4 = torch.arange(0, 32, 4, dtype=torch.int32, device=w_packed.device)
            w_repacked = ((w_4bit & 0xF) << shifts_4[None, None, :]).sum(dim=2).to(torch.int32)

            replace_parameter(
                layer,
                self.w_q_name,
                torch.nn.Parameter(w_repacked, requires_grad=False),
            )

            # Transpose scales for oneDNN format [n, groups] -> [groups, n]
            layer.weight_scale.data = layer.weight_scale.t().contiguous()

            # Set zero point (always 8 for int4 kernel, regardless of
            # original num_bits, since we re-center values during repacking)
            if not self.config.zero_points or not hasattr(layer, "weight_zero_point"):
                weight_zero_point = torch.tensor(
                    [8], dtype=torch.int8, device=w_packed.device
                )
                layer.weight_zero_point = Parameter(
                    weight_zero_point, requires_grad=False
                )
            else:
                layer.weight_zero_point.data = layer.weight_zero_point.t().contiguous()

            if hasattr(layer, 'g_idx') and layer.g_idx is not None:
                layer.g_idx.data = layer.g_idx.t().contiguous()
            else:
                layer.g_idx = None

            self._use_int4 = True
            self._in_features = in_features
        else:
            # 8-bit: dequantize to bf16 (for small layers like per_layer_*)
            out_features = w_packed.shape[0]

            # For int-quantized 8-bit, weight is stored as int8 directly
            w_float = w_packed.to(torch.float32)
            scale = layer.weight_scale.data
            if self.config.zero_points and hasattr(layer, "weight_zero_point"):
                zp = layer.weight_zero_point.data.to(torch.float32)
            else:
                zp = torch.tensor(
                    [128.0], dtype=torch.float32, device=w_packed.device
                )
            w_float = (w_float - zp) * scale.to(torch.float32)

            act_dtype = self.config.act_type
            replace_parameter(
                layer,
                self.w_q_name,
                torch.nn.Parameter(w_float.to(act_dtype), requires_grad=False),
            )
            self._use_int4 = False

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(self, '_use_int4', False):
            reshaped_x = x.reshape(-1, x.shape[-1])
            group_size = self.config.group_size
            if group_size == -1:
                group_size = self._in_features
            out = torch.ops._xpu_C.int4_gemm_w4a16(
                reshaped_x,
                layer.weight_packed.t(),
                bias,
                layer.weight_scale,
                layer.weight_zero_point,
                group_size,
                layer.g_idx,
            )
            return out
        else:
            w = layer.weight_packed
            return torch.nn.functional.linear(x, w, bias)
