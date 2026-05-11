# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
    unpack_quantized_values_into_int32,
)
from vllm.model_executor.layers.quantization.utils.zentorch import has_zentorch_op
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

logger = init_logger(__name__)

_CPUWNA16_SUPPORTED_QUANT_TYPES = (scalar_types.uint4, scalar_types.uint4b8)


def _import_unpack_from_int32():
    """Import compressed-tensors' ``unpack_from_int32`` across versions."""
    try:
        from compressed_tensors.compressors.pack_quantized.helpers import (
            unpack_from_int32,
        )
    except ImportError:
        from compressed_tensors.compressors.quantized_compressors.pack_quantized import (  # type: ignore[import-not-found]  # noqa: E501
            unpack_from_int32,
        )
    return unpack_from_int32


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
    def _process_gptq_weights_w4a16(self, layer: torch.nn.Module):
        packed_weight = getattr(layer, self.w_q_name)
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

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale`  is: {input_dim = 0, output_dim = 1}
    #  `weight_zp`     is: {input_dim = 0, output_dim = 1, packed_dim = 1}
    def _process_gptq_weights_w4a8(self, layer: torch.nn.Module):
        packed_weight = getattr(layer, self.w_q_name)
        scales = getattr(layer, self.w_s_name)
        group_num = scales.data.size(0)
        zp_output_size = scales.data.size(1) // 8
        if self.config.zero_points:
            assert self.w_zp_name
            packed_zp = getattr(layer, self.w_zp_name)
        else:
            # w4a8 kernel always requires zp, allocate a fake zp
            assert self.w_zp_name
            packed_zp = torch.nn.Parameter(
                torch.ones(group_num, zp_output_size, dtype=torch.int32) * -2004318072,
                requires_grad=False,
            )
            setattr(layer, self.w_zp_name, packed_zp)

        # FIXME: some bugs in convert_weight_packed_scale_zp with GPTQ format,
        # repack to AWQ weight
        weight = unpack_quantized_values_into_int32(
            packed_weight, self.config.weight_type, 0
        )
        input_size, output_size = weight.size()
        weight = weight.view(input_size, output_size // 8, 8)
        weight = weight[:, :, (0, 2, 4, 6, 1, 3, 5, 7)].reshape(input_size, output_size)
        weight = pack_quantized_values_into_int32(
            weight, self.config.weight_type, 1
        ).contiguous()

        zp = unpack_quantized_values_into_int32(packed_zp, self.config.weight_type, 1)
        zp = zp.view(group_num, output_size // 8, 8)
        zp = zp[:, :, (0, 2, 4, 6, 1, 3, 5, 7)].reshape(group_num, output_size)
        zp = pack_quantized_values_into_int32(
            zp, self.config.weight_type, 1
        ).contiguous()

        blocked_w, blocked_zp, blocked_s = ops.convert_weight_packed_scale_zp(
            weight,
            zp,
            scales.data,
            ops.CPUQuantAlgo.AWQ,
        )

        if layer.bias is not None:
            layer.bias.data = layer.bias.float()

        packed_weight.data = blocked_w
        scales.data = blocked_s
        packed_zp.data = blocked_zp

    # ------------------------------------------------------------------
    # zentorch fast path (Zen CPU, GPTQ-style W4A16)
    # ------------------------------------------------------------------

    def _zentorch_woq_eligible(self, layer: torch.nn.Module) -> bool:
        """Eligibility predicate for the zentorch W4A16 GPTQ fast path.

        Constraints (any failure -> legacy ``ops.cpu_gemm_wna16`` path with
        ``layer`` untouched).
        """
        if not has_zentorch_op("zentorch_woq_repack_weight", "zentorch_woq_linear"):
            return False
        if hasattr(layer, "weight_g_idx") or getattr(self.config, "has_g_idx", False):
            return False

        weight_packed = getattr(layer, "weight_packed", None)
        weight_scale = getattr(layer, "weight_scale", None)
        if weight_packed is None or weight_scale is None:
            return False

        # GPTQ-only. AWQ packs along the output dim instead.
        in_dim = getattr(weight_packed, "input_dim", None)
        pk_dim = getattr(weight_packed, "packed_dim", None)
        if in_dim is None or pk_dim is None or in_dim != pk_dim:
            return False

        if weight_packed.dim() != 2 or weight_scale.dim() != 2:
            return False

        # 4-bit -> 8 values per int32; in_features must be divisible by num_groups.
        in_features = weight_packed.shape[1] * 8
        num_groups = weight_scale.shape[1]
        return num_groups > 0 and in_features % num_groups == 0


    def _process_weights_for_zentorch_woq(self, layer: torch.nn.Module) -> None:
        """Repack compressed-tensors GPTQ weights into zentorch's WOQ layout
        and cache them on dedicated attributes.

        On success ``layer._zentorch_processed_weights`` is set to ``True``;
        original ``weight_packed`` / ``weight_scale`` (and optional zero-point)
        storages are released. The original ``Parameter`` objects are kept
        with empty storage so any downstream consumer that still inspects
        them via attribute access does not crash with ``AttributeError``.
        """
        unpack_from_int32 = _import_unpack_from_int32()
        repack_op = torch.ops.zentorch.zentorch_woq_repack_weight.default

        weight_packed = layer.weight_packed  # (out, in/8) compressed-tensors
        weight_scale = layer.weight_scale  # (out, num_groups)
        out_features = weight_scale.shape[0]
        num_groups = weight_scale.shape[1]
        in_features = weight_packed.shape[1] * 8
        original_shape = torch.Size([out_features, in_features])

        # Unpack: (out, in/8) -> (out, in) int8.
        weight_unpacked = unpack_from_int32(
            weight_packed, 4, original_shape, packed_dim=1
        )

        zp_param = getattr(layer, "weight_zero_point", None)
        # zentorch_woq_repack_weight expects unsigned-nibble [0, 15] storage.
        needs_unsigned_offset = self.config.weight_type == scalar_types.uint4

        if needs_unsigned_offset:
            weight_unpacked = (weight_unpacked.to(torch.int32) + 8).clamp(0, 15)
        repacked = repack_op(weight_unpacked.to(torch.int8).contiguous())

        if zp_param is None:
            zp_tc = None
        else:
            zp = unpack_from_int32(
                zp_param, 4, (out_features, num_groups), packed_dim=0
            )
            if needs_unsigned_offset:
                zp = (zp.to(torch.int32) + 8).clamp(0, 15)
            zp_tc = zp.to(torch.int8).t().contiguous()

        # Cache zentorch-ready tensors on dedicated attributes.
        layer._zentorch_woq_packed = repacked.t()
        layer._zentorch_woq_scale = weight_scale.data.t().contiguous()
        layer._zentorch_woq_zero_point = zp_tc

        # Free the underlying storage of the original buffers.
        layer.weight_packed.data = torch.empty(0)
        layer.weight_scale.data = torch.empty(0)
        if zp_param is not None:
            layer.weight_zero_point.data = torch.empty(0)

        layer._zentorch_kind = "compressed_tensors_w4a16_gptq"
        layer._zentorch_processed_weights = True
        logger.info_once(
            "[zen_cpu] Using zentorch_woq_linear for W4A16 GPTQ "
            "(weight_type=%s, has_zp=%s)",
            self.config.weight_type,
            zp_tc is not None,
        )


    def process_weights_after_loading(self, layer: torch.nn.Module):
        # Idempotency: a previous call already set up the zentorch fast path.
        if getattr(layer, "_zentorch_processed_weights", False):
            return

        if self._zentorch_woq_eligible(layer):
            self._process_weights_for_zentorch_woq(layer)
            return

        weights = getattr(layer, self.w_q_name)
        # Require GPTQ pack format
        assert weights.input_dim == weights.packed_dim

        # Weights in CT format is [output_size, input_size]
        if weights.input_dim == 1:
            weights.data = weights.t()

        # Scales in CT format is [output_size, group_num]
        scales = getattr(layer, self.w_s_name)
        if scales.output_dim == 0:
            scales.data = scales.t().contiguous()

        # Zero points in CT format is [output_size, group_num]
        # Zero points in awq_marlin format is [output_size, group_num]
        if self.config.zero_points:
            assert self.w_zp_name
            zp = getattr(layer, self.w_zp_name)
            if zp.output_dim == 0:
                zp.data = zp.t().contiguous()

        layer.use_w4a8 = (
            envs.VLLM_CPU_INT4_W4A8
            and not self.config.has_g_idx
            and self.config.act_type == torch.bfloat16
            and torch.cpu._is_amx_tile_supported()
        )
        # layer.use_w4a8 = False
        # AWQ format will be converted to GPTQ format in `AWQMarlinLinearMethod`
        if layer.use_w4a8:
            self._process_gptq_weights_w4a8(layer)
        else:
            self._process_gptq_weights_w4a16(layer)


    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(layer, "_zentorch_processed_weights", False):
            return torch.ops.zentorch.zentorch_woq_linear.default(
                x,
                layer._zentorch_woq_packed,
                layer._zentorch_woq_scale,
                layer._zentorch_woq_zero_point,
                bias,
            )
        w_q, w_s, w_zp, w_gidx = self._get_weight_params(layer)
        if layer.use_w4a8:
            x = ops.int4_scaled_mm_cpu(
                x=x,
                w=w_q,
                w_zeros=w_zp,
                w_scales=w_s,
                bias=bias,
            )
        else:
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
