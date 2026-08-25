# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Zentorch int4 weight-quantized linear kernels for AMD Zen CPUs.

Serves a W4 checkpoint either as W4A16 (weight-only, dequantized to bf16) or as
DA8W4/W4A8, which reuses the same int4 weights but quantizes activations to int8
per token at runtime so the GEMM runs as int8 x int8. DA8W4 is preferred and
``VLLM_CPU_INT4_W4A8=0`` forces W4A16.

Selected by ``choose_mp_linear_kernel`` ahead of the generic oneDNN-backed
``CPUWNA16LinearKernel``. When ``can_implement`` rejects a layer, the selector
falls through to the next kernel in ``_POSSIBLE_KERNELS[PlatformEnum.CPU]``.
"""

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.zentorch_utils import has_zentorch_op
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .cpu import CPUWNA16LinearKernel
from .MPLinearKernel import MPLinearLayerConfig

logger = init_logger(__name__)


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


class ZentorchWNA16LinearKernel(CPUWNA16LinearKernel):
    """Int4 kernel backed by ``zentorch_dynamic_qlinear`` (DA8W4) or
    ``zentorch_woq_linear`` (W4A16)."""

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        ok, reason = super().can_implement(c)
        if not ok:
            return ok, reason

        if not current_platform.is_zen_cpu():
            return False, "ZentorchWNA16 requires an AMD Zen CPU."

        if not has_zentorch_op(["zentorch_woq_repack_weight", "zentorch_woq_linear"]):
            return (
                False,
                "torch.ops.zentorch.{zentorch_woq_repack_weight, "
                "zentorch_woq_linear} are not registered.",
            )

        if c.has_g_idx:
            return False, "ZentorchWNA16 does not support activation re-ordering."
        return True, None

    def _zentorch_woq_eligible(self, layer: torch.nn.Module) -> bool:
        """Eligibility predicate for the zentorch W4A16 GPTQ fast path.

        Constraints (any failure -> ``cpu_gemm_wna16`` path via ``super()``
        with ``layer`` untouched).
        """
        if (
            self.w_gidx_name is not None
            and getattr(layer, self.w_gidx_name, None) is not None
        ) or (getattr(self.config, "has_g_idx", False)):
            return False

        weight_packed = getattr(layer, self.w_q_name, None)
        weight_scale = getattr(layer, self.w_s_name, None)
        if weight_packed is None or weight_scale is None:
            return False

        bits = self.config.weight_type.mantissa
        pack_factor = torch.iinfo(weight_packed.dtype).bits // bits
        # 4-bit -> 8 values per int32;
        if pack_factor != 8:
            return False

        # GPTQ-only. AWQ packs along the output dim instead.
        in_dim = getattr(weight_packed, "input_dim", None)
        pk_dim = getattr(weight_packed, "packed_dim", None)
        if in_dim is None or pk_dim is None or in_dim != pk_dim:
            return False

        is_ct_format = in_dim == pk_dim == 1
        if not is_ct_format:
            return False

        if weight_packed.dim() != 2 or weight_scale.dim() != 2:
            return False

        # 4-bit -> 8 values per int32; in_features must be divisible by num_groups.
        in_features = weight_packed.shape[1] * 8
        num_groups = weight_scale.shape[1]
        return num_groups > 0 and in_features % num_groups == 0

    def _zentorch_da8w4_eligible(self, layer: torch.nn.Module) -> bool:
        """Eligibility predicate for running a W4 layer as DA8W4 (W4A8).

        DA8W4 consumes the same checkpoint as the W4A16 path, so the W4A16
        checks apply on top of the symmetric/bf16 requirements below. Any
        failure leaves ``layer`` untouched for the W4A16 path.
        """
        if not envs.VLLM_CPU_INT4_W4A8:
            return False

        if not has_zentorch_op(
            ["zentorch_woq_repack_weight", "zentorch_dynamic_qlinear"]
        ):
            return False

        # DA8W4 is symmetric-only, and the kernel rejects f32 activations.
        if self.config.zero_points or self.config.weight_type == scalar_types.uint4:
            return False
        if self.config.act_type != torch.bfloat16:
            return False

        if not self._zentorch_woq_eligible(layer):
            return False

        weight_packed = getattr(layer, self.w_q_name)
        weight_scale = getattr(layer, self.w_s_name)
        in_features = weight_packed.shape[1] * 8
        num_groups = weight_scale.shape[1]
        group_size = in_features // num_groups
        # AOCL sym_quant requires K/G to be a multiple of 4; K must be even to
        # pack two s4 values per byte.
        return group_size % 4 == 0 and in_features % 2 == 0

    def _process_da8w4_weights(self, layer: torch.nn.Module) -> None:
        """Repack CT int4 weights into the packed-s4 layout DA8W4 consumes."""
        if self.w_zp_name is not None:
            setattr(layer, self.w_zp_name, None)
        if self.w_gidx_name is not None:
            setattr(layer, self.w_gidx_name, None)

        weight_q = getattr(layer, self.w_q_name)
        weight_s = getattr(layer, self.w_s_name)
        weight_packed = weight_q.data if hasattr(weight_q, "data") else weight_q
        weight_scale = weight_s.data if hasattr(weight_s, "data") else weight_s

        bits = self.config.weight_type.mantissa
        pack_factor = torch.iinfo(weight_packed.dtype).bits // bits
        out_features, num_groups = weight_scale.shape[0], weight_scale.shape[1]
        in_features = weight_packed.shape[1] * pack_factor
        unpack_from_int32 = _import_unpack_from_int32()

        weight_unpacked = unpack_from_int32(
            weight_packed,
            bits,
            torch.Size([out_features, in_features]),
            packed_dim=weight_q.packed_dim,
        )
        # The WOQ repack packs 8 int4 per int32, which on little-endian is the
        # same byte stream as s4 [N, K/2]; the kernel takes either view.
        layer._zentorch_da8w4_packed = (
            torch.ops.zentorch.zentorch_woq_repack_weight.default(
                weight_unpacked.to(torch.int8).contiguous()
            ).view(torch.int8)
        )
        # CT stores scales as [N, G]; DA8W4 wants per-group {G, N}.
        layer._zentorch_da8w4_scale = weight_scale.t().to(torch.bfloat16).contiguous()

        for param_name in (self.w_q_name, self.w_s_name):
            param = getattr(layer, param_name, None)
            if param is not None and hasattr(param, "data"):
                param.data = torch.empty(0)

        layer._zentorch_kind = "compressed_tensors_w4a8_da8w4"
        layer._zentorch_da8w4 = True
        layer._zentorch_processed_weights = True
        logger.info_once(
            "[zen_cpu] Using zentorch_dynamic_qlinear for DA8W4 (W4A8) "
            "(weight_type=%s, group_size=%d)",
            self.config.weight_type,
            in_features // num_groups,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Repack CT GPTQ weights into the zentorch DA8W4 or WOQ layout.

        Falls back to ``CPUWNA16LinearKernel.process_weights_after_loading``
        via ``super()`` when the layer doesn't satisfy
        ``_zentorch_woq_eligible``.

        On success, ``layer._zentorch_processed_weights`` is set to ``True``
        """
        if getattr(layer, "_zentorch_processed_weights", False):
            return

        if self._zentorch_da8w4_eligible(layer):
            self._process_da8w4_weights(layer)
            return

        if not self._zentorch_woq_eligible(layer):
            logger.info_once(
                "[zen_cpu] ZentorchWNA16 fast path not eligible for this "
                "layer (AWQ pack layout, g_idx, or non-int32 storage); "
                "falling back to CPUWNA16LinearKernel (cpu_gemm_wna16)."
            )
            super().process_weights_after_loading(layer)
            return

        if (not self.config.zero_points) and (self.w_zp_name is not None):
            setattr(layer, self.w_zp_name, None)

        if (not self.config.has_g_idx) and (self.w_gidx_name is not None):
            setattr(layer, self.w_gidx_name, None)

        weight_q = getattr(layer, self.w_q_name)
        weight_s = getattr(layer, self.w_s_name)
        weight_packed = weight_q.data if hasattr(weight_q, "data") else weight_q
        weight_scale = weight_s.data if hasattr(weight_s, "data") else weight_s

        bits = self.config.weight_type.mantissa
        pack_factor = torch.iinfo(weight_packed.dtype).bits // bits
        out_features, num_groups = weight_scale.shape[0], weight_scale.shape[1]
        in_features = weight_packed.shape[1] * pack_factor
        original_shape = torch.Size([out_features, in_features])
        unpack_from_int32 = _import_unpack_from_int32()
        repack_op = torch.ops.zentorch.zentorch_woq_repack_weight.default

        weight_unpacked = unpack_from_int32(
            weight_packed,
            bits,
            original_shape,
            packed_dim=weight_q.packed_dim,
        )

        zp_param = (
            getattr(layer, self.w_zp_name, None) if self.w_zp_name is not None else None
        )
        needs_unsigned_offset = self.config.weight_type == scalar_types.uint4

        if needs_unsigned_offset:
            weight_unpacked = (weight_unpacked.to(torch.int32) + 8).clamp(0, 15)
        repacked = repack_op(weight_unpacked.to(torch.int8).contiguous())

        if zp_param is None:
            zp_tc = None
        else:
            zp_tensor = zp_param.data if hasattr(zp_param, "data") else zp_param
            zp = unpack_from_int32(
                zp_tensor,
                bits,
                (out_features, num_groups),
                packed_dim=zp_param.packed_dim,
            )
            if needs_unsigned_offset:
                zp = (zp.to(torch.int32) + 8).clamp(0, 15)
            zp_tc = zp.to(torch.int8).t().contiguous()

        layer._zentorch_woq_packed = repacked.t()
        layer._zentorch_woq_scale = weight_scale.t().contiguous()
        layer._zentorch_woq_zero_point = zp_tc

        for param_name in (self.w_q_name, self.w_s_name, self.w_zp_name):
            if param_name is None:
                continue
            param = getattr(layer, param_name, None)
            if param is None:
                continue
            if hasattr(param, "data"):
                param.data = torch.empty(0)
            else:
                setattr(layer, param_name, torch.empty(0))

        layer._zentorch_kind = "compressed_tensors_w4a16_gptq"
        layer._zentorch_processed_weights = True
        logger.info_once(
            "[zen_cpu] Using zentorch_woq_linear for W4A16 GPTQ "
            "(weight_type=%s, has_zp=%s)",
            self.config.weight_type,
            zp_tc is not None,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(layer, "_zentorch_da8w4", False):
            # The kernel reads bias through a raw pointer, so it must be
            # contiguous; the weight and scales already are.
            return torch.ops.zentorch.zentorch_dynamic_qlinear.default(
                x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16),
                layer._zentorch_da8w4_packed,
                layer._zentorch_da8w4_scale,
                bias.contiguous() if bias is not None else None,
            )

        if getattr(layer, "_zentorch_processed_weights", False):
            return torch.ops.zentorch.zentorch_woq_linear.default(
                x,
                layer._zentorch_woq_packed,
                layer._zentorch_woq_scale,
                layer._zentorch_woq_zero_point,
                bias,
            )
        return super().apply_weights(layer, x, bias)


__all__ = ["ZentorchWNA16LinearKernel"]
