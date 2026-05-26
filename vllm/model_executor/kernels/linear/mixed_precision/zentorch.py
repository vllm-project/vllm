# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Zentorch W4A16 GPTQ weight-only-quantized linear kernel for AMD Zen CPUs.

Selected by ``choose_mp_linear_kernel`` ahead of the generic oneDNN-backed
``CPUWNA16LinearKernel``. When ``can_implement`` rejects a layer, the selector
falls through to the next kernel in ``_POSSIBLE_KERNELS[PlatformEnum.CPU]``.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.zentorch import has_zentorch_op
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
    """W4A16 GPTQ kernel backed by ``torch.ops.zentorch.zentorch_woq_linear``.
    """

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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Repack CT GPTQ weights into the zentorch WOQ layout.

        Falls back to ``CPUWNA16LinearKernel.process_weights_after_loading``
        via ``super()`` when the layer doesn't satisfy
        ``_zentorch_woq_eligible``.

        On success, ``layer._zentorch_processed_weights`` is set to ``True``
        """
        if getattr(layer, "_zentorch_processed_weights", False):
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
