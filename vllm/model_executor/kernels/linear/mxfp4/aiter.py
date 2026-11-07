# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig

logger = init_logger(__name__)


_ASM_FP4_SCALE_ROW_MULTIPLE = 32
_ASM_FP4_SCALE_COL_MULTIPLE = 8


def _asm_fp4_scale_swizzle_supported(weight_scale: torch.Tensor) -> bool:
    # The ASM swizzle reshapes weight_scale into a fixed tile layout, requiring
    # scale rows divisible by 32 and scale columns divisible by 8
    if weight_scale.ndim != 2:
        return False
    sm, sn = weight_scale.shape
    return (
        sm % _ASM_FP4_SCALE_ROW_MULTIPLE == 0 and sn % _ASM_FP4_SCALE_COL_MULTIPLE == 0
    )


class AiterMxfp4LinearKernel(MxFp4LinearKernel):
    """AITER-based native MXFP4 GEMM kernel for ROCm."""

    def __init__(self, config: MxFp4LinearLayerConfig) -> None:
        super().__init__(config)
        self.asm_gemm_supported = rocm_aiter_ops.is_asm_fp4_gemm_supported()
        self.use_asm_gemm = False
        self.out_dtype = torch.get_default_dtype()

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.supports_mx():
            return False, "current platform does not support native MXFP4 computation"

        from vllm._aiter_ops import is_aiter_found_and_supported
        from vllm.model_executor.kernels.linear import _get_linear_backend

        linear_backend = _get_linear_backend(quantization="mxfp4")

        if (
            current_platform.is_rocm()
            and current_platform.supports_mx()
            and "AiterMxfp4LinearKernel" not in envs.VLLM_DISABLED_KERNELS
            and linear_backend == "auto"
            and not is_aiter_found_and_supported()
        ):
            logger.warning_once(
                "This platform supports native MXFP4 W4A4 MOE "
                "computation via AITER MOE backend, but AITER is not "
                "found or not supported. Consider installing AITER: "
                "https://github.com/ROCm/aiter."
            )

        if is_aiter_found_and_supported():
            return True, None
        return False, "AITER not found or not supported on the current platform"

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        if config.activation_quant_key != kMxfp4Dynamic:
            return False, "only supports MXFP4 dynamic activation"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Use preshuffled ASM (P) and Triton (D) GEMM when tuned in aiter
        if self.asm_gemm_supported:
            n, k_bytes = layer.weight.shape
            self.use_asm_gemm = rocm_aiter_ops.is_asm_fp4_gemm_tuned(
                n, k_bytes
            ) and rocm_aiter_ops.is_triton_gemm_afp4wfp4_presh_ws_tuned(n, k_bytes)

        if self.use_asm_gemm and not _asm_fp4_scale_swizzle_supported(
            layer.weight_scale.data
        ):
            logger.warning_once(
                "AITER ASM FP4 GEMM requires weight_scale dims divisible by "
                "(%d, %d), but this layer has weight_scale shape %s. Falling "
                "back to the AITER Triton FP4 GEMM for this layer.",
                _ASM_FP4_SCALE_ROW_MULTIPLE,
                _ASM_FP4_SCALE_COL_MULTIPLE,
                tuple(layer.weight_scale.data.shape),
            )
            self.use_asm_gemm = False

        if self.use_asm_gemm:
            from aiter.ops.shuffle import shuffle_weight

            weight_scale = layer.weight_scale.data
            sm, sn = weight_scale.shape
            weight_scale = weight_scale.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
            weight_scale = weight_scale.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
            weight_scale = weight_scale.view(sm, sn)
            replace_parameter(layer, "weight_scale", weight_scale)

            replace_parameter(
                layer, "weight", shuffle_weight(layer.weight.data, layout=(16, 16))
            )
        else:
            replace_parameter(
                layer, "weight_scale", layer.weight_scale.data.T.contiguous()
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = rocm_aiter_ops.mxfp4_gemm_dynamic_quant(
            x,
            layer.weight,
            layer.weight_scale,
            self.use_asm_gemm,
            self.out_dtype,
        )
        if bias is not None:
            y = y + bias
        return y
