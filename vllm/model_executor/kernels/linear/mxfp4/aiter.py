# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import CompilationMode, get_current_vllm_config
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

    # class level so a kernel built without __init__ (object.__new__, as the
    # tests do) keeps the single mxfp4_gemm_dynamic_quant call
    fold_act_quant = False
    split_act_quant = False
    fold_asm_act_quant = False
    split_asm_act_quant = False

    def __init__(self, config: MxFp4LinearLayerConfig) -> None:
        super().__init__(config)
        self.asm_gemm_supported = rocm_aiter_ops.is_asm_fp4_gemm_supported()
        self.use_asm_gemm = False
        self.is_hadamard_fused = False
        self.out_dtype = torch.get_default_dtype()
        # only emit the quant as its own op when a pass will fold it, so eager
        # mode and -O0 keep the single mxfp4_gemm_dynamic_quant call
        compilation_config = get_current_vllm_config().compilation_config
        pass_config = compilation_config.pass_config
        compiled = compilation_config.mode == CompilationMode.VLLM_COMPILE
        self.fold_act_quant = compiled and (
            bool(pass_config.fuse_act_quant)
            or bool(pass_config.fuse_norm_quant and pass_config.fuse_norm_mxfp4_quant)
        )
        # the ASM layout only has the SiLU-mul fold
        self.fold_asm_act_quant = compiled and bool(pass_config.fuse_act_quant)
        self.split_act_quant = False
        self.split_asm_act_quant = False

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

        # the folds into SiLU-mul and RMSNorm write the Triton afp4wfp4 layout,
        # the ASM layout (HIP quant, shuffled scales) has a SiLU-mul fold of its own
        self.split_act_quant = (
            not self.use_asm_gemm
            and self.fold_act_quant
            and bool(rocm_aiter_ops.is_mxfp4_quant_fusion_supported())
        )
        self.split_asm_act_quant = (
            self.use_asm_gemm
            and self.fold_asm_act_quant
            and bool(rocm_aiter_ops.is_mxfp4_asm_quant_fusion_supported())
        )

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

    def fuse_input_hadamard(self, size: int) -> bool:
        # Only the HIP activation quant of the ASM path rotates, in 32-blocks.
        self.is_hadamard_fused = self.use_asm_gemm and size == 32
        return self.is_hadamard_fused

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.split_act_quant:
            # Emit the activation quant as its own op so the aiter fusion
            # passes can fold it into the producing SiLU-mul / RMSNorm. Left
            # unfused it is the same dynamic_mxfp4_quant + gemm_afp4wfp4.
            x_q, x_s = torch.ops.vllm.rocm_aiter_dynamic_mxfp4_quant(x)
            y = rocm_aiter_ops.mxfp4_gemm_dynamic_quant(
                x_q,
                layer.weight,
                layer.weight_scale,
                False,
                self.out_dtype,
                x_s,
            )
        elif self.split_asm_act_quant and not self.is_hadamard_fused:
            # Same for the ASM layout: the HIP quant mxfp4_gemm_dynamic_quant
            # runs otherwise, scales laid out the way the preshuffle / a4w4
            # GEMM reads them for this M. The standalone quant doesnt rotate,
            # so a layer that fused its input Hadamard keeps the single call
            # (fuse_input_hadamard runs after the split is decided).
            x_q, x_s = torch.ops.vllm.rocm_aiter_dynamic_mxfp4_asm_quant(x)
            y = rocm_aiter_ops.mxfp4_gemm_dynamic_quant(
                x_q,
                layer.weight,
                layer.weight_scale,
                True,
                self.out_dtype,
                x_s,
            )
        else:
            y = rocm_aiter_ops.mxfp4_gemm_dynamic_quant(
                x,
                layer.weight,
                layer.weight_scale,
                self.use_asm_gemm,
                self.out_dtype,
                hadamard_rotate=self.is_hadamard_fused,
            )
        if bias is not None:
            y = y + bias
        return y
