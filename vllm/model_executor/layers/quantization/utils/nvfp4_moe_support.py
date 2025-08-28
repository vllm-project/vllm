# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    is_flashinfer_fp4_cutlass_moe_available)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    is_fp4_marlin_supported)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    cutlass_fp4_supported)

__all__ = ["detect_nvfp4_moe_support", "NvFp4Support"]

_logger = init_logger(__name__)


@dataclass(frozen=True)
class NvFp4Support:
    """Result container for NV-FP4 capability probing."""

    cutlass_supported: bool
    allow_flashinfer: bool
    use_marlin: bool


def detect_nvfp4_moe_support(class_name: str = "") -> NvFp4Support:
    """Detect platform support for NV-FP4 fused-MoE path"""
    cutlass_supported = cutlass_fp4_supported()

    allow_flashinfer = (cutlass_supported
                        and is_flashinfer_fp4_cutlass_moe_available())

    if allow_flashinfer:
        _logger.info_once("Using FlashInfer kernels for %s.", class_name
                          or "NVFP4 path")
    else:
        if envs.VLLM_USE_FLASHINFER_MOE_FP4:
            _logger.warning_once(
                "FlashInfer kernels unavailable for %s on current platform.",
                class_name or "NVFP4 path",
            )

    use_marlin = False
    if not cutlass_supported:
        if is_fp4_marlin_supported():
            use_marlin = True
            _logger.info_once("Falling back to Marlin FP4 MoE kernel.")
        else:
            raise ValueError(
                "Current platform does not support NVFP4 quantization. "
                "Please use Blackwell GPUs or enable FlashInfer.")

    return NvFp4Support(
        cutlass_supported=cutlass_supported,
        allow_flashinfer=allow_flashinfer,
        use_marlin=use_marlin,
    )
