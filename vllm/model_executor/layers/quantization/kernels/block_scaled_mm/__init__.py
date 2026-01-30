# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)
from vllm.platforms import PlatformEnum, current_platform

from .aiter import AiterBlockScaledMMKernel
from .BlockScaledMMKernel import Fp8BlockMMScaledConfig, Fp8BlockScaledMMKernel
from .cuda import CudaBlockScaledMMKernel

logger = init_logger(__name__)

__all__ = ["init_fp8_block_scaled_linear_kernel"]


_PLATFORM_PRIORITIES: dict[PlatformEnum, type[Fp8BlockScaledMMKernel]] = {
    PlatformEnum.CUDA: CudaBlockScaledMMKernel,
    PlatformEnum.ROCM: AiterBlockScaledMMKernel,
}


def is_supported_and_can_implement_kernel(
    kernel: type[Fp8BlockScaledMMKernel], config: Fp8BlockMMScaledConfig
) -> tuple[bool, str]:
    is_supported, reason = kernel.is_supported()
    if not is_supported:
        return False, f"{kernel.__name__} not supported. {reason}"

    can_implement, failure_reason = kernel.can_implement(config)
    if not can_implement:
        return (
            False,
            f"{kernel.__name__} not supported for \
            requested config {config}. {failure_reason}.",
        )

    return True, ""


def init_fp8_block_scaled_linear_kernel(
    activation_quant_key: QuantKey,
    weight_quant_key: QuantKey,
    out_dtype: torch.dtype,
    module_name: str | None = None,
) -> Fp8BlockScaledMMKernel:
    config = Fp8BlockMMScaledConfig(
        activation_quant_key=activation_quant_key,
        weight_quant_key=weight_quant_key,
        out_dtype=out_dtype,
    )

    platform_enum = current_platform._enum

    # Check if the current platform has a priority kernel implementation
    if platform_enum in _PLATFORM_PRIORITIES:
        prioritized_kernel = _PLATFORM_PRIORITIES[platform_enum]
        can_dispatch, reason = is_supported_and_can_implement_kernel(
            prioritized_kernel,
            config,
        )
        if can_dispatch:
            module_prefix = f"[{module_name}] " if module_name else ""
            logger.info_once(
                f"{module_prefix} Selected kernel: {prioritized_kernel.__name__}"
            )
            return prioritized_kernel(config)
        else:
            logger.warning_once(f"{reason}")

            fall_back_kernels = prioritized_kernel.ordered_fallback_kernels()
            for kernel in fall_back_kernels:
                can_dispatch, _ = is_supported_and_can_implement_kernel(kernel, config)
                if can_dispatch:
                    module_prefix = f"[{module_name}] " if module_name else ""
                    logger.info_once(
                        f"{module_prefix}Selected kernel: {kernel.__name__} (fallback)"
                    )
                    return kernel(config)

            raise ValueError("None of the exsiting quantization kernel can be selected")

    raise ValueError(
        f"{current_platform} platform is not supported for any scaled mm kernel"
    )
