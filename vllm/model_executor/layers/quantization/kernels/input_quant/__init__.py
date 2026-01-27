# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import PlatformEnum, current_platform

from .aiter import AiterInputQuantKernel
from .deep_gemm import DeepGemmInputQuantKernel
from .InputQuantKernel import InputQuantConfig, InputQuantKernel
from .pytorch import PytorchInputQuantKernel

logger = init_logger(__name__)

__all__ = [
    "select_quant_kernel",
]

_PLATFORM_PRIORITIES: dict[PlatformEnum, type[InputQuantKernel]] = {
    PlatformEnum.CUDA: DeepGemmInputQuantKernel,
    PlatformEnum.ROCM: AiterInputQuantKernel,
}


def is_supported_and_can_implement_kernel(
    kernel: type[InputQuantKernel], config: InputQuantConfig
) -> tuple[bool, str]:
    is_supported, reason = kernel.is_supported()
    if not is_supported:
        return False, f"{kernel.__name__} not supported, due to {reason}"

    can_implement, reason = kernel.can_implement(config)
    if not can_implement:
        return False, f"{kernel.__name__} cannot not be used, due to {reason}"

    return True, ""


def select_quant_kernel(
    static: bool,
    group_shape: GroupShape,
    column_major_scales: bool = False,
    use_ue8m0: bool = False,
    num_token_padding: int | None = None,
    tma_aligned_scales: bool = False,
    return_native: bool = False,
) -> InputQuantKernel:
    """
    Select the appropriate input quantization kernel based on the current platform.

    Args:
        static: Whether to use static quantization
        group_shape: The shape of the quantization groups
        column_major_scales: Whether to use column-major scale storage
        use_ue8m0: Whether to use ue8m0 format
        num_token_padding: Optional number of token padding
        return_native: Returns native pytorch quantization implementation.

    Returns:
        The appropriate InputQuantKernel class for the current platform
    """
    config = InputQuantConfig(
        static=static,
        group_shape=group_shape,
        column_major_scales=column_major_scales,
        use_ue8m0=use_ue8m0,
        num_token_padding=num_token_padding,
        tma_aligned_scales=tma_aligned_scales,
    )

    if return_native:
        return PytorchInputQuantKernel(config)

    platform_enum = current_platform._enum

    # Check if the current platform has a priority kernel implementation
    if platform_enum in _PLATFORM_PRIORITIES:
        prioritized_kernel = _PLATFORM_PRIORITIES[platform_enum]
        can_dispatch, reason = is_supported_and_can_implement_kernel(
            prioritized_kernel, config
        )
        if can_dispatch:
            return prioritized_kernel(config)
        else:
            logger.warning_once(f"{reason}")
            fall_back_kernels = prioritized_kernel.ordered_fallback_kernels()
            for kernel in fall_back_kernels:
                can_dispatch, reason = is_supported_and_can_implement_kernel(
                    kernel, config
                )
                if can_dispatch:
                    logger.warning_once(
                        f"fall back quantization kernel: {kernel.__name__}"
                    )
                    return kernel(config)

            raise ValueError("None of the exsiting quantization kernel can be selected")

    # Fall back to the Pytorch implementation for all other platforms
    return PytorchInputQuantKernel(config)
