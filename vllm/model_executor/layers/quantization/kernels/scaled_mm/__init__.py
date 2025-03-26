# SPDX-License-Identifier: Apache-2.0

import os
from typing import Dict, List, Optional, Type

from vllm.model_executor.layers.quantization.kernels.scaled_mm.aiter import (
    AiterScaledMMLinearKernel)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass import (
    CutlassScaledMMLinearKernel)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import (  # noqa: E501
    ScaledMMLinearKernel, ScaledMMLinearLayerConfig)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.triton import (
    TritonScaledMMLinearKernel)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.xla import (
    XLAScaledMMLinearKernel)
from vllm.platforms import PlatformEnum, current_platform

# in priority/performance order (when available)
_POSSIBLE_KERNELS: Dict[PlatformEnum, List[Type[ScaledMMLinearKernel]]] = {
    PlatformEnum.CPU: [CutlassScaledMMLinearKernel],
    PlatformEnum.CUDA: [CutlassScaledMMLinearKernel],
    PlatformEnum.ROCM: [AiterScaledMMLinearKernel, TritonScaledMMLinearKernel],
    PlatformEnum.TPU: [XLAScaledMMLinearKernel],
}


def choose_scaled_mm_linear_kernel(
        config: ScaledMMLinearLayerConfig,
        compute_capability: Optional[int] = None
) -> Type[ScaledMMLinearKernel]:
    """
    Choose an ScalledMMLinearKernel that can implement the given config for the 
    given compute capability. Attempts to choose the best kernel in terms of 
    performance.

    Args:
        config (ScaledMMLinearLayerConfig): Description of the linear layer 
            to be implemented.
        compute_capability (Optional[int], optional): The compute capability of
            the target device, if None uses `current_platform` to get the 
            compute capability. Defaults to None.

    Raises:
        ValueError: If no kernel can implement the given config.

    Returns:
        Type[ScaledMMLinearKernel]: Chosen kernel.
    """

    failure_reasons = []
    for kernel in _POSSIBLE_KERNELS[current_platform._enum]:
        if kernel.__name__ in os.environ.get("VLLM_DISABLED_KERNELS", "")\
            .split(","):
            failure_reasons.append(
                f' {kernel.__name__} disabled by environment variable')
            continue

        is_supported, reason = kernel.is_supported(compute_capability)
        if not is_supported:
            failure_reasons.append(
                f' {kernel.__name__} not supported: {reason}')
            continue

        can_implement, failure_reason = kernel.can_implement(config)
        if not can_implement:
            failure_reasons.append(
                f' {kernel.__name__} cannot implement given config ({config}): '
                f'{failure_reason}')
            continue

        # Kernel enabled, supported, and can implement scheme!
        return kernel

    raise ValueError(
        "Failed to find a kernel that can implement the "\
        "ScaledMM linear layer. Reasons: \n"
        + '\n'.join(failure_reasons))
