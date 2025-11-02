# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import TypeVar

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.kernels.scaled_mm.aiter import (
    AiterScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.cpu import (
    CPUScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass import (
    CutlassFP8ScaledMMLinearKernel,
    CutlassScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.pytorch import (
    ChannelWiseTorchScaledMMLinearKernel,
    PerTensorTorchScaledMMLinearKernel,
    RowWiseTorchScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.rocm import (
    ROCmScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import (  # noqa: E501
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
    ScaledMMLinearKernel,
    ScaledMMLinearLayerConfig,
    ScaledMMLinearQuantStrategy,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.triton import (
    TritonScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.xla import (
    XLAScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import PlatformEnum, current_platform

logger = init_logger(__name__)

# in priority/performance order (when available)
_POSSIBLE_INT8_KERNELS: dict[PlatformEnum, list[type[Int8ScaledMMLinearKernel]]] = {
    PlatformEnum.CPU: [CPUScaledMMLinearKernel],
    PlatformEnum.CUDA: [CutlassScaledMMLinearKernel],
    PlatformEnum.ROCM: [AiterScaledMMLinearKernel, TritonScaledMMLinearKernel],
    PlatformEnum.TPU: [XLAScaledMMLinearKernel],
}

# in priority/performance order (when available)
_POSSIBLE_FP8_KERNELS: dict[PlatformEnum, list[type[FP8ScaledMMLinearKernel]]] = {
    PlatformEnum.CUDA: [CutlassFP8ScaledMMLinearKernel],
    PlatformEnum.ROCM: [
        ROCmScaledMMLinearKernel,
        PerTensorTorchScaledMMLinearKernel,
        RowWiseTorchScaledMMLinearKernel,
        ChannelWiseTorchScaledMMLinearKernel,
    ],
}

_KernelT = TypeVar("_KernelT", bound=ScaledMMLinearKernel)
_KernelConfigT = TypeVar("_KernelConfigT", bound=ScaledMMLinearLayerConfig)


def choose_scaled_mm_linear_kernel(
    config: _KernelConfigT,
    possible_kernels: dict[PlatformEnum, list[type[_KernelT]]],
    compute_capability: int | None = None,
) -> type[_KernelT]:
    """
    Choose a _KernelT that can implement the given config for the
    given compute capability. Attempts to choose the best kernel in terms of
    performance.

    Args:
        config (_KernelConfigT): Description of the linear layer
            to be implemented.
        possible_kernels (dict[PlatformEnum, list[_KernelT]]): A
            dictionary of platforms and their list list of possible kernels.
        compute_capability (Optional[int], optional): The compute capability of
            the target device, if None uses `current_platform` to get the
            compute capability. Defaults to None.

    Raises:
        ValueError: If no kernel can implement the given config.

    Returns:
        _KernelT: Chosen kernel.
    """

    if compute_capability is None:
        _cc = current_platform.get_device_capability()
        if _cc is not None:
            compute_capability = _cc[0] * 10 + _cc[1]

    failure_reasons = []
    for kernel in possible_kernels[current_platform._enum]:
        if kernel.__name__ in os.environ.get("VLLM_DISABLED_KERNELS", "").split(","):
            failure_reasons.append(
                f" {kernel.__name__} disabled by environment variable"
            )
            continue

        # If the current platform uses compute_capability,
        # make sure the kernel supports the compute cability.
        if compute_capability is not None:
            kernel_min_capability = kernel.get_min_capability()
            if (
                kernel_min_capability is not None
                and kernel_min_capability > compute_capability
            ):
                failure_reasons.append(
                    f"{kernel.__name__} requires capability "
                    f"{kernel_min_capability}, current compute capability "
                    f"is {compute_capability}"
                )
                continue

        can_implement, failure_reason = kernel.can_implement(config)
        if can_implement:
            return kernel
        else:
            failure_reasons.append(
                f" {kernel.__name__} cannot implement due to: {failure_reason}"
            )

    raise ValueError(
        "Failed to find a kernel that can implement the "
        "ScaledMM linear layer. Reasons: \n" + "\n".join(failure_reasons)
    )


def init_fp8_linear_kernel(
    act_q_static: bool,
    act_q_group_shape: GroupShape,
    weight_quant_strategy: ScaledMMLinearQuantStrategy,
    out_dtype: torch.dtype,
    module_name: str,
) -> FP8ScaledMMLinearKernel:
    scaled_mm_linear_kernel_config = FP8ScaledMMLinearLayerConfig(
        is_static_input_scheme=act_q_static,
        weight_quant_strategy=weight_quant_strategy,
        activation_group_shape=act_q_group_shape,
        out_dtype=out_dtype,
    )

    kernel_type = choose_scaled_mm_linear_kernel(
        scaled_mm_linear_kernel_config,
        _POSSIBLE_FP8_KERNELS,
    )

    logger.info_once(
        "Selected %s for %s",
        kernel_type.__class__.__name__,
        module_name,
        scope="global",
    )

    return kernel_type(
        scaled_mm_linear_kernel_config,
        layer_param_names=["weight", "weight_scale", "input_scale", "input_scale_ub"],
    )


def init_int8_linear_kernel(
    is_channelwise: bool,
    is_static_input_scheme: bool,
    input_symmetric: bool,
    module_name: str,
) -> Int8ScaledMMLinearKernel:
    config = Int8ScaledMMLinearLayerConfig(
        is_channelwise=is_channelwise,
        is_static_input_scheme=is_static_input_scheme,
        input_symmetric=input_symmetric,
    )

    kernel_type = choose_scaled_mm_linear_kernel(
        config,
        _POSSIBLE_INT8_KERNELS,
    )

    logger.info_once(
        "Selected %s for %s",
        kernel_type.__class__.__name__,
        module_name,
        scope="global",
    )

    return kernel_type(
        config,
        layer_param_names=[
            "weight",
            "weight_scale",
            "input_scale",
            "input_zero_point",
            "azp_adj",
        ],
    )
