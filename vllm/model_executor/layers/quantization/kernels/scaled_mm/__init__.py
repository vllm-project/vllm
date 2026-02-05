# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import TypeVar

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.kernels.base import (
    MMLinearKernel,
    MMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.aiter import (
    AiterFp8BlockScaledMMKernel,
    AiterInt8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.BlockScaledMMLinearKernel import (  # noqa: E501
    Fp8BlockScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.cpu import (
    CPUInt8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.cuda import (
    CudaFp8BlockScaledMMKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass import (
    CutlassFp8BlockScaledMMKernel,
    CutlassFP8ScaledMMLinearKernel,
    CutlassInt8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.flashinfer import (
    FlashInferFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.pytorch import (
    ChannelWiseTorchFP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
    RowWiseTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.rocm import (
    ROCmFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import (  # noqa: E501
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.triton import (
    TritonFp8BlockScaledMMKernel,
    TritonInt8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import PlatformEnum, current_platform

logger = init_logger(__name__)


# in priority/performance order (when available)
_POSSIBLE_INT8_KERNELS: dict[PlatformEnum, list[type[Int8ScaledMMLinearKernel]]] = {
    PlatformEnum.CPU: [CPUInt8ScaledMMLinearKernel],
    PlatformEnum.CUDA: [
        CutlassInt8ScaledMMLinearKernel,
        TritonInt8ScaledMMLinearKernel,
    ],
    PlatformEnum.ROCM: [AiterInt8ScaledMMLinearKernel, TritonInt8ScaledMMLinearKernel],
}


# in priority/performance order (when available)
_POSSIBLE_FP8_KERNELS: dict[PlatformEnum, list[type[FP8ScaledMMLinearKernel]]] = {
    PlatformEnum.CUDA: [
        FlashInferFP8ScaledMMLinearKernel,
        CutlassFP8ScaledMMLinearKernel,
        PerTensorTorchFP8ScaledMMLinearKernel,
        ChannelWiseTorchFP8ScaledMMLinearKernel,
    ],
    PlatformEnum.ROCM: [
        ROCmFP8ScaledMMLinearKernel,
        PerTensorTorchFP8ScaledMMLinearKernel,
        RowWiseTorchFP8ScaledMMLinearKernel,
        ChannelWiseTorchFP8ScaledMMLinearKernel,
    ],
    PlatformEnum.CPU: [
        PerTensorTorchFP8ScaledMMLinearKernel,
        ChannelWiseTorchFP8ScaledMMLinearKernel,
    ],
}


# in priority/performance order (when available)
_POSSIBLE_FP8_BLOCK_KERNELS: dict[
    PlatformEnum, list[type[Fp8BlockScaledMMLinearKernel]]
] = {
    PlatformEnum.CUDA: [
        CudaFp8BlockScaledMMKernel,
        CutlassFp8BlockScaledMMKernel,
        TritonFp8BlockScaledMMKernel,
    ],
    PlatformEnum.ROCM: [
        AiterFp8BlockScaledMMKernel,
        TritonFp8BlockScaledMMKernel,
    ],
}

_KernelT = TypeVar("_KernelT", bound=MMLinearKernel)
_KernelConfigT = TypeVar("_KernelConfigT", bound=MMLinearLayerConfig)


def is_supported_and_can_implement_kernel(
    kernel: type[_KernelT],
    config: _KernelConfigT,
    compute_capability: int | None = None,
) -> tuple[bool, str]:
    # TODO: Fetch `VLLM_DISABLED_KERNELS` from vllm.envs instead.
    if kernel.__name__ in os.environ.get("VLLM_DISABLED_KERNELS", "").split(","):
        return False, f" {kernel.__name__} is disabled by environment variable"

    if compute_capability is None:
        _cc = current_platform.get_device_capability()
        if _cc is not None:
            compute_capability = _cc[0] * 10 + _cc[1]

    is_supported, failure_reason = kernel.is_supported(compute_capability)
    if not is_supported:
        return False, f"{kernel.__name__} {failure_reason}."

    can_implement, failure_reason = kernel.can_implement(config)
    if not can_implement:
        return (
            False,
            f"{kernel.__name__} {failure_reason}.",
        )

    return True, ""


def choose_scaled_mm_linear_kernel(
    config: _KernelConfigT,
    possible_kernels: dict[PlatformEnum, list[type[_KernelT]]],
    compute_capability: int | None = None,
    force_kernel: type[_KernelT] | None = None,
) -> type[_KernelT]:
    """
    Choose a _KernelT that can implement the given config for the
    given compute capability. Attempts to choose the best kernel in terms of
    performance.

    Args:
        config (_KernelConfigT): Description of the linear layer
            to be implemented.
        possible_kernels (dict[PlatformEnum, list[_KernelT]]): A
            dictionary of platforms and their list of possible kernels.
        compute_capability (Optional[int], optional): The compute capability of
            the target device, if None uses `current_platform` to get the
            compute capability. Defaults to None.
        force_kernel (Optional[type[_KernelT]]): An Optional forced kernel to override
            the possible_kernels if it can be implemented. If None, it will only try the
            possible kernels.

    Raises:
        ValueError: If no kernel can implement the given config.

    Returns:
        _KernelT: Chosen kernel.
    """

    failure_reason_list = []

    if force_kernel is not None:
        can_implement, failure_reason = is_supported_and_can_implement_kernel(
            force_kernel, config, compute_capability
        )
        if can_implement:
            return force_kernel

        logger.info_once(
            "Tried to force %s, but the kernel couldn't be implemented",
            force_kernel.__name__,
            scope="global",
        )

    for kernel in possible_kernels[current_platform._enum]:
        is_supported_and_can_implement, failure_reason = (
            is_supported_and_can_implement_kernel(kernel, config, compute_capability)
        )
        if is_supported_and_can_implement:
            return kernel
        failure_reason_list.append(failure_reason)

    raise ValueError(
        "Failed to find a kernel that can implement the "
        "ScaledMM linear layer. Reasons: \n" + "\n".join(failure_reason_list)
    )


def init_fp8_linear_kernel(
    activation_quant_key: QuantKey,
    weight_quant_key: QuantKey,
    out_dtype: torch.dtype,
    force_kernel: type[_KernelT] | None = None,
    module_name: str | None = None,
) -> FP8ScaledMMLinearKernel | Fp8BlockScaledMMLinearKernel:
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=weight_quant_key,
        activation_quant_key=activation_quant_key,
        out_dtype=out_dtype,
    )

    if activation_quant_key.scale.group_shape.is_per_group():
        kernel_type = choose_scaled_mm_linear_kernel(
            config=config,
            possible_kernels=_POSSIBLE_FP8_BLOCK_KERNELS,  # type: ignore[misc]
            force_kernel=force_kernel,
        )
    else:
        kernel_type = choose_scaled_mm_linear_kernel(
            config=config,
            possible_kernels=_POSSIBLE_FP8_KERNELS,  # type: ignore[misc]
            force_kernel=force_kernel,
        )

    logger.info_once(
        "Selected %s for %s",
        kernel_type.__name__,
        module_name,
        scope="global",
    )
    return kernel_type(
        config,
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
        kernel_type.__name__,
        module_name,
        scope="global",
    )
    return kernel_type(
        config,
    )
