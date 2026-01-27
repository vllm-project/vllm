# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.kernels.wFP8a16.fp8_marlin import (
    FP8MarlinLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.wFP8a16.WFP8A16_kernel import (
    FP8WoQLinearKernel,
    FP8WoQLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)
from vllm.platforms import PlatformEnum, current_platform

logger = init_logger(__name__)

_POSSIBLE_WFP8A16_KERNELS: dict[PlatformEnum, list[type[FP8WoQLinearKernel]]] = {
    PlatformEnum.CUDA: [FP8MarlinLinearKernel],
}


def is_supported_and_can_implement_kernel(
    kernel: type[FP8WoQLinearKernel],
    config: FP8WoQLinearLayerConfig,
    compute_capability: int | None,
) -> tuple[bool, str]:
    # TODO: Fetch `VLLM_DISABLED_KERNELS` from vllm.envs instead.
    if kernel.__name__ in envs.VLLM_DISABLED_KERNELS:
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


def choose_mp_linear_kernel(
    config: FP8WoQLinearLayerConfig,
    possible_kernels: dict[PlatformEnum, list[type[FP8WoQLinearKernel]]],
    compute_capability: int | None = None,
    force_kernel: type[FP8WoQLinearKernel] | None = None,
) -> type[FP8WoQLinearKernel]:
    """
    Choose an MPLinearKernel that can implement the given config for the given
     compute capability. Attempts to choose the best kernel in terms of
     performance.

    Args:
        config (MPLinearLayerConfig): Description of the linear layer to be
            implemented.
        compute_capability (Optional[int], optional): The compute capability of
            the target device, if None uses `current_platform` to get
            the compute capability. Defaults to None.

    Raises:
        ValueError: If no kernel can implement the given config.

    Returns:
        type[MPLinearKernel]: Chosen kernel.
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


def init_wfp8a16_kernels(
    weight_quant_key: QuantKey,
    input_dtype: torch.dtype,
    is_block_quant: bool = False,
    force_kernel: type[FP8WoQLinearKernel] | None = None,
    module_name: str | None = None,
) -> FP8WoQLinearKernel:
    """Initialize wFP8a16 kernels."""
    kernel_config = FP8WoQLinearLayerConfig(
        weight_quant_key=weight_quant_key,
        input_dtype=input_dtype,
        is_block_quant=is_block_quant,
    )
    kernel_type = choose_mp_linear_kernel(
        kernel_config,
        possible_kernels=_POSSIBLE_WFP8A16_KERNELS,
        force_kernel=force_kernel,
    )
    if module_name:
        logger.info_once(
            "Selected %s for %s",
            kernel_type.__name__,
            module_name,
            scope="global",
        )
    return kernel_type(kernel_config)
