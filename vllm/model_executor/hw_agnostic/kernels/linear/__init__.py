# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TypeVar

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.hw_agnostic.quantization.quant_keys import QuantKey
from vllm.platforms import PlatformEnum, current_platform

from .base import MMLinearKernel, MMLinearLayerConfig
from .scaled_mm import (
    ChannelWiseTorchFP8ScaledMMLinearKernel,
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    ScaledMMLinearKernel,
    TritonFp8BlockScaledMMKernel,
)

logger = init_logger(__name__)

__all__ = [
    "ChannelWiseTorchFP8ScaledMMLinearKernel",
    "FP8ScaledMMLinearKernel",
    "FP8ScaledMMLinearLayerConfig",
    "Fp8BlockScaledMMLinearKernel",
    "MMLinearKernel",
    "MMLinearLayerConfig",
    "ScaledMMLinearKernel",
    "TritonFp8BlockScaledMMKernel",
    "init_fp8_linear_kernel",
]


def _get_linear_backend() -> str:
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    if config is not None:
        return config.kernel_config.linear_backend
    return "auto"


# Triton + native-PyTorch are the only portable kernels on the
# hw-agnostic path; HW-specific kernel families (FlashInfer/DeepGEMM/
# Cutlass/Marlin) load CUDA-binary code that an OOT host cannot run.
_LINEAR_BACKEND_KERNEL_MAP: dict[str, set[type]] = {
    "triton": {TritonFp8BlockScaledMMKernel},
    "torch": {ChannelWiseTorchFP8ScaledMMLinearKernel},
}


def _filter_kernels_by_backend(
    backend: str,
    kernels: list[type],
) -> list[type]:
    backend_kernels = _LINEAR_BACKEND_KERNEL_MAP.get(backend, set())
    return [k for k in kernels if k in backend_kernels]


# Block-scaled FP8 priority list. CUDA only — the hw-agnostic path always
# runs on a platform whose ``_enum`` resolves to CUDA (OOT plugins inherit
# from NvmlCudaPlatform). ChannelWise torch is the portable fallback.
_POSSIBLE_FP8_BLOCK_KERNELS: dict[
    PlatformEnum, list[type[Fp8BlockScaledMMLinearKernel | FP8ScaledMMLinearKernel]]
] = {
    PlatformEnum.CUDA: [
        TritonFp8BlockScaledMMKernel,
        ChannelWiseTorchFP8ScaledMMLinearKernel,
    ],
}


_KernelT = TypeVar("_KernelT", bound=ScaledMMLinearKernel | MMLinearKernel)
_KernelConfigT = TypeVar("_KernelConfigT", bound=MMLinearLayerConfig)


def is_supported_and_can_implement_kernel(
    kernel: type[_KernelT], config: _KernelConfigT, compute_capability: int | None
) -> tuple[bool, str]:
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
        return False, f"{kernel.__name__} {failure_reason}."

    return True, ""


def choose_scaled_mm_linear_kernel(
    config: _KernelConfigT,
    possible_kernels: dict[PlatformEnum, list[type[_KernelT]]],
    compute_capability: int | None = None,
) -> type[_KernelT]:
    failure_reason_list = []

    platform_kernels = possible_kernels[current_platform._enum]

    linear_backend = _get_linear_backend()
    if linear_backend != "auto":
        filtered = _filter_kernels_by_backend(linear_backend, platform_kernels)
        if not filtered:
            raise ValueError(
                f"--linear-backend={linear_backend} was requested but no "
                f"'{linear_backend}' kernel exists for this layer type."
            )
        platform_kernels = filtered

    for kernel in platform_kernels:
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
    input_dtype: torch.dtype,
    out_dtype: torch.dtype,
    weight_shape: tuple[int, int],
    module_name: str | None = None,
) -> Fp8BlockScaledMMLinearKernel | FP8ScaledMMLinearKernel:
    """Select an FP8 block-scaled linear kernel."""
    if not activation_quant_key.scale.group_shape.is_per_group():
        raise NotImplementedError(
            "hw-agnostic only supports block-scaled FP8 (per-group "
            "activation scales); per-tensor / per-token FP8 is not supported."
        )

    scaled_mm_linear_kernel_config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=weight_quant_key,
        activation_quant_key=activation_quant_key,
        input_dtype=input_dtype,
        out_dtype=out_dtype,
        weight_shape=weight_shape,
    )

    kernel_type = choose_scaled_mm_linear_kernel(
        config=scaled_mm_linear_kernel_config,
        possible_kernels=_POSSIBLE_FP8_BLOCK_KERNELS,
    )
    if module_name:
        logger.info_once(
            "Selected %s for %s",
            kernel_type.__name__,
            module_name,
            scope="global",
        )

    if issubclass(kernel_type, FP8ScaledMMLinearKernel):
        return kernel_type(
            scaled_mm_linear_kernel_config,
            layer_param_names=[
                "weight",
                "weight_scale",
                "input_scale",
                "input_scale_ub",
            ],
        )

    return kernel_type(scaled_mm_linear_kernel_config)
