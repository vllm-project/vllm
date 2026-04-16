# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
This module re-exports linear kernel implementations to provide a
stable import interface during an ongoing reorganization. Upcoming
PRs will remove the scaled_mm and mixed_precision subdirectories
and reorganize kernels by provider (aiter, cutlass, flashinfer, etc.)
rather than by precision type. By centralizing exports here, we
minimize the need to update imports across other modules when the
internal structure changes. If you are adding a new kernel selector
or kernel implementation, add it to this __init__.py to maintain
import stability.
"""

from typing import TypeVar

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.base import (
    MMLinearKernel,
    MMLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision import (
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision.allspark import (
    AllSparkLinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.conch import (
    ConchLinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.cpu import (
    CPUWNA16LinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.cutlass import (
    CutlassW4A8LinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.dynamic_4bit import (
    Dynamic4bitLinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.exllama import (
    ExllamaLinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.machete import (
    MacheteLinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.marlin import (
    MarlinLinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
    TritonW4A16LinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.xpu import (
    XPUW4A8IntLinearKernel,
    XPUwNa16LinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8 import (
    Mxfp8LinearKernel,
    Mxfp8LinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mxfp8.emulation import (
    EmulationMxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutlassMxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8.marlin import (
    MarlinMxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8.xpu import (
    XPUMxFp8LinearKernel,
)
from vllm.model_executor.kernels.linear.nvfp4 import (
    NvFp4LinearKernel,
    NvFp4LinearLayerConfig,
)
from vllm.model_executor.kernels.linear.nvfp4.cutlass import (
    CutlassNvFp4LinearKernel,
)
from vllm.model_executor.kernels.linear.nvfp4.emulation import (
    EmulationNvFp4LinearKernel,
)
from vllm.model_executor.kernels.linear.nvfp4.fbgemm import (
    FbgemmNvFp4LinearKernel,
)
from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
    FlashInferCudnnNvFp4LinearKernel,
    FlashInferCutlassNvFp4LinearKernel,
    FlashInferTrtllmNvFp4LinearKernel,
)
from vllm.model_executor.kernels.linear.nvfp4.marlin import (
    MarlinNvFp4LinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm import (
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
    ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterFp8BlockScaledMMKernel,
    AiterInt8ScaledMMLinearKernel,
    AiterPerTokenFp8ScaledMMLinearKernel,
    AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.cpu import (
    CPUInt8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFp8BlockScaledMMKernel,
    CutlassFP8ScaledMMLinearKernel,
    CutlassInt8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
    DeepGemmFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
    FlashInferFp8DeepGEMMDynamicBlockScaledKernel,
    FlashInferFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
    MarlinFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    ChannelWiseTorchFP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
    RowWiseTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.rocm import (
    ROCmFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    TritonFp8BlockScaledMMKernel,
    TritonInt8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.xpu import (
    XPUFP8ScaledMMLinearKernel,
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
        MarlinFP8ScaledMMLinearKernel,
        FlashInferFP8ScaledMMLinearKernel,
        CutlassFP8ScaledMMLinearKernel,
        PerTensorTorchFP8ScaledMMLinearKernel,
        ChannelWiseTorchFP8ScaledMMLinearKernel,
    ],
    PlatformEnum.ROCM: [
        AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
        AiterPerTokenFp8ScaledMMLinearKernel,
        ROCmFP8ScaledMMLinearKernel,
        PerTensorTorchFP8ScaledMMLinearKernel,
        RowWiseTorchFP8ScaledMMLinearKernel,
        ChannelWiseTorchFP8ScaledMMLinearKernel,
    ],
    PlatformEnum.CPU: [
        PerTensorTorchFP8ScaledMMLinearKernel,
        ChannelWiseTorchFP8ScaledMMLinearKernel,
    ],
    PlatformEnum.XPU: [
        XPUFP8ScaledMMLinearKernel,
    ],
}


# in priority/performance order (when available)
_POSSIBLE_FP8_BLOCK_KERNELS: dict[
    PlatformEnum, list[type[Fp8BlockScaledMMLinearKernel]]
] = {
    PlatformEnum.CUDA: [
        FlashInferFp8DeepGEMMDynamicBlockScaledKernel,
        DeepGemmFp8BlockScaledMMKernel,
        CutlassFp8BlockScaledMMKernel,
        TritonFp8BlockScaledMMKernel,
    ],
    PlatformEnum.ROCM: [
        AiterFp8BlockScaledMMKernel,
        TritonFp8BlockScaledMMKernel,
    ],
}

_POSSIBLE_WFP8A16_KERNELS: dict[PlatformEnum, list[type[FP8ScaledMMLinearKernel]]] = {
    PlatformEnum.CUDA: [
        MarlinFP8ScaledMMLinearKernel,
    ],
    PlatformEnum.ROCM: [
        # To be added
    ],
    PlatformEnum.CPU: [
        # To be added
    ],
    PlatformEnum.XPU: [
        XPUFP8ScaledMMLinearKernel,
    ],
}

# in priority/performance order (when available)
_POSSIBLE_KERNELS: dict[PlatformEnum, list[type[MPLinearKernel]]] = {
    PlatformEnum.CUDA: [
        CutlassW4A8LinearKernel,
        MacheteLinearKernel,
        AllSparkLinearKernel,
        MarlinLinearKernel,
        ConchLinearKernel,
        ExllamaLinearKernel,
    ],
    PlatformEnum.ROCM: [
        TritonW4A16LinearKernel,
        ConchLinearKernel,
        ExllamaLinearKernel,
    ],
    PlatformEnum.XPU: [
        XPUW4A8IntLinearKernel,
        XPUwNa16LinearKernel,
    ],
    PlatformEnum.CPU: [
        Dynamic4bitLinearKernel,
        CPUWNA16LinearKernel,
    ],
}

# in priority/performance order (when available)
_POSSIBLE_MXFP8_KERNELS: dict[PlatformEnum, list[type[Mxfp8LinearKernel]]] = {
    PlatformEnum.CUDA: [
        FlashInferCutlassMxfp8LinearKernel,
        MarlinMxfp8LinearKernel,
        EmulationMxfp8LinearKernel,
    ],
    PlatformEnum.ROCM: [
        EmulationMxfp8LinearKernel,
    ],
    PlatformEnum.XPU: [
        XPUMxFp8LinearKernel,
        EmulationMxfp8LinearKernel,
    ],
}

_POSSIBLE_NVFP4_KERNELS: dict[PlatformEnum, list[type[NvFp4LinearKernel]]] = {
    PlatformEnum.CUDA: [
        FlashInferCutlassNvFp4LinearKernel,
        CutlassNvFp4LinearKernel,
        MarlinNvFp4LinearKernel,
        FlashInferTrtllmNvFp4LinearKernel,
        FlashInferCudnnNvFp4LinearKernel,
        FbgemmNvFp4LinearKernel,
        EmulationNvFp4LinearKernel,
    ],
    PlatformEnum.ROCM: [
        EmulationNvFp4LinearKernel,
    ],
}

# TODO make all kernels inherit from MMLinearKernel
# then bound _KernelT only to MMLinearKernel
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
    input_dtype: torch.dtype,
    out_dtype: torch.dtype,
    weight_shape: tuple[int, int],
    force_kernel: type[FP8ScaledMMLinearKernel] | None = None,
    module_name: str | None = None,
) -> FP8ScaledMMLinearKernel | Fp8BlockScaledMMLinearKernel:
    scaled_mm_linear_kernel_config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=weight_quant_key,
        activation_quant_key=activation_quant_key,
        input_dtype=input_dtype,
        out_dtype=out_dtype,
        weight_shape=weight_shape,
    )

    if activation_quant_key.scale.group_shape.is_per_group():
        kernel_type = choose_scaled_mm_linear_kernel(
            config=scaled_mm_linear_kernel_config,
            possible_kernels=_POSSIBLE_FP8_BLOCK_KERNELS,  # type: ignore[misc]
            force_kernel=force_kernel,
        )
        if module_name:
            logger.info_once(
                "Selected %s for %s",
                kernel_type.__name__,
                module_name,
                scope="global",
            )

        return kernel_type(
            scaled_mm_linear_kernel_config,
        )

    else:
        kernel_type = choose_scaled_mm_linear_kernel(
            config=scaled_mm_linear_kernel_config,
            possible_kernels=_POSSIBLE_FP8_KERNELS,  # type: ignore[misc]
            force_kernel=force_kernel,
        )
        if module_name:
            logger.info_once(
                "Selected %s for %s",
                kernel_type.__name__,
                module_name,
                scope="global",
            )

        return kernel_type(
            scaled_mm_linear_kernel_config,
            layer_param_names=[
                "weight",
                "weight_scale",
                "input_scale",
                "input_scale_ub",
            ],
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
        layer_param_names=[
            "weight",
            "weight_scale",
            "input_scale",
            "input_zero_point",
            "azp_adj",
        ],
    )


def choose_mp_linear_kernel(
    config: MPLinearLayerConfig, compute_capability: int | None = None
) -> type[MPLinearKernel]:
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
    if compute_capability is None:
        if current_platform is None:
            raise ValueError("Cannot determine compute capability")
        _cc = current_platform.get_device_capability()
        if _cc is not None:
            compute_capability = _cc[0] * 10 + _cc[1]

    failure_reasons = []
    for kernel in _POSSIBLE_KERNELS[current_platform._enum]:
        if kernel.__name__ in envs.VLLM_DISABLED_KERNELS:
            failure_reasons.append(
                f" {kernel.__name__} disabled by environment variable"
            )
            continue
        if (
            compute_capability is not None
            and kernel.get_min_capability() > compute_capability
        ):
            failure_reasons.append(
                f"{kernel.__name__} requires capability "
                f"{kernel.get_min_capability()}, current compute "
                f" capability is {compute_capability}"
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
        "WNA16 linear layer. Reasons: \n" + "\n".join(failure_reasons)
    )


def init_mxfp8_linear_kernel() -> Mxfp8LinearKernel:
    """Select and instantiate the best MXFP8 linear kernel for the
    current platform."""
    config = Mxfp8LinearLayerConfig()

    platform = current_platform._enum
    possible = _POSSIBLE_MXFP8_KERNELS.get(platform, [])

    failure_reasons = []
    for kernel_cls in possible:
        if kernel_cls.__name__ in envs.VLLM_DISABLED_KERNELS:
            failure_reasons.append(
                f" {kernel_cls.__name__} disabled by environment variable"
            )
            continue

        is_supported, reason = kernel_cls.is_supported()
        if not is_supported:
            failure_reasons.append(f"{kernel_cls.__name__}: {reason}")
            continue

        can_implement, reason = kernel_cls.can_implement(config)
        if not can_implement:
            failure_reasons.append(f"{kernel_cls.__name__}: {reason}")
            continue

        logger.info_once("Using %s for MXFP8 GEMM", kernel_cls.__name__)
        return kernel_cls(config)

    raise ValueError(
        "Failed to find a kernel that can implement the "
        "MXFP8 linear layer. Reasons: \n" + "\n".join(failure_reasons)
    )


def init_wfp8_a16_linear_kernel(
    weight_quant_key: QuantKey,
    activation_quant_key: QuantKey,
    weight_shape: tuple[int, int],
    input_dtype: torch.dtype,
    out_dtype: torch.dtype,
    force_kernel: type[FP8ScaledMMLinearKernel] | None = None,
    module_name: str | None = None,
) -> FP8ScaledMMLinearKernel:
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=weight_quant_key,
        activation_quant_key=activation_quant_key,
        weight_shape=weight_shape,
        input_dtype=input_dtype,
        out_dtype=out_dtype,
    )

    kernel_type = choose_scaled_mm_linear_kernel(
        config, _POSSIBLE_WFP8A16_KERNELS, force_kernel=force_kernel
    )

    if module_name:
        logger.info_once(
            "Selected %s for %s",
            kernel_type.__name__,
            module_name,
            scope="global",
        )

    return kernel_type(
        config,
        layer_param_names=["weight", "weight_scale", "input_scale", "input_scale_ub"],
    )


# Maps VLLM_NVFP4_GEMM_BACKEND env var values to kernel classes.
_NVFP4_BACKEND_TO_KERNEL: dict[str, type[NvFp4LinearKernel]] = {
    "flashinfer-cutlass": FlashInferCutlassNvFp4LinearKernel,
    "cutlass": CutlassNvFp4LinearKernel,
    "marlin": MarlinNvFp4LinearKernel,
    "flashinfer-trtllm": FlashInferTrtllmNvFp4LinearKernel,
    "flashinfer-cudnn": FlashInferCudnnNvFp4LinearKernel,
    "emulation": EmulationNvFp4LinearKernel,
}


def init_nvfp4_linear_kernel() -> NvFp4LinearKernel:
    """Select and instantiate the best NVFP4 linear kernel for the
    current platform."""
    config = NvFp4LinearLayerConfig()

    # Env-var overrides.
    force_kernel: type[NvFp4LinearKernel] | None = None
    if envs.VLLM_BATCH_INVARIANT:
        logger.info_once(
            "VLLM_BATCH_INVARIANT forces NVFP4 linear to use the "
            "emulation backend for deterministic execution."
        )
        force_kernel = EmulationNvFp4LinearKernel
    elif envs.VLLM_USE_FBGEMM:
        force_kernel = FbgemmNvFp4LinearKernel
    elif envs.VLLM_USE_NVFP4_CT_EMULATIONS:
        force_kernel = EmulationNvFp4LinearKernel
    elif envs.VLLM_NVFP4_GEMM_BACKEND is not None:
        backend_name = envs.VLLM_NVFP4_GEMM_BACKEND
        force_kernel = _NVFP4_BACKEND_TO_KERNEL.get(backend_name)
        if force_kernel is None:
            raise ValueError(
                f"Unknown VLLM_NVFP4_GEMM_BACKEND={backend_name!r}. "
                f"Valid choices: {list(_NVFP4_BACKEND_TO_KERNEL.keys())}"
            )

    if force_kernel is not None:
        is_supported, reason = force_kernel.is_supported()
        if not is_supported:
            raise ValueError(
                f"Forced NVFP4 kernel {force_kernel.__name__} is not "
                f"supported: {reason}"
            )
        logger.info_once("Using %s for NVFP4 GEMM", force_kernel.__name__)
        return force_kernel(config)

    # Auto-select from registry.
    platform = current_platform._enum
    possible = _POSSIBLE_NVFP4_KERNELS.get(platform, [])

    failure_reasons = []
    for kernel_cls in possible:
        if kernel_cls.__name__ in envs.VLLM_DISABLED_KERNELS:
            failure_reasons.append(
                f" {kernel_cls.__name__} disabled by environment variable"
            )
            continue

        is_supported, reason = kernel_cls.is_supported()
        if not is_supported:
            failure_reasons.append(f"{kernel_cls.__name__}: {reason}")
            continue

        can_implement, reason = kernel_cls.can_implement(config)
        if not can_implement:
            failure_reasons.append(f"{kernel_cls.__name__}: {reason}")
            continue

        if kernel_cls is EmulationNvFp4LinearKernel and failure_reasons:
            logger.warning_once(
                "NVFP4 linear falling back to the slow and unoptimized "
                "emulation backend as no optimized backend is available "
                "(unavailable reasons:\n - %s\n). "
                "In case you expect one of these backends to be used, "
                "please verify your environment.",
                "\n - ".join(failure_reasons),
            )

        logger.info_once("Using %s for NVFP4 GEMM", kernel_cls.__name__)
        return kernel_cls(config)

    raise ValueError(
        "Failed to find a kernel that can implement the "
        "NVFP4 linear layer. Reasons: \n" + "\n".join(failure_reasons)
    )


def register_linear_kernel(
    kernel_class: type,
    platform: PlatformEnum,
    kernel_type: str = "mp",
) -> None:
    """
    Register a new linear kernel class to be considered in kernel selection.

    Args:
        kernel_class (type): The kernel class to register.
        platform (PlatformEnum): The platform for which this kernel is applicable.
        kernel_type (str): The type of the kernel, either "mp", "int8", or "fp8".
            Defaults to "mp".

    Raises:
        ValueError: If the kernel_type is not recognized.
    """
    if kernel_type == "mp":
        if platform not in _POSSIBLE_KERNELS:
            _POSSIBLE_KERNELS[platform] = []
        _POSSIBLE_KERNELS[platform].append(kernel_class)
    elif kernel_type == "int8":
        if platform not in _POSSIBLE_INT8_KERNELS:
            _POSSIBLE_INT8_KERNELS[platform] = []
        _POSSIBLE_INT8_KERNELS[platform].append(kernel_class)
    elif kernel_type == "fp8":
        if platform not in _POSSIBLE_FP8_KERNELS:
            _POSSIBLE_FP8_KERNELS[platform] = []
        _POSSIBLE_FP8_KERNELS[platform].append(kernel_class)
    elif kernel_type == "mxfp8":
        if platform not in _POSSIBLE_MXFP8_KERNELS:
            _POSSIBLE_MXFP8_KERNELS[platform] = []
        _POSSIBLE_MXFP8_KERNELS[platform].append(kernel_class)
    elif kernel_type == "nvfp4":
        if platform not in _POSSIBLE_NVFP4_KERNELS:
            _POSSIBLE_NVFP4_KERNELS[platform] = []
        _POSSIBLE_NVFP4_KERNELS[platform].append(kernel_class)
    else:
        raise ValueError(f"Unrecognized kernel type: {kernel_type}")


__all__ = [
    "init_fp8_linear_kernel",
    "init_int8_linear_kernel",
    "init_nvfp4_linear_kernel",
    "choose_mp_linear_kernel",
    "register_linear_kernel",
    "init_wfp8_a16_linear_kernel",
    "FP8ScaledMMLinearKernel",
    "Int8ScaledMMLinearKernel",
    "ScaledMMLinearKernel",
    "FP8ScaledMMLinearLayerConfig",
    "Int8ScaledMMLinearLayerConfig",
    "ScaledMMLinearLayerConfig",
    "AiterPreshuffledPerTokenFp8ScaledMMLinearKernel",
    "AiterPerTokenFp8ScaledMMLinearKernel",
    "NvFp4LinearKernel",
    "NvFp4LinearLayerConfig",
    "AiterInt8ScaledMMLinearKernel",
    "CPUInt8ScaledMMLinearKernel",
    "CutlassFP8ScaledMMLinearKernel",
    "CutlassInt8ScaledMMLinearKernel",
    "FlashInferFP8ScaledMMLinearKernel",
    "ChannelWiseTorchFP8ScaledMMLinearKernel",
    "PerTensorTorchFP8ScaledMMLinearKernel",
    "RowWiseTorchFP8ScaledMMLinearKernel",
    "ROCmFP8ScaledMMLinearKernel",
    "TritonInt8ScaledMMLinearKernel",
    "MPLinearKernel",
    "MPLinearLayerConfig",
    "AllSparkLinearKernel",
    "ConchLinearKernel",
    "CPUWNA16LinearKernel",
    "CutlassW4A8LinearKernel",
    "Dynamic4bitLinearKernel",
    "ExllamaLinearKernel",
    "MacheteLinearKernel",
    "MarlinLinearKernel",
    "TritonW4A16LinearKernel",
    "XPUW4A8IntLinearKernel",
    "XPUwNa16LinearKernel",
    "init_mxfp8_linear_kernel",
    "Mxfp8LinearKernel",
    "Mxfp8LinearLayerConfig",
    "FlashInferCutlassMxfp8LinearKernel",
    "MarlinMxfp8LinearKernel",
    "XPUMxFp8LinearKernel",
    "EmulationMxfp8LinearKernel",
    "CutlassNvFp4LinearKernel",
    "EmulationNvFp4LinearKernel",
    "FbgemmNvFp4LinearKernel",
    "FlashInferCutlassNvFp4LinearKernel",
    "FlashInferTrtllmNvFp4LinearKernel",
    "FlashInferCudnnNvFp4LinearKernel",
    "MarlinNvFp4LinearKernel",
    "_KernelT",
    "DeepGemmFp8BlockScaledMMKernel",
    "FlashInferFp8DeepGEMMDynamicBlockScaledKernel",
]
