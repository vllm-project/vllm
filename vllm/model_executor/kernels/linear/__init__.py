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

from typing import Generic, Protocol, TypeVar

import torch

import vllm.envs as envs
import vllm.model_executor.kernels.linear.aiter.w8a8 as aiter_w8a8
import vllm.model_executor.kernels.linear.base.w8a8 as w8a8
import vllm.model_executor.kernels.linear.cpu.w8a8 as cpu_w8a8
import vllm.model_executor.kernels.linear.cutlass.w8a8 as cutlass_w8a8
import vllm.model_executor.kernels.linear.flashinfer.w8a8 as flashinfer_w8a8
import vllm.model_executor.kernels.linear.hip.w8a8 as hip_w8a8
import vllm.model_executor.kernels.linear.pytorch.w8a8 as pytorch_w8a8
import vllm.model_executor.kernels.linear.triton.w8a8 as triton_w8a8
import vllm.model_executor.kernels.linear.xpu.w8a8 as xpu_w8a8
from vllm.logger import init_logger
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
from vllm.model_executor.kernels.linear.mixed_precision.xpu import (
    XPUwNa16LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import PlatformEnum, current_platform

logger = init_logger(__name__)

_FpKernelT = TypeVar("_FpKernelT", bound=w8a8.FpKernel)
_IntKernelT = TypeVar("_IntKernelT", bound=w8a8.IntKernel)


class _FpW8A8KernelModule(Protocol, Generic[_FpKernelT]):
    """Protocol for w8a8 kernel modules that expose an FpKernel class."""

    FpKernel: type[_FpKernelT]


class _IntW8A8KernelModule(Protocol, Generic[_IntKernelT]):
    """Protocol for w8a8 kernel modules that expose an IntKernel class."""

    IntKernel: type[_IntKernelT]


# in priority/performance order (when available)
_POSSIBLE_FP_W8A8_KERNEL_MODULES: dict[PlatformEnum, list[_FpW8A8KernelModule]] = {
    PlatformEnum.CUDA: [flashinfer_w8a8, cutlass_w8a8, pytorch_w8a8],
    PlatformEnum.ROCM: [hip_w8a8, pytorch_w8a8],
    PlatformEnum.CPU: [pytorch_w8a8],
    PlatformEnum.XPU: [xpu_w8a8],
}

# in priority/performance order (when available)
_POSSIBLE_INT_W8A8_KERNEL_MODULES: dict[PlatformEnum, list[_IntW8A8KernelModule]] = {
    PlatformEnum.CPU: [cpu_w8a8],
    PlatformEnum.CUDA: [cutlass_w8a8, triton_w8a8],
    PlatformEnum.ROCM: [aiter_w8a8, triton_w8a8],
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
        ConchLinearKernel,
        ExllamaLinearKernel,
    ],
    PlatformEnum.XPU: [
        XPUwNa16LinearKernel,
    ],
    PlatformEnum.CPU: [
        Dynamic4bitLinearKernel,
        CPUWNA16LinearKernel,
    ],
}

_KernelT = TypeVar("_KernelT", bound=w8a8.Kernel)
_KernelConfigT = TypeVar("_KernelConfigT", bound=w8a8.KernelConfig)
_KernelModuleT = TypeVar("_KernelModuleT", _FpW8A8KernelModule, _IntW8A8KernelModule)


def choose_w8a8_linear_kernel(
    config: _KernelConfigT,
    possible_kernel_modules: dict[PlatformEnum, list[_KernelModuleT]],
    kernel_attr: str,  # "FpKernel" or "IntKernel"
    compute_capability: int | None = None,
    force_kernel: type[_KernelT] | None = None,
) -> type[_KernelT]:
    if compute_capability is None:
        _cc = current_platform.get_device_capability()
        if _cc is not None:
            compute_capability = _cc[0] * 10 + _cc[1]

    failure_reason_list = []

    if force_kernel is not None:
        maybe_forced_kernel, failure_reason = force_kernel.try_select(
            config, compute_capability
        )
        if maybe_forced_kernel:
            return maybe_forced_kernel

        logger.info_once(
            "Tried to force %s, but the kernel couldn't be selected",
            force_kernel.get_name(),
            scope="global",
        )
        failure_reason_list.extend(failure_reason)

    for kernel_module in possible_kernel_modules[current_platform._enum]:
        kernel_class = getattr(kernel_module, kernel_attr)
        maybe_kernel, failure_reason = kernel_class.try_select(
            config, compute_capability
        )
        if maybe_kernel:
            return maybe_kernel
        failure_reason_list.extend(failure_reason)

    raise ValueError(
        "Failed to find a kernel that can implement the w8a8 linear layer. "
        "Reasons: \n" + "\n".join(failure_reason_list)
    )


def create_w8a8_fp_kernel(
    activation_quant_key: QuantKey,
    weight_quant_key: QuantKey,
    out_dtype: torch.dtype,
    force_kernel: type[w8a8.FpKernel] | None = None,
    module_name: str | None = None,
) -> w8a8.FpKernel:
    """
    Initialize a w8a8 floating-point kernel for the given configuration.
    """
    config = w8a8.FpKernelConfig(
        weight_quant_key=weight_quant_key,
        activation_quant_key=activation_quant_key,
        out_dtype=out_dtype,
    )

    kernel_type = choose_w8a8_linear_kernel(
        config, _POSSIBLE_FP_W8A8_KERNEL_MODULES, "FpKernel", force_kernel=force_kernel
    )

    if module_name:
        logger.info_once(
            "Selected %s for %s",
            kernel_type.get_name(),
            module_name,
            scope="global",
        )

    return kernel_type(
        config,
        layer_param_names=["weight", "weight_scale", "input_scale", "input_scale_ub"],
    )


def create_w8a8_int_kernel(
    is_channelwise: bool,
    is_static_input_scheme: bool,
    input_symmetric: bool,
    module_name: str | None = None,
    force_kernel: type[w8a8.IntKernel] | None = None,
) -> w8a8.IntKernel:
    """
    Initialize a w8a8 integer kernel for the given configuration.
    """
    config = w8a8.IntKernelConfig(
        is_channelwise=is_channelwise,
        is_static_input_scheme=is_static_input_scheme,
        input_symmetric=input_symmetric,
    )

    kernel_type = choose_w8a8_linear_kernel(
        config,
        _POSSIBLE_INT_W8A8_KERNEL_MODULES,
        "IntKernel",
        force_kernel=force_kernel,
    )

    if module_name:
        logger.info_once(
            "Selected %s for %s",
            kernel_type.get_name(),
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


__all__ = [
    "create_w8a8_fp_kernel",
    "create_w8a8_int_kernel",
    "choose_mp_linear_kernel",
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
    "XPUwNa16LinearKernel",
]
