# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import vllm.envs as envs
from vllm.model_executor.layers.quantization.kernels.mixed_precision.allspark import (  # noqa: E501
    AllSparkLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.conch import (  # noqa: E501
    ConchLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.cpu import (  # noqa: E501
    CPUWNA16LinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.cutlass import (  # noqa: E501
    CutlassW4A8LinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.dynamic_4bit import (  # noqa: E501
    Dynamic4bitLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.exllama import (  # noqa: E501
    ExllamaLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.machete import (  # noqa: E501
    MacheteLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.marlin import (  # noqa: E501
    MarlinLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.MPLinearKernel import (  # noqa: E501
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.xpu import (  # noqa: E501
    XPUwNa16LinearKernel,
)
from vllm.platforms import current_platform

# in priority/performance order (when available)
_POSSIBLE_KERNELS: list[type[MPLinearKernel]] = [
    CutlassW4A8LinearKernel,
    MacheteLinearKernel,
    AllSparkLinearKernel,
    MarlinLinearKernel,
    Dynamic4bitLinearKernel,
    ConchLinearKernel,
    ExllamaLinearKernel,
    XPUwNa16LinearKernel,
    CPUWNA16LinearKernel,
]


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
    for kernel in _POSSIBLE_KERNELS:
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
