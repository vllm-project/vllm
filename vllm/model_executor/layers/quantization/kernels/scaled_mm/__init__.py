from typing import List, Optional, Type

import vllm.envs as envs
from vllm.model_executor.layers.quantization.kernels.exllama import (
    ExllamaLinearKernel)
from vllm.model_executor.layers.quantization.kernels.machete import (
    MacheteLinearKernel)
from vllm.model_executor.layers.quantization.kernels.marlin import (
    MarlinLinearKernel)
from vllm.model_executor.layers.quantization.kernels.MPLinearKernel import (
    MPLinearKernel, MPLinearLayerConfig)
from vllm.platforms import current_platform

# in priority/performance order (when available)
_POSSIBLE_KERNELS: List[Type[MPLinearKernel]] = [
    MacheteLinearKernel,
    MarlinLinearKernel,
    ExllamaLinearKernel,
]


def choose_mp_linear_kernel(
        config: MPLinearLayerConfig,
        compute_capability: Optional[int] = None) -> Type[MPLinearKernel]:
    """
    Choose an MPLinearKernel that can implement the given config for the given
     compute capability. Attempts to choose the best kernel in terms of 
     performance.

    Args:
        config (MPLinearLayerConfig): Description of the linear layer to be 
          implemented.
        compute_capability (Optional[int], optional): The compute capability of
          the target device, if None uses `current_platform` to get the compute 
          capability. Defaults to None.

    Raises:
        ValueError: If no kernel can implement the given config.

    Returns:
        Type[MPLinearKernel]: Chosen kernel.
    """
    if compute_capability is None:
        if current_platform is None:
            raise ValueError("Cannot determine compute capability")
        _cc = current_platform.get_device_capability()
        compute_capability = _cc[0] * 10 + _cc[1]

    failure_reasons = []
    for kernel in _POSSIBLE_KERNELS:
        if kernel.__name__ in envs.VLLM_DISABLED_KERNELS:
            failure_reasons.append(
                f' {kernel.__name__} disabled by environment variable')
            continue

        if kernel.get_min_capability() > compute_capability:
            failure_reasons.append(
                f"{kernel.__name__} requires capability "
                f"{kernel.get_min_capability()}, current compute capability "
                f"is {compute_capability}")
            continue

        can_implement, failure_reason = kernel.can_implement(config)
        if can_implement:
            return kernel
        else:
            failure_reasons.append(
                f' {kernel.__name__} cannot implement due to: {failure_reason}'
            )

    raise ValueError(
        "Failed to find a kernel that can implement the "\
        "WNA16 linear layer. Reasons: \n"
        + '\n'.join(failure_reasons))
