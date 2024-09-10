import os
from typing import List, Optional, Type

from vllm.platforms import current_platform

from .MacheteLinearKernel import MacheteLinearKernel
from .MarlinLinearKernel import MarlinLinearKernel
from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

# in priority/performance order (when available)
_POSSIBLE_KERNELS: List[Type[MPLinearKernel]] = [
    MacheteLinearKernel,
    MarlinLinearKernel,
]


def choose_mp_linear_kernel(
        config: MPLinearLayerConfig,
        compute_capability: Optional[int] = None) -> Type[MPLinearKernel]:
    if compute_capability is None:
        if current_platform is None:
            raise ValueError("Cannot determine compute capability")
        _cc = current_platform.get_device_capability()
        compute_capability = _cc[0] * 10 + _cc[1]

    failure_reasons = []
    for kernel in _POSSIBLE_KERNELS:
        if kernel.__name__ in os.environ.get("VLLM_DISABLED_KERNELS", "")\
            .split(","):
            failure_reasons.append(
                f' {kernel.__name__} disabled by environment variable')
            continue
        
        if kernel.get_min_capability() > compute_capability:
            failure_reasons.append(
                f"{kernel.__name__} requires capability "
                f"{kernel.get_min_capability()}, current compute capability "
                f"is {compute_capability}")

        can_implement, failure_reason = kernel.can_implement(config)
        if can_implement:
            return kernel
        else:
            failure_reasons.append(
                f' {kernel.__name__} cannot implement due to: {failure_reason}')

    raise ValueError(
        "Failed to find a kernel that can implement the "\
        "WNA16 linear layer. Reasons: \n"
        + '\n'.join(failure_reasons))
