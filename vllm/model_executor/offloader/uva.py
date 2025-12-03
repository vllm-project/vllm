# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UVA-based CPU offloading using Unified Virtual Addressing."""

from collections.abc import Callable, Generator

import torch
import torch.nn as nn
from torch.func import functional_call

from vllm.model_executor.offloader.base import BaseOffloader
from vllm.utils.platform_utils import is_pin_memory_available, is_uva_available
from vllm.utils.torch_utils import get_cuda_view_from_cpu_tensor


class UVAOffloader(BaseOffloader):
    """Offloader using Unified Virtual Addressing (UVA) for zero-copy access.

    This offloader moves parameters to pinned CPU memory and creates CUDA views
    using UVA. The GPU can then directly access the CPU memory without explicit
    transfers, at the cost of PCIe bandwidth (slower than GPU memory).

    Args:
        cpu_offload_max_bytes: Maximum bytes to offload to CPU.
    """

    def __init__(self, cpu_offload_max_bytes: int):
        assert is_uva_available(), "UVA offloading requires UVA (pin memory) support"

        self.cpu_offload_max_bytes = cpu_offload_max_bytes
        self.cpu_offload_bytes = 0

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
        submodule_accessor: Callable[[nn.Module], nn.Module] | None = None,
        whitelist_param_names_creator: Callable[[nn.Module], list[str]] | None = None,
    ) -> list[nn.Module]:
        """Wrap modules with UVA offloading.

        Note: UVA offloading operates at module level, so submodule_accessor
        and whitelist_param_names_creator are ignored.
        """
        return [self._maybe_offload_to_cpu(module) for module in modules_generator]

    def _maybe_offload_to_cpu(self, module: nn.Module) -> nn.Module:
        """Offload module parameters to CPU using UVA if budget allows."""
        if (params := next(module.parameters(), None)) is None:
            return module

        device = params.device

        if device == torch.device("cpu"):
            return module

        global _CPU_OFFLOAD_MAX_BYTES, _CPU_OFFLOAD_BYTES
        if _CPU_OFFLOAD_BYTES >= _CPU_OFFLOAD_MAX_BYTES:
            return module

        pin_memory = is_pin_memory_available()
        uva_available = is_uva_available()

        assert uva_available, "V1 CPU offloading requires uva (pin memory) support"
        uva_offloading = True

        # offload parameters to CPU
        # use pin_memory if possible, which helps cudagraph capture speed
        offloaded_parameters = False
        for p in module.parameters():
            if _CPU_OFFLOAD_BYTES >= _CPU_OFFLOAD_MAX_BYTES:
                # we use per-parameter offloading
                # one module might have some parameters offloaded and some not
                break

            # `torch.empty_like` does not support `pin_memory` argument
            cpu_data = torch.empty_strided(
                size=p.data.size(),
                stride=p.data.stride(),
                dtype=p.data.dtype,
                layout=p.data.layout,
                device="cpu",
                pin_memory=pin_memory,
            )
            cpu_data.copy_(p.data)
            if not uva_offloading:
                p.data = cpu_data
            else:
                # keep the cpu data alive
                p._vllm_offloaded_cpu_data = cpu_data
                p.data = get_cuda_view_from_cpu_tensor(cpu_data)
            _CPU_OFFLOAD_BYTES += p.data.numel() * p.data.element_size()
            offloaded_parameters = True

        if offloaded_parameters and not uva_offloading:
            original_forward = module.forward

            def forward(*args, **kwargs):
                module.forward = original_forward
                device_state = {
                    # here we blindly call `to(device)`
                    # if the parameter is already on the device, it will be a no-op
                    k: v.to(device, non_blocking=True)
                    for k, v in module.state_dict().items()
                }
                output = functional_call(module, device_state, args=args, kwargs=kwargs)
                module.forward = forward
                return output

            module.forward = forward

        return module


# Backward compatibility: Global state for legacy set_cpu_offload_max_bytes()
_CPU_OFFLOAD_BYTES = 0
_CPU_OFFLOAD_MAX_BYTES = 0


def set_cpu_offload_max_bytes(max_bytes: int) -> None:
    """Set maximum bytes to offload for legacy UVA offloading.

    Deprecated: Use UVAOffloader class directly.
    """
    global _CPU_OFFLOAD_MAX_BYTES, _CPU_OFFLOAD_BYTES
    _CPU_OFFLOAD_BYTES = 0
    _CPU_OFFLOAD_MAX_BYTES = max_bytes
