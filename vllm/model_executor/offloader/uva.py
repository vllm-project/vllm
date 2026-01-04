# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UVA-based CPU offloading using Unified Virtual Addressing."""

from collections.abc import Callable, Generator

import torch
import torch.nn as nn

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

        if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
            return module

        pin_memory = is_pin_memory_available()

        # offload parameters to CPU
        # use pin_memory if possible, which helps cudagraph capture speed
        for p in module.parameters():
            if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
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
            # keep the cpu data alive
            p._vllm_offloaded_cpu_data = cpu_data
            p.data = get_cuda_view_from_cpu_tensor(cpu_data)
            self.cpu_offload_bytes += p.data.numel() * p.data.element_size()

        return module
