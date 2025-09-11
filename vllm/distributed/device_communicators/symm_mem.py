# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.distributed.device_communicators.all_reduce_utils import (
    SYMM_MEM_ALL_REDUCE_MAX_SIZES)
from vllm.logger import init_logger
from vllm.platforms import current_platform

try:
    import torch.distributed._symmetric_memory as torch_symm_mem

    symm_mem_available = True
except ImportError:
    symm_mem_available = False

logger = init_logger(__name__)


class SymmMemCommunicator:
    _WORLD_SIZES_MULTIMEM = {
        "9.0": [4, 6, 8],
        "10.0": [6, 8],
    }

    def __init__(
            self,
            group: ProcessGroup,
            device: Union[int, str, torch.device],
            # add options for testing
            force_multimem: Optional[bool] = None,
            max_size_override: Optional[int] = None):
        self.disabled = True

        if not symm_mem_available:
            return

        if not current_platform.is_cuda():
            logger.warning("SymmMemCommunicator: symmetric "
                           "memory is not available.")
            return
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        torch.cuda.set_device(device)
        self.dtype = torch.bfloat16
        self.device = device
        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.device_capability = current_platform.get_device_capability(
        ).as_version_str()
        if self.device_capability not in SYMM_MEM_ALL_REDUCE_MAX_SIZES:
            logger.warning(
                "SymmMemCommunicator: Device capability %s not supported, "
                "communicator is not available.",
                self.device_capability,
            )
            return
        if self.world_size not in SYMM_MEM_ALL_REDUCE_MAX_SIZES[
                self.device_capability]:
            logger.warning(
                "SymmMemCommunicator: World size %d not supported, "
                "communicator is not available.",
                self.world_size,
            )
            return
        # Use override max_size if provided, otherwise use default
        if max_size_override is not None:
            self.max_size = max_size_override
            logger.info(
                "SymmMemCommunicator: Using override max_size: %s bytes",
                self.max_size,
            )
        else:
            self.max_size = SYMM_MEM_ALL_REDUCE_MAX_SIZES[
                self.device_capability][self.world_size]

        self.buffer = torch_symm_mem.empty(
            self.max_size // self.dtype.itemsize,
            device=self.device,
            dtype=self.dtype,
        )
        handle = torch_symm_mem.rendezvous(self.buffer, self.group.group_name)
        if handle.multicast_ptr == 0:
            logger.warning("SymmMemCommunicator: symmetric memory "
                           "multicast operations are not supported.")
            return
        self.force_multimem = force_multimem
        self.disabled = False

    def should_use_symm_mem(self, inp: torch.Tensor):
        if self.disabled:
            return False
        if inp.dtype != self.dtype:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 4 != 0:
            return False
        return inp_size < self.max_size

    def all_reduce(
            self,
            inp: torch.Tensor,
            *,
            out: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if not self.should_use_symm_mem(inp):
            return None
        if out is None:
            out = torch.empty_like(inp)
        self.buffer[:inp.numel()].copy_(inp.view(-1))

        # Determine which algorithm to use
        use_multimem = False
        if self.force_multimem is not None:
            # Test override: use forced setting
            use_multimem = self.force_multimem
        else:
            # Normal logic: use multimem for supported world sizes
            use_multimem = self.world_size in self._WORLD_SIZES_MULTIMEM[
                self.device_capability]

        if use_multimem:
            torch.ops.symm_mem.multimem_all_reduce_(self.buffer[:inp.numel()],
                                                    "sum",
                                                    self.group.group_name)
        else:
            torch.ops.symm_mem.two_shot_all_reduce_(self.buffer[:inp.numel()],
                                                    "sum",
                                                    self.group.group_name)
        out.copy_(self.buffer[:inp.numel()].view(out.shape))
        return out
