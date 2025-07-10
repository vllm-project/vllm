# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

try:
    import torch.distributed._symmetric_memory as torch_symm_mem

    symm_mem_available = True
except ImportError:
    symm_mem_available = False

logger = init_logger(__name__)


class SymmMemCommunicator:
    MB = 1024 * 1024
    # Max sizes for each world size
    _MAX_SIZES = {
        2: 8 * MB,
        4: 32 * MB,
        6: 64 * MB,
        8: 256 * MB,
    }

    def __init__(self, group: ProcessGroup, device: Union[int, str,
                                                          torch.device]):
        self.disabled = True

        if not symm_mem_available:
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
        if self.world_size not in self._MAX_SIZES:
            logger.warning(
                "SymmMemCommunicator: World size %d not supported, "
                "communicator is not available.",
                self.world_size,
            )
            return
        self.buffer = torch_symm_mem.empty(
            self._MAX_SIZES[self.world_size] // self.dtype.itemsize,
            device=self.device,
            dtype=self.dtype,
        )
        handle = torch_symm_mem.rendezvous(self.buffer, self.group.group_name)
        if handle.multicast_ptr == 0:
            logger.warning("SymmMemCommunicator: symmetric memory "
                           "multicast operations are not supported.")
            return
        self.disabled = False

    def should_use_symm_mem(self, inp: torch.Tensor):
        if self.disabled:
            return False
        if inp.dtype != self.dtype:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 4 != 0:
            return False
        return inp_size <= self._MAX_SIZES[self.world_size]

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
        if self.world_size in [2, 4]:
            # Use two-shot all-reduce for 2 and 4 GPUs
            torch.ops.symm_mem.two_shot_all_reduce_(self.buffer[:inp.numel()],
                                                    "sum",
                                                    self.group.group_name)
        else:
            # Use multi-mem all-reduce for 6 and 8 GPUs
            torch.ops.symm_mem.multimem_all_reduce_(self.buffer[:inp.numel()],
                                                    "sum",
                                                    self.group.group_name)
        out.copy_(self.buffer[:inp.numel()].view(out.shape))
        return out
