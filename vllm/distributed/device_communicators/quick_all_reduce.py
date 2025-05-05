# SPDX-License-Identifier: Apache-2.0
import logging
from enum import Enum
from typing import Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm import _custom_ops as ops

logger = logging.getLogger(__name__)

try:
    ops.meta_size()
    ops_available = True
except Exception:
    # For CPUs
    ops_available = False


class QuickReduceAlgo(Enum):
    OneShot = 0
    TwoShot = 1
    TwoShot_FP8 = 2
    TwoShot_Q8 = 3
    TwoShot_Q6 = 4
    TwoShot_Q4 = 5
    TwoShot_MAX_MIN_Q8 = 6


class QuickAllReduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    def __init__(self,
                 group: ProcessGroup,
                 device: Union[int, str, torch.device],
                 algo: QuickReduceAlgo = QuickReduceAlgo.TwoShot) -> None:
        self.disabled = True
        if not ops_available:
            # disable because of missing custom allreduce library
            # e.g. in a non-cuda environment
            return
        self.max_size = ops.qr_max_size()
        self.group = group
        self.algo = algo

        assert dist.get_backend(group) != dist.Backend.NCCL, (
            "QuickReduce should be attached to a non-NCCL group.")

        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            # No need to initialize custom allreduce for single GPU case.
            return

        if world_size not in QuickAllReduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "QuickReduce allreduce is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly.",
                world_size, str(QuickAllReduce._SUPPORTED_WORLD_SIZES))
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        torch.cuda.set_device(self.device)

        self.disabled = False
        self._ptr = ops.init_custom_qr(rank, world_size)
        self.create_shared_buffer()

    def create_shared_buffer(self):
        """
        Creates a shared buffer for quickreduce. 
        Has to be called after qr_init_device_collectives
        """
        handle = ops.qr_get_handle(self._ptr)
        world_size = dist.get_world_size(group=self.group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=self.group)
        ops.qr_open_handles(self._ptr, handles)

    def all_reduce(self, inp: torch.Tensor, *, out: torch.Tensor = None):
        """Performs an out-of-place all reduce.
        
        If registered is True, this assumes inp's pointer is already
        IPC-registered. Otherwise, inp is first copied into a pre-registered
        buffer.
        """
        inp_size = inp.numel() * inp.element_size()
        if inp_size >= self.max_size:
            return None

        if out is None:
            out = torch.empty_like(inp)

        ops.qr_all_reduce(self._ptr, inp, out, self.algo.value)
        return out

    def is_enabled(self):
        return not self.disabled

    @staticmethod
    def is_available():
        return ops_available

    def close(self):
        if not self.disabled and getattr(self, "_ptr", None):
            ops.qr_destroy(self._ptr)
            self._ptr = 0

    def __del__(self):
        self.close()

    def should_quick_allreduce(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        return inp.dtype == torch.float16 and inp_size < self.max_size
