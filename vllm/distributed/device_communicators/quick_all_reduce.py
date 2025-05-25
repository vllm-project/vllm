# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import cuda_device_count_stateless

try:
    ops.is_quickreduce_available()
    quick_ar = True
except Exception:
    # For CPUs
    quick_ar = False

logger = init_logger(__name__)


def is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (inp.storage().nbytes() -
                                   inp.storage_offset() * inp.element_size()
                                   == inp.numel() * inp.element_size())


'''
quantization level & int
ONESHOT_F16 = 0,
TWOSHOT_F16 = 1,
TWOSHOT_FP8 = 2,
TWOSHOT_Q8 = 3,
TWOSHOT_Q6 = 4,
TWOSHOT_Q4 = 5,
'''


class QuickAllreduce:

    _SUPPORTED_WORLD_SIZES = [2, 4, 8]
    _SUPPORTED_LEVEL = [0, 1, 2, 3, 4, 5]

    def __init__(self,
                 group: ProcessGroup,
                 device: Union[int, str, torch.device],
                 max_size=1024 * 1024 * 512,
                 min_size=1024 * 1024) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the QuickAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
            max_size: max supported size.
            min_size: Less than this size, custom_allreduce is better.
            (custom_allreduce is available when less than 16MB)
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self.disabled = True
        assert envs.VLLM_QUICK_ALLREDUCE in QuickAllreduce._SUPPORTED_LEVEL, (
            "quick allreduce level must be in [0, 1, 2, 3, 4, 5], "
            f"but got {envs.VLLM_QUICK_ALLREDUCE}")

        if not quick_ar:
            # disable because of missing quick allreduce library
            # e.g. in a non-GPU environment
            logger.info("Quick allreduce is disabled because "
                        "of missing quick allreduce library")
            return

        self.group = group

        assert dist.get_backend(group) != dist.Backend.NCCL, (
            "QuickAllreduce should be attached to a non-NCCL group.")

        if not all(in_the_same_node_as(group, source_rank=0)):
            # No need to initialize quick allreduce for multi-node case.
            logger.warning(
                "Quick allreduce is disabled because this process group"
                " spans across nodes.")
            return

        rank = dist.get_rank(group=self.group)
        self.rank = rank
        world_size = dist.get_world_size(group=self.group)
        self.world_size = world_size

        if world_size not in QuickAllreduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "Quick allreduce is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_quick_all_reduce=0 explicitly.",
                world_size, str(QuickAllreduce._SUPPORTED_WORLD_SIZES))
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device

        cuda_visible_devices = envs.CUDA_VISIBLE_DEVICES
        if cuda_visible_devices:
            device_ids = list(map(int, cuda_visible_devices.split(",")))
        else:
            device_ids = list(range(cuda_device_count_stateless()))

        physical_device_id = device_ids[device.index]
        tensor = torch.tensor([physical_device_id],
                              dtype=torch.int,
                              device="cpu")
        gather_list = [
            torch.tensor([0], dtype=torch.int, device="cpu")
            for _ in range(world_size)
        ]
        dist.all_gather(gather_list, tensor, group=self.group)
        physical_device_ids = [t.item() for t in gather_list]

        # test nvlink first, this will filter out most of the cases
        # where quick allreduce is not supported
        # this checks hardware and driver support for NVLink
        assert current_platform.is_cuda_alike()
        fully_connected = current_platform.is_fully_connected(
            physical_device_ids)
        if not fully_connected:
            logger.warning(
                "Quick allreduce is disabled because it's not supported on"
                " more than two PCIe-only GPUs. To silence this warning, "
                "specify disable_quick_all_reduce=0 explicitly.")
            return

        self.max_size = (max_size if envs.VLLM_QUICK_ALLREDUCE > 0 else
                         max_size / self.world_size * 2)
        self.min_size = min_size
        self._ptr = ops.init_quick_ar(world_size, rank)
        my_handle = ops.qr_get_comm_handle(self._ptr)

        all_handles = [[None] for _ in range(world_size)]
        all_handles[rank][0] = my_handle

        for src in range(world_size):
            dist.broadcast_object_list(all_handles[src], src=src)
        comm_handles = [h[0] for h in all_handles]
        ops.qr_set_comm_handles(self._ptr, comm_handles)
        self.disabled = False

    def should_quick_ar(self, inp: torch.Tensor):
        '''
        Check if quickreduce is available
        '''
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # quick allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        if inp.dtype == torch.float16:
            return inp_size <= self.max_size and inp_size > self.min_size
        elif inp.dtype == torch.bfloat16:
            return inp_size <= self.max_size and inp_size > 1024 * 1024 * 16 \
                  and self.world_size == 2
        return False

    def all_reduce(self, inp: torch.Tensor, *, out: torch.Tensor = None):
        """Performs an out-of-place all reduce."""
        if out is None:
            out = torch.empty_like(inp)
        ops.qr_all_reduce(self._ptr, envs.VLLM_QUICK_ALLREDUCE, inp, out)
        return out

    def quick_all_reduce(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        """The main allreduce API that provides support for cuda graph."""
        # When quick allreduce is disabled, this will be None.
        if self.disabled or not self.should_quick_ar(input):
            return None

        return self.all_reduce(input)

    def close(self):
        '''del self._ptr and del buffer'''
        if not self.disabled and self._ptr:
            ops.qr_destroy(self._ptr)
            self._ptr = 0

    def __del__(self):
        self.close()
