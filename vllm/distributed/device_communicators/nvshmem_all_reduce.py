# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import cuda_device_count_stateless

try:
    from flashinfer.comm.nvshmem_allreduce import NVSHMEMAllReduce
    nvshmem_ar = True
except ImportError:
    nvshmem_ar = False

logger = init_logger(__name__)


class NVSHMEMAllreduce:

    # max_size: max supported allreduce size
    def __init__(self,
                 group: ProcessGroup,
                 device: Union[int, str, torch.device],
                 dtype: torch.dtype = torch.float16,
                 max_size=8192 * 1024) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the NVSHMEMAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
            dtype: the dtype of the allreduce.
            max_size: the max supported allreduce size. This is used to allocate
                memory in nvshmem symm heap. set to the largest tensor size you
                will be reducing.
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same NVLINK domain.
        """
        self._IS_CAPTURING = False
        self.disabled = True

        if not nvshmem_ar:
            # disable because of missing flashinfer NVSHMEM allreduce import
            # e.g. in a non-GPU environment
            logger.info("NVSHMEM allreduce is disabled because "
                        "of missing flashinfer NVSHMEM allreduce import")
            return

        self.group = group

        # TODO(asamani): check both NCCL and GLOO are in dist backend
        # assert dist.get_backend(group) != dist.Backend.NCCL, (
        #     "CustomAllreduce should be attached to a non-NCCL group.")

        rank = dist.get_rank(group=self.group)
        self.rank = rank
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            # No need to initialize NVSHMEM allreduce for single GPU case.
            return

        # TODO(asamani): check if all gpus are in the same NVLINK domain

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
        # where custom allreduce is not supported
        # this checks hardware and driver support for NVLink
        assert current_platform.is_cuda_alike()
        fully_connected = current_platform.is_fully_connected(
            physical_device_ids)
        if not fully_connected:
            logger.warning(
                "NVSHMEM allreduce is disabled because it's not supported on"
                "non NVLINK domain only groups. To silence this warning, "
                "specify disable_nvshmem_all_reduce=True explicitly.")
            #TODO(asamani): add a flag to disable this warning
            return

        self.disabled = False

        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size
        self.fully_connected = fully_connected
        self.dtype = dtype
        self.nvshmem_allreduce = NVSHMEMAllReduce(
            self.rank,
            self.world_size,
            self.max_size,
            self.dtype,
            self.device,
            self.group,
        )

    # TODO(asamani): how to work with cuda graph?
    # @contextmanager
    # def capture(self):
    #     """
    #     The main responsibility of this context manager is the
    #     `register_graph_buffers` call at the end of the context.
    #     It records all the buffer addresses used in the CUDA graph.
    #     """
    #     try:
    #         self._IS_CAPTURING = True
    #         yield
    #     finally:
    #         self._IS_CAPTURING = False
    #         if not self.disabled:
    #             self.register_graph_buffers()

    def should_nvshmem_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        # TODO(asamani): check if this is needed
        #inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        # if inp_size % 16 != 0:
        #     return False
        if not inp.is_contiguous():
            return False
        if inp.numel() > self.max_size:
            return False
        if inp.dtype != self.dtype:
            return False
        return True

    def all_reduce(self, inp: torch.Tensor, *, out: torch.Tensor = None):
        """Performs an out-of-place all reduce.
        
        """
        if out is None:
            out = torch.empty_like(inp)
        self.nvshmem_allreduce.all_reduce(inp, out)
        return out

    def nvshmem_all_reduce(self,
                           input: torch.Tensor) -> Optional[torch.Tensor]:
        """The main allreduce API that provides support for cuda graph."""
        # TODO(asamani): check if this works with nvshmem
        # TODO(asamani): check if we can add output too!
        # When custom allreduce is disabled, this will be None.
        if self.disabled or not self.should_nvshmem_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input)
            else:
                # If warm up, mimic the allocation pattern since custom
                # allreduce is out-of-place.
                return torch.empty_like(input)
        else:
            # Note: outside of cuda graph context, custom allreduce incurs a
            # cost of cudaMemcpy, which should be small (<=1% of overall
            # latency) compared to the performance gain of using custom kernels
            return self.all_reduce(input)

    def close(self):
        if not self.disabled:
            self.nvshmem_allreduce.shutdown()

    # TODO(asamani): check if I need to have this
    # def __del__(self):
    #     self.close()
