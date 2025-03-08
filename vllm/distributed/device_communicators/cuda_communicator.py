# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
from torch.distributed import ProcessGroup

from .base_device_communicator import DeviceCommunicatorBase


class CudaCommunicator(DeviceCommunicatorBase):

    def __init__(self,
                 cpu_group: ProcessGroup,
                 device: Optional[torch.device] = None,
                 device_group: Optional[ProcessGroup] = None,
                 unique_name: str = ""):
        super().__init__(cpu_group, device, device_group, unique_name)
        if "tp" not in unique_name:
            # only tp uses custom allreduce
            use_custom_allreduce = False
        else:
            from vllm.distributed.parallel_state import (
                _ENABLE_CUSTOM_ALL_REDUCE)
            use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE
        use_pynccl = True

        self.use_pynccl = use_pynccl
        self.use_custom_allreduce = use_custom_allreduce

        # lazy import to avoid documentation build error
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce)
        from vllm.distributed.device_communicators.pynccl import (
            PyNcclCommunicator)

        self.pynccl_comm: Optional[PyNcclCommunicator] = None
        if use_pynccl and self.world_size > 1:
            self.pynccl_comm = PyNcclCommunicator(
                group=self.cpu_group,
                device=self.device,
            )

        self.ca_comm: Optional[CustomAllreduce] = None
        if use_custom_allreduce and self.world_size > 1:
            # Initialize a custom fast all-reduce implementation.
            self.ca_comm = CustomAllreduce(
                group=self.cpu_group,
                device=self.device,
            )

    def all_reduce(self, input_):
        # always try custom allreduce first,
        # and then pynccl.
        ca_comm = self.ca_comm
        if ca_comm is not None and not ca_comm.disabled and \
            ca_comm.should_custom_ar(input_):
            out = ca_comm.custom_all_reduce(input_)
            assert out is not None
            return out
        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None
        out = pynccl_comm.all_reduce(input_)
        if out is None:
            # fall back to the default all-reduce using PyTorch.
            # this usually happens during testing.
            # when we run the model, allreduce only happens for the TP
            # group, where we always have either custom allreduce or pynccl.
            out = input_.clone()
            torch.distributed.all_reduce(out, group=self.device_group)
        return out

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):
        world_size = self.world_size
        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size, ) + input_tensor.shape[1:]

        output = torch.empty(output_shape,
                             dtype=input_tensor.dtype,
                             device=input_tensor.device)

        pynccl_comm.reduce_scatter(output, input_)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def send(self, tensor: torch.Tensor, dst: Optional[int] = None) -> None:
        """Sends a tensor to the destination rank in a non-blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.send(tensor, dst)
        else:
            torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(self,
             size: torch.Size,
             dtype: torch.dtype,
             src: Optional[int] = None) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.recv(tensor, src)
        else:
            torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self):
        if self.pynccl_comm is not None:
            self.pynccl_comm = None
        if self.ca_comm is not None:
            self.ca_comm = None
