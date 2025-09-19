# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

import torch
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .base_device_communicator import DeviceCommunicatorBase

logger = init_logger(__name__)


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

        # ep does not use pynccl
        use_pynccl = "ep" not in unique_name

        self.use_pynccl = use_pynccl
        self.use_custom_allreduce = use_custom_allreduce

        # lazy import to avoid documentation build error
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce)
        from vllm.distributed.device_communicators.pynccl import (
            PyNcclCommunicator)
        from vllm.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce)
        from vllm.distributed.device_communicators.symm_mem import (
            SymmMemCommunicator)

        self.pynccl_comm: Optional[PyNcclCommunicator] = None
        if use_pynccl and self.world_size > 1:
            self.pynccl_comm = PyNcclCommunicator(
                group=self.cpu_group,
                device=self.device,
            )

        self.ca_comm: Optional[CustomAllreduce] = None
        self.qr_comm: Optional[QuickAllReduce] = None
        self.symm_mem_comm: Optional[SymmMemCommunicator] = None
        if envs.VLLM_ALLREDUCE_USE_SYMM_MEM and current_platform.is_cuda():
            self.symm_mem_comm = SymmMemCommunicator(
                group=self.cpu_group,
                device=self.device,
            )

        if use_custom_allreduce and self.world_size > 1:
            # Initialize a custom fast all-reduce implementation.
            self.ca_comm = CustomAllreduce(
                group=self.cpu_group,
                device=self.device,
                symm_mem_enabled=(self.symm_mem_comm is not None
                                  and not self.symm_mem_comm.disabled),
            )

            if current_platform.is_rocm():
                # Initialize a custom quick all-reduce implementation for AMD.
                # Quick reduce is designed as a complement to custom allreduce.
                # Based on quickreduce (https://github.com/mk1-project/quickreduce).
                # If it's a rocm, 'use_custom_allreduce==True' means it must
                # currently be an MI300 series.
                self.qr_comm = QuickAllReduce(group=self.cpu_group,
                                              device=self.device)

        if self.use_all2all:
            all2all_backend = envs.VLLM_ALL2ALL_BACKEND
            if all2all_backend == "naive":
                from .all2all import NaiveAll2AllManager
                self.all2all_manager = NaiveAll2AllManager(self.cpu_group)
                logger.info("Using naive all2all manager.")
            elif all2all_backend == "allgather_reducescatter":
                from .all2all import AgRsAll2AllManager
                self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
                logger.info("Using AllGather-ReduceScatter all2all manager.")
            elif all2all_backend == "pplx":
                from .all2all import PPLXAll2AllManager
                self.all2all_manager = PPLXAll2AllManager(self.cpu_group)
                logger.info("Using PPLX all2all manager.")
            elif all2all_backend == "deepep_high_throughput":
                from .all2all import DeepEPHTAll2AllManager
                self.all2all_manager = DeepEPHTAll2AllManager(self.cpu_group)
                logger.info("Using DeepEP High-Throughput all2all manager.")
            elif all2all_backend == "deepep_low_latency":
                from .all2all import DeepEPLLAll2AllManager
                self.all2all_manager = DeepEPLLAll2AllManager(self.cpu_group)
                logger.info("Using DeepEP Low-Latency all2all manager.")
            else:
                raise ValueError(f"Unknown all2all backend: {all2all_backend}")

    def all_reduce(self, input_):
        # always try quick reduce first, then custom allreduce,
        # and then pynccl. (quick reduce just for ROCM MI3*)
        qr_comm = self.qr_comm
        if qr_comm is not None and not qr_comm.disabled and \
            qr_comm.should_quick_allreduce(input_):
            out = qr_comm.quick_all_reduce(input_)
            assert out is not None
            return out
        ca_comm = self.ca_comm
        if ca_comm is not None and not ca_comm.disabled and \
            ca_comm.should_custom_ar(input_):
            out = ca_comm.custom_all_reduce(input_)
            assert out is not None
            return out
        symm_mem_comm = self.symm_mem_comm
        if symm_mem_comm is not None and \
            symm_mem_comm.should_use_symm_mem(input_):
            out = symm_mem_comm.all_reduce(input_)
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

        pynccl_comm.reduce_scatter(output, input_tensor)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def reduce_scatterv(self,
                        input_: torch.Tensor,
                        dim: int = -1,
                        sizes: Optional[list[int]] = None):
        world_size = self.world_size
        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        if sizes is not None:
            assert len(sizes) == world_size
            assert input_tensor.shape[0] == sum(sizes)
            chunk_size = sizes[self.rank_in_group]
        else:
            assert input_tensor.shape[0] % world_size == 0
            chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size, ) + input_tensor.shape[1:]

        output = torch.empty(output_shape,
                             dtype=input_tensor.dtype,
                             device=input_tensor.device)

        if sizes is not None:
            pynccl_comm.reduce_scatterv(output, input_tensor, sizes=sizes)
        else:
            pynccl_comm.reduce_scatter(output, input_tensor)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def send(self, tensor: torch.Tensor, dst: Optional[int] = None) -> None:
        """Sends a tensor to the destination rank in a blocking way"""
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
        if self.all2all_manager is not None:
            self.all2all_manager.destroy()
            self.all2all_manager = None

    def all_gatherv(self,
                    input_: Union[torch.Tensor, list[torch.Tensor]],
                    dim: int = 0,
                    sizes: Optional[list[int]] = None):
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        world_size = self.world_size
        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None and not pynccl_comm.disabled

        # 'sizes' is not needed if all inputs in the same group have the same
        # shape
        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

        def _all_gather_single(input_: torch.Tensor,
                               sizes: Optional[list[int]] = None):
            input_size = input_.size()
            if sizes is not None:
                assert len(sizes) == world_size
                assert input_.shape[dim] == sizes[self.rank_in_group], (
                    f"{input_.shape[dim]} != {sizes[self.rank_in_group]}")
                output_size = (sum(sizes), ) + input_size[1:]
            else:
                output_size = (input_size[0] * world_size, ) + input_size[1:]
            # Allocate output tensor.
            output_tensor = torch.empty(output_size,
                                        dtype=input_.dtype,
                                        device=input_.device)
            if sizes is not None:
                pynccl_comm.all_gatherv(output_tensor, input_, sizes=sizes)
            else:
                pynccl_comm.all_gather(output_tensor, input_)
            return output_tensor

        if isinstance(input_, torch.Tensor):
            return _all_gather_single(input_, sizes)

        output_list = []
        pynccl_comm.group_start()
        for inp in input_:
            output_list.append(_all_gather_single(inp, sizes=sizes))
        pynccl_comm.group_end()

        return output_list

    def dispatch(
            self, hidden_states: torch.Tensor,
            router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.all2all_manager is not None
        hidden_states, router_logits = self.all2all_manager.dispatch(
            hidden_states, router_logits)
        return hidden_states, router_logits

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.all2all_manager is not None
        hidden_states = self.all2all_manager.combine(hidden_states)
        return hidden_states
