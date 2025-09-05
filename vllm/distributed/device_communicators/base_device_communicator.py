# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from typing import Optional, Union
from weakref import WeakValueDictionary

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


class Cache:

    def __init__(self):
        self._cache: WeakValueDictionary = WeakValueDictionary()
        self._lock = threading.RLock()  # Reentrant lock for thread safety

    def get_or_create(self, kwargs, func):
        # Create a hashable key from the kwargs
        key = tuple(sorted((k, v) for k, v in kwargs.items()))

        with self._lock:
            instance = self._cache.get(key)
            if instance is None:
                instance = func(**kwargs)
                self._cache[key] = instance
            return instance


class All2AllManagerBase:

    def __init__(self, cpu_group):
        self.cpu_group = cpu_group

        # compute some common properties
        from vllm.distributed.parallel_state import (get_dp_group,
                                                     get_tp_group,
                                                     in_the_same_node_as)

        # all2all lives in ep group, which is merged from dp and tp group
        self.dp_group = get_dp_group()
        self.tp_group = get_tp_group()
        # no self.ep_group since self.ep_group is still in construction
        # when we create this object
        self.dp_rank = self.dp_group.rank_in_group
        self.dp_world_size = self.dp_group.world_size
        self.rank = dist.get_rank(cpu_group)
        self.world_size = dist.get_world_size(cpu_group)

        # all2all communication often has separate implementations for
        # intra-node and inter-node communication
        self.internode = not all(in_the_same_node_as(cpu_group, source_rank=0))

    def get_handle(self, kwargs):
        # get a handle for the all2all communication,
        # based on the kwargs.
        # different layers can have different configs,
        # e.g. one layer has hidden size 1024, another has 2048.
        # usually the underlying implementation caches the handle
        # and reuse it for the same config.
        raise NotImplementedError

    def dispatch(self, hidden_states: torch.Tensor,
                 router_logits: torch.Tensor):
        raise NotImplementedError

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def destroy(self):
        pass


class DeviceCommunicatorBase:
    """
    Base class for device-specific communicator.
    It can use the `cpu_group` to initialize the communicator.
    If the device has PyTorch integration (PyTorch can recognize its
    communication backend), the `device_group` will also be given.
    """

    def __init__(self,
                 cpu_group: ProcessGroup,
                 device: Optional[torch.device] = None,
                 device_group: Optional[ProcessGroup] = None,
                 unique_name: str = ""):
        self.device = device or torch.device("cpu")
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.unique_name = unique_name
        self.rank = dist.get_rank(cpu_group)
        self.world_size = dist.get_world_size(cpu_group)
        self.ranks = dist.get_process_group_ranks(cpu_group)
        self.global_rank = dist.get_rank()
        self.global_world_size = dist.get_world_size()
        self.rank_in_group = dist.get_group_rank(self.cpu_group,
                                                 self.global_rank)

        use_ep = False
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config()
        if config is not None:
            # as long as we use data parallel (coupled data parallel
            # where all data parallel ranks execute forward together),
            # we initialize the all2all manager used in expert parallel.
            use_ep = config.parallel_config.data_parallel_size > 1

        self.is_ep_communicator = "ep" in unique_name
        self.use_all2all = self.is_ep_communicator and use_ep
        self.all2all_manager: Optional[All2AllManagerBase] = None

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(input_, group=self.device_group)
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # NOTE: we have to use concat-style all-gather here,
        # stack-style all-gather has compatibility issues with
        # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * self.world_size, ) + input_size[1:]
        # Allocate output tensor.
        output_tensor = torch.empty(output_size,
                                    dtype=input_.dtype,
                                    device=input_.device)
        # All-gather.
        dist.all_gather_into_tensor(output_tensor,
                                    input_,
                                    group=self.device_group)
        # Reshape
        output_tensor = output_tensor.reshape((self.world_size, ) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(input_size[:dim] +
                                              (self.world_size *
                                               input_size[dim], ) +
                                              input_size[dim + 1:])
        return output_tensor

    def all_gatherv(
        self,
        input_: Union[torch.Tensor, list[torch.Tensor]],
        dim: int = 0,
        sizes: Optional[list[int]] = None
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        raise NotImplementedError

    def reduce_scatter(self,
                       input_: torch.Tensor,
                       dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size, ) + input_tensor.shape[1:]

        output_tensor = torch.empty(output_shape,
                                    dtype=input_tensor.dtype,
                                    device=input_tensor.device)

        # Perform reduce-scatter operation
        torch.distributed.reduce_scatter_tensor(output_tensor,
                                                input_tensor,
                                                group=self.device_group)

        # Reshape before returning
        return output_tensor.movedim(0, dim).contiguous()

    def reduce_scatterv(self,
                        input_: torch.Tensor,
                        dim: int = -1,
                        sizes: Optional[list[int]] = None) -> torch.Tensor:
        raise NotImplementedError

    def gather(self,
               input_: torch.Tensor,
               dst: int = 0,
               dim: int = -1) -> Optional[torch.Tensor]:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Allocate output tensor.
        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None
        # Gather.
        torch.distributed.gather(input_,
                                 gather_list,
                                 dst=self.ranks[dst],
                                 group=self.device_group)
        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    def send(self, tensor: torch.Tensor, dst: Optional[int] = None) -> None:
        """Sends a tensor to the destination rank in a blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
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
        torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self):
        pass

    def prepare_communication_buffer_for_model(self,
                                               model: torch.nn.Module) -> None:
        """
        Prepare the communication buffer for the model.
        """
        if not self.is_ep_communicator:
            return

        moe_modules = [
            module for module in model.modules()
            # TODO(bnell): Should use isinstance but can't.  Maybe search for
            # presence of quant_method.init_prepare_finalize?
            if (module.__class__.__name__ == "FusedMoE"
                or module.__class__.__name__ == "SharedFusedMoE")
        ]
        for module in moe_modules:
            module.quant_method.init_prepare_finalize(module)

    def dispatch(
            self, hidden_states: torch.Tensor,
            router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch the hidden states and router logits to the appropriate device.
        This is a no-op in the base class.
        """
        return hidden_states, router_logits

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Combine the hidden states and router logits from the appropriate device.
        This is a no-op in the base class.
        """
        return hidden_states
