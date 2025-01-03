from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


class CommunicatorBase:
    """
    Base class for device-specific communicator.

    The class provides primary communication ops that are frequently
    overridden by devices.

    TODO (MengqingCao): make `PyNcclCommunicator` a child class 
                        of `CommunicatorBase`.
    """

    def __init__(self,
                 group: ProcessGroup,
                 rank_in_group: int,
                 ranks: list[int],
                 unique_name: str = ""):
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(self.group)
        self.device_communicator = None
        self.rank_in_group = rank_in_group
        self.ranks = ranks
        self.unique_name = unique_name

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """
        All-reduce function only for cpu and gpu.
        
        NOTE: `torch.ops.vllm.all_reduce` is registered in
            `vllm/distributed/parallel_state.py`
        """
        if input_.is_cpu:
            import intel_extension_for_pytorch as ipex
            ipex.distributed.all_reduce(input_, group=self.group)
            return input_

        return torch.ops.vllm.all_reduce(input_, group_name=self.unique_name)

    def gather(self,
               input_: torch.Tensor,
               dst: int = 0,
               dim: int = -1) -> Optional[torch.Tensor]:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """

        # Allocate output tensor.
        if self.rank_in_group == dst:
            gather_list = [
                torch.empty_like(input_) for _ in range(self.world_size)
            ]
        else:
            gather_list = None
        # Gather.
        dist.gather(input_, gather_list, dst=self.ranks[dst], group=self.group)
        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

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
        dist.all_gather_into_tensor(output_tensor, input_, group=self.group)
        # Reshape
        output_tensor = output_tensor.reshape((self.world_size, ) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(input_size[:dim] +
                                              (self.world_size *
                                               input_size[dim], ) +
                                              input_size[dim + 1:])
        return output_tensor
