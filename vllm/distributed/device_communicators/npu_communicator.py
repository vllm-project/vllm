import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.platforms import current_platform


class NpuCommunicator:

    def __init__(self, group: ProcessGroup):
        if not current_platform.is_npu():
            self.disabled = True
            return
        self.disabled = False
        self.group = group
        self.world_size = dist.get_world_size(self.group)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(x, group=self.group)
        return x

    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        if dim < 0:
            # Convert negative dim to positive.
            dim += x.dim()
        input_size = x.size()
        # Allocate output tensor.
        output_tensor = torch.empty((world_size, ) + input_size,
                                    dtype=x.dtype,
                                    device=x.device)
        # All-gather.
        dist.all_gather_into_tensor(output_tensor, x, group=self.group)
        # Reshape
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(input_size[:dim] +
                                              (world_size *
                                               input_size[dim], ) +
                                              input_size[dim + 1:])
        return output_tensor

    def gather(self,
               input_: torch.Tensor,
               rank_in_group: int,
               dst: int = 0,
               dim: int = -1):
        # For xpu path, gather doesn't work properly together with ray
        # cluster so we use all_gather instead for now.
        input_size = input_.size()
        # Allocate output tensor.
        output_tensor = torch.empty((self.world_size, ) + input_size,
                                    dtype=input_.dtype,
                                    device=input_.device)
        # All-gather.
        dist.all_gather_into_tensor(output_tensor, input_, group=self.group)
        if rank_in_group == dst:
            # Reshape
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(input_size[:dim] +
                                                  (self.world_size *
                                                   input_size[dim], ) +
                                                  input_size[dim + 1:])
        else:
            output_tensor = None
        return output_tensor
