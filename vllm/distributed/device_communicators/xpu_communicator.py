import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.base_communicator import (
    CommunicatorBase)


class XpuCommunicator(CommunicatorBase):

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(input_, group=self.group)
        return input_

    def gather(self, input_: torch.Tensor, dst: int = 0, dim: int = -1):
        # For xpu path, gather doesn't work properly together with ray
        # cluster so we use all_gather instead for now.
        input_size = input_.size()
        # Allocate output tensor.
        output_tensor = torch.empty((self.world_size, ) + input_size,
                                    dtype=input_.dtype,
                                    device=input_.device)
        # All-gather.
        torch.distributed.all_gather_into_tensor(output_tensor,
                                                 input_,
                                                 group=self.group)
        if self.rank_in_group == dst:
            # Reshape
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(input_size[:dim] +
                                                  (self.world_size *
                                                   input_size[dim], ) +
                                                  input_size[dim + 1:])
        else:
            output_tensor = None
        return output_tensor
