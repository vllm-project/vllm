# SPDX-License-Identifier: Apache-2.0

import torch
import torch.distributed as dist

from vllm.platforms import current_platform

from .base_device_communicator import DeviceCommunicatorBase

if current_platform.is_hpu():
    import habana_frameworks.torch as htorch  # noqa: F401


class HpuCommunicator(DeviceCommunicatorBase):

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # FIXME(kzawora): this is a workaround for a bug in Habana PT bridge
        # occurring when PT_HPU_ENABLE_LAZY_COLLECTIVES=true env var is used
        # (which is required for tensor parallel HPUGraph inference)
        htorch.core.mark_step()
        dist.all_reduce(input_, group=self.device_group)
        return input_

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
        htorch.core.mark_step()
        dist.all_gather_into_tensor(output_tensor, x, group=self.device_group)
        # Reshape
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(input_size[:dim] +
                                              (world_size *
                                               input_size[dim], ) +
                                              input_size[dim + 1:])
        return output_tensor
