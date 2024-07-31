import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.platforms import current_platform

if current_platform.is_tpu():
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    from torch_xla._internal import pjrt


class TpuCommunicator:

    def __init__(self, group: ProcessGroup):
        if not current_platform.is_tpu():
            self.disabled = True
            return
        self.disabled = False

        local_rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        pjrt.initialize_multiprocess(local_rank, world_size)
        xr._init_world_size_ordinal()

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return xm.all_reduce(xm.REDUCE_SUM, x)

    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == -1, "TPUs only support dim=-1 for all-gather."
        return xm.all_gather(x, dim=dim)
